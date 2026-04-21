"""RLM (Recursive Language Model) factory for secretagent.

Implements the REPL-based approach from Zhang, Kraska & Khattab
(arXiv:2512.24601): the LLM writes Python code in a persistent REPL
that can recursively call llm_query() on sub-pieces of its input.
"""

import contextlib
import io
import re
import traceback

from string import Template
from typing import Any

from pydantic import Field

from secretagent import config, llm_util, record
from secretagent.core import Implementation, register_factory
from secretagent.implement.core import PROMPT_TEMPLATE_DIR


def _load_template(name: str) -> Template:
    return Template((PROMPT_TEMPLATE_DIR / name).read_text())


class RLMFactory(Implementation.Factory):
    """Run an LLM in a multi-turn REPL loop with recursive sub-LLM calls.

    The LLM writes Python code in ```repl blocks, which is executed in a
    persistent namespace. The namespace contains:
    - context: the input data
    - llm_query(prompt): call a sub-LLM
    - FINAL(answer): return the final answer
    - FINAL_VAR(name): return the value of a variable as the final answer

    Examples:
      foo.implement_via('rlm')
      foo.implement_via('rlm', max_turns=5, sub_model='gemini/gemini-2.5-flash')
    """
    max_turns: int = 10
    max_output_chars: int = 20000
    sub_model: str | None = None
    prompt_kw: dict = Field(default_factory=dict)

    def setup(self, max_turns=10, max_output_chars=20000, sub_model=None, **prompt_kw):
        self.max_turns = max_turns
        self.max_output_chars = max_output_chars
        self.sub_model = sub_model
        self.prompt_kw = prompt_kw

    def __call__(self, *args, **kw):
        interface = self.bound_interface
        with config.configuration(**self.prompt_kw):
            return self._run_repl(interface, *args, **kw)

    def _run_repl(self, interface, *args, **kw):
        # --- prepare context ---
        arg_names = list(interface.annotations.keys())[:-1]
        if len(arg_names) == 1 and isinstance(args[0], str):
            context_val = args[0]
        else:
            context_val = dict(zip(arg_names, args))
            context_val.update(kw)

        context_info = f"type={type(context_val).__name__}"
        if isinstance(context_val, str):
            context_info += f", length={len(context_val)} chars"
        elif isinstance(context_val, dict):
            context_info += f", keys={list(context_val.keys())}"

        # --- result holder: [value, is_set] ---
        result_holder = [None, False]
        all_stats = []

        # --- build callables for the sandbox ---
        def llm_query(prompt: str) -> str:
            sub_model = self.sub_model or self.llm_model
            output, stats = llm_util.llm(prompt, sub_model)
            all_stats.append(stats)
            return output

        def FINAL(answer):
            result_holder[0] = answer
            result_holder[1] = True
            return answer

        def FINAL_VAR(name):
            if name not in namespace:
                raise NameError(f"Variable {name!r} not found in REPL namespace")
            result_holder[0] = namespace[name]
            result_holder[1] = True
            return namespace[name]

        # --- persistent namespace ---
        namespace = {
            'context': context_val,
            'llm_query': llm_query,
            'FINAL': FINAL,
            'FINAL_VAR': FINAL_VAR,
            'print': print,  # will be redirected per-block
        }

        # --- build prompts ---
        system_template = _load_template('rlm_system.txt')
        user_template = _load_template('rlm_user.txt')

        if config.get('llm.thinking'):
            thoughts = "\n<thought>\nANY THOUGHTS\n</thought>\n"
        else:
            thoughts = ""

        system_prompt = system_template.substitute(
            stub_src=interface.src,
            context_info=context_info,
            thoughts=thoughts,
        )

        input_args = interface.format_args(*args, **kw)

        # --- initial prompt ---
        iter0_prompt = "You have not interacted with the REPL environment yet. " \
            "Look through the context to understand what you're working with, " \
            "then plan and execute your strategy."
        user_prompt_0 = user_template.substitute(
            iteration_prompt=iter0_prompt,
            input_args=input_args,
        )

        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt_0}"

        # --- REPL loop ---
        conversation_log = []
        for turn in range(self.max_turns):
            llm_output, stats = llm_util.llm(full_prompt, self.llm_model)
            all_stats.append(stats)
            conversation_log.append({'turn': turn, 'llm_output': llm_output})

            # check for FINAL/FINAL_VAR in text (outside code blocks)
            text_outside_code = _strip_code_blocks(llm_output)
            final_match = re.search(
                r'FINAL\((.+?)\)', text_outside_code, re.MULTILINE)
            final_var_match = re.search(
                r'FINAL_VAR\((.+?)\)', text_outside_code, re.MULTILINE)

            if final_match and not final_var_match:
                raw_answer = final_match.group(1).strip().strip('"\'')
                answer = _type_cast(raw_answer, interface)
                self._record_success(
                    interface, args, kw, answer, all_stats,
                    turn + 1, conversation_log)
                return answer

            if final_var_match:
                var_name = final_var_match.group(1).strip().strip('"\'')
                if var_name in namespace:
                    answer = _type_cast(namespace[var_name], interface)
                    self._record_success(
                        interface, args, kw, answer, all_stats,
                        turn + 1, conversation_log)
                    return answer

            # extract and execute code blocks
            code_blocks = _extract_code_blocks(llm_output)
            all_outputs = []

            for code in code_blocks:
                output = _exec_code(code, namespace)
                if len(output) > self.max_output_chars:
                    remaining = len(output) - self.max_output_chars
                    output = output[:self.max_output_chars] + \
                        f"\n... [truncated, {remaining} chars omitted]"
                all_outputs.append((code, output))

                # check if FINAL was called inside code
                if result_holder[1]:
                    answer = _type_cast(result_holder[0], interface)
                    self._record_success(
                        interface, args, kw, answer, all_stats,
                        turn + 1, conversation_log)
                    return answer

            # build continuation prompt
            repl_section = ""
            for code, output in all_outputs:
                repl_section += f"\n[REPL OUTPUT]\nCode executed:\n```python\n{code}\n```\n\nOutput:\n{output}\n"

            iterN_prompt = "The history above shows your previous REPL interactions. " \
                "Continue using the REPL and sub-LLMs to answer the query."
            user_prompt_n = user_template.substitute(
                iteration_prompt=iterN_prompt,
                input_args=input_args,
            )

            full_prompt += f"\n\n[ASSISTANT]\n{llm_output}{repl_section}\n\n[USER]\n{user_prompt_n}"

        # exhausted max_turns — try to salvage
        error_msg = f"RLM did not converge within {self.max_turns} turns"
        self._record_exception(interface, args, kw, error_msg, all_stats,
                               conversation_log)
        raise RuntimeError(error_msg)

    def _record_success(self, interface, args, kw, answer, all_stats,
                        turn_count, conversation_log):
        total_stats = _aggregate_stats(all_stats)
        record.record(
            func=interface.name, args=args, kw=kw,
            output=answer, stats=total_stats,
            step_info=dict(
                turns=turn_count,
                n_llm_calls=len(all_stats),
                trajectory=conversation_log,
            ))

    def _record_exception(self, interface, args, kw, error_msg, all_stats,
                          conversation_log):
        total_stats = _aggregate_stats(all_stats)
        record.record(
            func=interface.name, args=args, kw=kw,
            output=f'**exception**: {error_msg}', stats=total_stats,
            step_info=dict(
                turns=self.max_turns,
                n_llm_calls=len(all_stats),
                trajectory=conversation_log,
            ))


# --- helpers ---

def _extract_code_blocks(text: str) -> list[str]:
    """Extract ```repl or ```python code blocks from LLM output."""
    blocks = re.findall(r'```(?:repl|python)\n(.*?)\n```', text, re.DOTALL)
    return blocks


def _strip_code_blocks(text: str) -> str:
    """Remove code blocks from text, leaving only prose."""
    return re.sub(r'```(?:repl|python)\n.*?\n```', '', text, flags=re.DOTALL)


def _exec_code(code: str, namespace: dict) -> str:
    """Execute code in the given namespace, capturing stdout."""
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)
    except Exception:
        stdout_capture.write(traceback.format_exc())
    return stdout_capture.getvalue()


def _type_cast(value, interface):
    """Cast value to the interface's return type."""
    return_type = interface.annotations.get('return', str)
    if isinstance(value, return_type):
        return value
    try:
        return return_type(value)
    except (ValueError, TypeError):
        return value


_STDOUT_FULL_THRESHOLD = 500
_METADATA_PREFIX_LEN = 200


def _build_context_metadata(context_val) -> str:
    """Build metadata string describing context without revealing full content.

    Per RLM paper (Algorithm 1), the model only sees Metadata(state),
    never the raw input.  It must write code to access `context`.
    """
    lines = [f"type: {type(context_val).__name__}"]
    if isinstance(context_val, str):
        lines.append(f"length: {len(context_val)} chars")
        if context_val:
            lines.append(f"prefix: {repr(context_val[:_METADATA_PREFIX_LEN])}")
    elif isinstance(context_val, dict):
        lines.append(f"keys: {list(context_val.keys())}")
        for k, v in context_val.items():
            if isinstance(v, str):
                lines.append(f"  context[{k!r}]: str, {len(v)} chars")
            elif isinstance(v, (list, tuple)):
                lines.append(f"  context[{k!r}]: {type(v).__name__}, {len(v)} items")
            else:
                lines.append(f"  context[{k!r}]: {type(v).__name__}")
    elif isinstance(context_val, (list, tuple)):
        lines.append(f"length: {len(context_val)} items")
    return "\n".join(lines)


def _build_stdout_metadata(output: str) -> str:
    """Build metadata for stdout — full text if short, else prefix + length.

    Per RLM paper (Algorithm 1), hist appends Metadata(stdout), not
    the raw output.  Short outputs are shown in full since they *are*
    effectively metadata-sized.
    """
    if not output.strip():
        return "(no output)"
    if len(output) <= _STDOUT_FULL_THRESHOLD:
        return output.rstrip()
    prefix = output[:_METADATA_PREFIX_LEN]
    return f"[{len(output)} chars] {repr(prefix)} ..."


def _aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate LLM stats from multiple calls."""
    return dict(
        input_tokens=sum(s.get('input_tokens', 0) for s in all_stats),
        output_tokens=sum(s.get('output_tokens', 0) for s in all_stats),
        cost=sum(s.get('cost', 0) for s in all_stats),
        latency=sum(s.get('latency', 0) for s in all_stats),
    )


register_factory('rlm', RLMFactory())
