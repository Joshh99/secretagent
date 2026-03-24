"""Generate pipeline code using an LLM.

The composer sends a single prompt to a powerful LLM with the available
ptool signatures and a task description, and extracts a Python function
body from the response.
"""

import pathlib
import re
import subprocess
import tempfile
import textwrap
from string import Template
from typing import Callable

from secretagent import config
from secretagent.llm_util import llm
from secretagent.orchestrate.catalog import PtoolCatalog

PROMPT_TEMPLATE_DIR = pathlib.Path(__file__).parent / 'prompt_templates'

RETRY_ADDENDUM = """

## Previous attempt failed

Your previous code:
```python
$previous_code
```

Failed with this error:
```
$error_message
```

Fix the error and return corrected code in a ```python block.
"""


def compose(
    task_description: str,
    catalog: PtoolCatalog,
    entry_signature: str,
    model: str | None = None,
    _retry_context: str = '',
) -> str:
    """Generate a pipeline function body using an LLM.

    Args:
        task_description: what the pipeline should accomplish
        catalog: available ptools and their metadata
        entry_signature: function signature line, e.g.
            "def my_workflow(x: str, y: int) -> str:"
        model: LLM model; defaults to config 'orchestrate.model'
        _retry_context: internal — appended to prompt on retries

    Returns:
        Generated Python code (the function body, not wrapped in def).
    """
    model = model or config.require('orchestrate.model')
    template = Template((PROMPT_TEMPLATE_DIR / 'compose.txt').read_text())

    prompt = template.substitute(
        tool_stubs=catalog.render(),
        tool_names=', '.join(catalog.names),
        task_description=task_description,
        entry_signature=entry_signature,
    )
    prompt += _retry_context

    llm_output, stats = llm(prompt, model)

    if config.get('echo.orchestrate_llm'):
        from secretagent.llm_util import echo_boxed
        echo_boxed(llm_output, 'orchestrator LLM output')

    code = _extract_code(llm_output)
    code = _strip_def_line(code, entry_signature)
    code = _ruff_fix(code, entry_signature)
    return code


def compose_with_retry(
    task_description: str,
    catalog: PtoolCatalog,
    entry_signature: str,
    test_fn: Callable,
    model: str | None = None,
    max_retries: int | None = None,
) -> tuple[str, int]:
    """Generate pipeline code with smoke-test validation and retry.

    Calls compose() then runs test_fn to validate. On failure, retries
    with the error message appended to the prompt.

    Args:
        task_description: what the pipeline should accomplish
        catalog: available ptools
        entry_signature: function signature line
        test_fn: callable that takes a Pipeline and runs a smoke test;
            should raise on failure
        model: LLM model; defaults to config
        max_retries: max attempts; defaults to config 'orchestrate.max_retries' (3)

    Returns:
        (code, attempt) tuple where attempt is 1-indexed (for pass@k reporting)
    """
    from secretagent.orchestrate.pipeline import Pipeline

    max_retries = max_retries or config.get('orchestrate.max_retries', 3)
    model = model or config.require('orchestrate.model')

    attempts = []
    retry_context = ''

    for attempt in range(1, max_retries + 1):
        code = compose(
            task_description, catalog, entry_signature,
            model=model, _retry_context=retry_context,
        )
        attempts.append(code)

        # Try building and running the pipeline
        try:
            test_fn(code)
            return code, attempt
        except Exception as e:
            error_msg = f'{type(e).__name__}: {e}'
            if config.get('echo.orchestrate'):
                print(f'[orchestrate] attempt {attempt}/{max_retries} failed: {error_msg}')
            retry_context = Template(RETRY_ADDENDUM).substitute(
                previous_code=code,
                error_message=error_msg,
            )

    # All retries exhausted
    raise RuntimeError(
        f'Pipeline generation failed after {max_retries} attempts.\n'
        f'Last error: {error_msg}\n'
        f'Attempted codes:\n' +
        '\n---\n'.join(f'Attempt {i+1}:\n{c}' for i, c in enumerate(attempts))
    )


def _extract_code(text: str) -> str:
    """Extract Python code from LLM output.

    Looks for ```python ... ``` blocks. If multiple exist,
    takes the last one (the refined version).
    """
    matches = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    if not matches:
        raise ValueError(
            f'No ```python``` code block found in LLM output:\n{text[:500]}'
        )
    return matches[-1].strip()


def _strip_def_line(code: str, entry_signature: str) -> str:
    """Strip leading def line if the LLM included one."""
    lines = code.strip().split('\n')
    if lines and lines[0].strip().startswith('def '):
        code = textwrap.dedent('\n'.join(lines[1:]))
    return code.strip()


def _ruff_fix(code: str, entry_signature: str) -> str:
    """Run ruff check --fix on the generated code.

    Wraps the code body in the function signature so ruff sees valid
    Python, runs the fix, then strips the wrapper back off.
    """
    indented = textwrap.indent(code, '    ')
    full_source = f'{entry_signature}\n{indented}\n'

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(full_source)
        tmp_path = f.name

    try:
        subprocess.run(
            ['ruff', 'check', '--fix', '--quiet', tmp_path],
            capture_output=True, timeout=10,
        )
        with open(tmp_path) as f:
            fixed_source = f.read()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ruff not available or timed out — return code as-is
        return code
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    # Strip the wrapper: remove the def line and dedent
    lines = fixed_source.strip().split('\n')
    if lines and lines[0].strip().startswith('def '):
        body_lines = lines[1:]
        body = textwrap.dedent('\n'.join(body_lines))
        return body.strip()
    return code
