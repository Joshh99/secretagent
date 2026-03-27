import pathlib
import os
import time

from typing import Callable, Any
from together import Together

from secretagent import config, record
from secretagent.core import Interface, register_factory
from secretagent.implement_core import SimulateFactory

# Together pricing in USD per 1M tokens: (input, output)
# Keep this intentionally small and explicit for current benchmark usage.
_TOGETHER_PRICE_PER_MTOKENS: dict[str, tuple[float, float]] = {
    'Qwen/Qwen3-VL-8B-Instruct': (0.18, 0.68),
    'Qwen/Qwen2.5-VL-72B-Instruct': (1.95, 8.0),
}
_DEFAULT_VLM_SYSTEM_PROMPT = 'You are an expert frontend developer.'


def _normalize_together_model_name(model_name: str) -> str:
    return model_name.removeprefix('together_ai/')


def _estimate_together_cost_usd(
    model_name: str,
    input_tokens: float,
    output_tokens: float,
) -> float:
    model_id = _normalize_together_model_name(model_name)
    rates = _TOGETHER_PRICE_PER_MTOKENS.get(model_id)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    # Together pricing metadata is expressed in USD per 1M tokens.
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000.0


class ImplementVLMFactory(SimulateFactory):
    """Implement an Interface using a VLM."""
    def build_fn(
        self,
        interface: Interface,
        example_file=None,
        images=None,
        output_mode: str = 'answer_tag',
        prompt_mode: str = 'simulate',
        user_text: str | None = None,
        **prompt_kw,
    ) -> Callable:
        examples_cases = None
        if example_file:
            import json
            data = json.loads(pathlib.Path(example_file).read_text())
            examples_cases = data.get(interface.name, [])

        def result_fn(*args, **call_kw):
            call_images = call_kw.pop('images', None)
            merged_images = call_images if call_images is not None else images
            with config.configuration(**prompt_kw):
                messages = self.create_vlm_prompt(
                    interface,
                    *args,
                    images=merged_images,
                    examples=examples_cases,
                    prompt_mode=prompt_mode,
                    user_text=user_text,
                    **call_kw,
                )
                output, stats = self._call_vlm(messages=messages, output_mode=output_mode)
                record.record(
                    func=interface.name,
                    args=args,
                    kw=call_kw,
                    output=output,
                    stats=stats,
                )
                return output

        return result_fn

    def create_vlm_prompt(
        self,
        interface,
        *args,
        images: dict[str, Any] | None = None,
        examples=None,
        prompt_mode: str = 'simulate',
        user_text: str | None = None,
        **kw,
    ) -> list[dict[str, Any]]:
        if prompt_mode == 'docstring':
            arg_names = list(interface.annotations.keys())[:-1]
            arg_dict = dict(zip(arg_names, args))
            arg_dict.update(kw)
            framework = arg_dict.get('framework')
            prompt = (interface.doc or '').strip()
            if framework:
                prompt = f'{prompt}\n\nTarget framework: {framework}'
            if user_text:
                prompt = f'{prompt}\n\n{user_text}'
            system_prompt = config.get('vlm.system_prompt') or _DEFAULT_VLM_SYSTEM_PROMPT
        else:
            prompt = self.create_prompt(interface, *args, examples=examples, **kw)
            system_prompt = ''

        if images:
            content_parts = []
            for _, image_url in images.items():
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_url}"},
                })
            content_parts.append({
                "type": "text",
                "text": prompt,
            })
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_parts},
                ]
            else:
                messages = [{"role": "user", "content": content_parts}]
        else:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        return messages

    def _call_vlm(
        self,
        messages: list[dict[str, Any]],
        output_mode: str = 'answer_tag',
    ) -> tuple[str, dict[str, float]]:
        start_time = time.time()
        model_name = config.get('vlm.model') or config.require('llm.model')
        together_client = Together(api_key=os.environ['TOGETHER_API_KEY'])
        response = together_client.chat.completions.create(
            model=model_name,
            max_tokens=config.get('vlm.max_tokens') or 32768,
            messages=messages,
        )
        usage = getattr(response, 'usage', None)
        input_tokens = float(getattr(usage, 'prompt_tokens', 0) or 0)
        output_tokens = float(getattr(usage, 'completion_tokens', 0) or 0)
        cost = _estimate_together_cost_usd(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        stats = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency': time.time() - start_time,
            'cost': cost,
        }
        raw_content = str(response.choices[0].message.content)
        if output_mode == 'freeform':
            output = raw_content
        else:
            output = self.parse_output(str, raw_content)
        return output, stats

register_factory('vlm', ImplementVLMFactory())