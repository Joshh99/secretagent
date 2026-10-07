### Configuration settings

Settings are named with dots, such as `llm.model` or `echo.llm_input`.

- `llm.model`: the model name passed to litellm. Some useful values, with prices in USD per million input and output tokens:
  - `together_ai/Qwen/Qwen3.5-9B`: good value ($0.10 / $0.15). It cannot call tools, which the pydantic-ai methods need.
  - `together_ai/google/gemma-3n-E4B-it`: very cheap ($0.02 / $0.04). It cannot call tools, which the pydantic-ai methods need.
  - `claude-haiku-4-5-20251001`: fast, cheap and stable. Needs an Anthropic API key.
  - `together_ai/deepseek-ai/DeepSeek-V3.1`: cheap and good at reasoning ($0.60 / $1.70).
  - `together_ai/openai/gpt-oss-20b`: very cheap ($0.05 / $0.20).
  - `together_ai/openai/gpt-oss-120b`: good value and larger ($0.15 / $0.60).
  - `together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct`: good value, a mixture-of-experts model ($0.15 / $1.50).
  - `gemini/gemini-2.5-flash`: a thinking model ($0.30 / $2.50, up to 65K output tokens).
  - `gemini/gemini-2.5-flash-lite`: cheap Gemini ($0.10 / $0.40, up to 65K output tokens).
  - `gemini/gemini-3.1-flash-lite-preview`: very cheap Gemini preview ($0.25 / $1.50, up to 65K output tokens).
- `llm.thinking`: if true, the simulate prompts ask the model to think first, inside `<thought>` tags.
- `llm.reasoning_effort`: for Gemini thinking models, `low`, `medium` or `high`.
- `simulate.full_src`: if true, the whole function body is kept in `Interface.src`; otherwise only the signature and docstring are kept.
- `echo.model`: print which model is being called.
- `echo.llm_input`: print each prompt sent to the model, in a box.
- `echo.llm_output`: print each model reply, in a box.
- `echo.code_eval_output`: print the result of running code written by the model.
- `echo.service`: print information about the service being called.
- `echo.call`: print each function call's signature.
- `echo.box_width`: the widest the printed boxes can be. With `0`, the default, it uses the terminal width (from `shutil.get_terminal_size`, or 120 columns if that fails) minus the box frame. Long lines wrap, and existing line breaks are kept.
- `evaluate.expt_name`: the name of the experiment, used in result file names and tables.
- `evaluate.result_dir`: the folder where the results CSV and a YAML copy of the config are saved.
- `evaluate.record_details`: if true, save the full record of every run in the JSONL output. The default is false.
- `evaluate.max_workers`: how many questions to run in parallel. The default is 1.
- `pydantic.retries`: how many times a pydantic-ai agent may retry when its output fails validation. The default is 1.
- `cachier.enable_caching`: if false, skip the cache completely. The default is true.
- `cachier.cache_dir`: the folder where model calls are cached.
- Any other `cachier.*` setting is passed on to `@cachier()`, for example `stale_after` or `allow_none`.
