# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**secretagent** is a lightweight Python framework for building agentic
systems where "everything looks like code." You define Python stubs
with only a type signature and docstring, and decorate them with
`@interface`.  These stubs are later *bound* to implementations
via `implement_via()` and a registry of `Implementation.Factory` classes
(each Factory subclass overrides `setup()` and `__call__()`).

## Running code: use uv for running code and installing packages

 * Use 'uv run' to run a script.
 * Use 'uv python pin 3.11.9' to make sure python3 runs
 * Use 'uv add FOO' to add a package
 * Use 'uv sync' after a 'git pull' to update the environment
 * Analysis scripts that use matplotlib, pandas, etc are tagged with
   'uv --script FOO.py PACKAGE'
 
## Core API (`secretagent.core`)

 * `@interface` — decorator that turns a stub function into an `Interface`
 * `@implement_via(method, **kw)` — decorator that creates an Interface and binds it in one step
 * `interface.implement_via(method, **kw)` — bind an existing Interface to an implementation
 * `all_interfaces()` — list all registered Interfaces
 * `all_factories()` — list all registered Factory name/instance pairs
 * `register_factory()` - add a new Implementation.Factory to the registery

### Built-in factories (registered in `_FACTORIES`)

 * `'direct'` — use the function body as the implementation
 * `'simulate'` — prompt an LLM to simulate a function call (NOT react, NOT an agent, just a single LLM call to predict function output)
 * `'prompt_llm'` — use a custom prompt template to predict the function
 * `'program_of_thought'` — generate Python code with an LLM and execute it in a sandboxed executor
 * `'simulate_pydantic'` — like simulate but uses a pydantic-ai Agent (in `implement/pydantic.py`)

### Key files

 * `src/secretagent/core.py` — Interface, Implementation, Factory base class
 * `src/secretagent/implement/pydantic.py` — SimulatePydanticFactory (pydantic-ai Agent backend)
 * `src/secretagent/implement/core.py` — built-in factories (direct, simulate, prompt_llm, program_of_thought)
 * `src/secretagent/config.py` — global/local configuration via `configure()` and `configuration()` context manager
 * `src/secretagent/record.py` — recording of interface calls via `recorder()` context manager
 * `src/secretagent/cache_util.py` — runtime cachier wrapper that reads config at call time
 * `src/secretagent/llm_util.py` — low-level LLM call helper
 * `src/secretagent/dataset.py` — Case and Dataset models for evaluation data
 * `src/secretagent/evaluate.py` — Evaluator base class for running experiments on datasets
 * `src/secretagent/cli/` — command-line tools (see below)
 * `tests/` — pytest tests (`test_core.py`, `test_config.py`, `test_record.py`)
 * `examples/` — quickstart.py, sports_understanding.py

## Configuration

This project is heavily configuration-driven, like most ML systems.

 * `src/secretagent/config.py` manages configurations 
 * `config.configure(yaml_file=...)` loads a hierarchical config
   * Dot notation is used for config keys, eg 'llm.model' or 'echo.llm_input'
 * `config.configure(cfg={...})` loads a user-specified config
 * `config.configure(llm=dict(model='gpt-5', echo={...})` also adds specific config values
 * `with config.configuration(echo=dict(service=True, ...)):` is a context manager
 that sets config parameters temporarily and restores them when it exits.

### Configuration keys

 * `llm.model` — LLM model name passed to litellm. Some useful llm.model values:
   * `together_ai/Qwen/Qwen3.5-9B` - good value ($0.10/$0.15 per 1M tokens)
     * doesn't support tool use, needed for pydantic-ai models
   * `together_ai/google/gemma-3n-E4B-it` - ultra-cheap ($0.02/$0.04 per 1M tokens)
     * doesn't support tool use, needed for pydantic-ai models
   * `claude-haiku-4-5-20251001` - quick cheap and stable, needs Anthropic API key
   * `together_ai/deepseek-ai/DeepSeek-V3.1` - cheap but strong reasoning ($0.60/$1.70 per 1M tokens)
   * `together_ai/openai/gpt-oss-20b` - very cheap ($0.05/$0.20 per 1M tokens)
   * `together_ai/openai/gpt-oss-120b` - good value, larger ($0.15/$0.60 per 1M tokens)
   * `together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct` - good value, MoE ($0.15/$1.50 per 1M tokens)
 * `llm.thinking` — if truthy, include `<thought>` scaffolding in simulate prompts
 * `echo.model` — print which model is being called
 * `echo.llm_input` — print the prompt sent to the LLM in a box
 * `echo.llm_output` — print the LLM response in a box
 * `echo.code_eval_output` — print result of executing LLM-generated code
 * `echo.service` — print service information
 * `echo.call` — print function call signatures
 * `evaluate.expt_name` — name tag for the experiment (used in result filenames and dataframes)
 * `evaluate.result_dir` — directory to save results CSV and config YAML snapshot
 * `evaluate.record_details` — if `True`, include full rollout recordings in JSONL output (default `False`)
 * `cachier.enable_caching` — if `False`, bypass cachier entirely (default `True`)
 * `cachier.cache_dir` — directory for cachier's on-disk cache
 * Other `cachier.*` keys are passed through to `@cachier()` (e.g. `stale_after`, `allow_none`)

## Caching

Calls to llm models should be routed thru litellm, usually through
llm_util.  Calls can be cached in a directory, which caches output and
other stats (e.g., input/output tokens and cost).

## CLI tools

See @docs/CLI.md
