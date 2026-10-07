# Supplementary material

This folder holds the code and the saved results behind the paper "Learning to Construct Practical Agentic Systems". With it you can rebuild the paper's main tables on your own laptop in about a minute. You do not need an API key, and no model gets called.

## Check the tables in one minute

You need Python 3.9 or newer. Nothing else has to be installed for these two checks.

```
python rebuild_tables.py --check
python aflow/build_aflow_numbers.py --check
```

The first command rebuilds the paper's main results tables and learning tables from the saved results, then compares every number with the paper. Its last line should say `528 of 528 numbers match the paper.` The second does the same for the AFlow comparison and should say `323 of 323 numbers match the paper.` The rebuilt tables are written to `tables/rebuilt/`.

## What is in this folder

| Path | What it is |
|---|---|
| `appendix.pdf` | The paper's appendix. |
| `src/secretagent/` | The framework. You write a Python function with only a type signature and a docstring, mark it as an interface, and then choose how it is carried out: by plain Python, by one LLM call, or by a learned component. |
| `benchmarks/<task>/` | The code for each task: its pseudo-tools, workflows, configuration files and prompts. |
| `benchmarks/COMMON/...` | The saved results that the tables read. Each `results.csv` has one row per question: its ID, 1 if the answer was right and 0 if not, and what the question cost in US dollars. Each run's `config.yaml` records the settings it used. |
| `benchmarks/COMMON/optimize-results/` | Summaries of the NSGA-II configuration searches. |
| `tables/index.json` | For every number in the rebuilt tables: which saved file it comes from, how many questions that file has, and a checksum of the original full file. A checksum is a fingerprint of a file's contents. |
| `tables/expected_tables.json` | The numbers exactly as printed in the paper, read from the paper's LaTeX source. |
| `rebuild_tables.py` | Rebuilds the tables and compares them with the paper. |
| `aflow/` | The comparison with AFlow, an automatic workflow builder. It has its own README. |
| `scripts/` | The scripts that made the tables from the full result files. They are here so you can see exactly how each table was made. |
| `tests/` | Unit tests for the framework. |
| `pyproject.toml`, `uv.lock` | The Python packages we used, with exact versions. |

## Tests

After `uv sync`, run `uv run pytest tests`. On our Windows laptop with no API keys set, 384 tests pass and 15 are skipped, because they call a real model and need an API key. Five tests in `tests/test_config_extras.py` fail on Windows only: Windows joins file paths with `\` and those tests expect `/`. Tests that need the full saved data or a local copy of AFlow are not included here.

## A few terms

Pseudo-tool: a tool that looks like an ordinary function to the rest of the system, but inside it calls an LLM on a small, restricted piece of the input.

Workflow: a fixed series of steps, some LLM calls and some plain Python, that solves one kind of task. The paper compares hand-built workflows, learned workflows and ReAct, where the model decides its next step as it goes.

Held out questions: test questions that a method never saw while it was being built or tuned, so its score on them is fair.

## How each number is computed

Accuracy is the share of questions answered correctly.

In the main results tables (`hero-compact`, `hero`, `hero-cost`), cost is the average cost of the questions that have a recorded cost, times 100. That gives US dollars per 100 questions.

In the learning cost table, cost is the total recorded cost divided by all the questions, times 100. A question with no recorded cost adds nothing. Most of these are pure Python steps that never call a model; a few are model calls that timed out.

In the learning tables, the Murder column uses only the 50 test questions that AFlow never saw during its search, so every method is scored on the same questions. The two CodeDist rows are the exception, because they were tested on a different set of 75 questions.

Averages are the mean of the printed values in the row, rounded half up, the same way as in the paper.

AFlow's costs are what the model provider actually charged us. The other rows are priced at the provider's list prices. The `aflow/` README explains both, and how they compare.

## What is not included, and why

The question text and the model answers are not in the saved results. Leaving them out keeps the folder small and avoids sharing benchmark data whose license may not allow it. The datasets themselves are not included for the same reason. The task code in `benchmarks/<task>/` shows which files each loader expects.

The saved model calls, caches and run logs are not included either. Together they come to several gigabytes.

To read or run the code, install [uv](https://docs.astral.sh/uv/) and run `uv sync` in this folder. On Windows, unzip into a short folder such as `C:\supp` first: some installed package paths are long, and Windows refuses paths over 260 characters.

Running the experiments again needs your own API keys and costs money. Each run's `config.yaml` records the model and settings it used.

## Anonymity

Names, user names, personal file paths and repository links have been replaced with `anonymous`, `<path>` or `<repository-url>`.
