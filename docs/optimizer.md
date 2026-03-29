# Grid Search Optimizer

The optimizer searches over a discrete space of configuration overrides
to find the best-performing config for a benchmark. Each config is
evaluated as a subprocess to ensure clean state.

## Architecture

```
              ┌──────────────────────────┐
              │  User provides:          │
              │  - base command          │
              │  - search space (YAML)   │
              │  - base dotlist overrides │
              └─────────┬────────────────┘
                        │
              ┌─────────▼────────────────┐
              │  SearchSpace              │
              │  Generates all combos     │
              │  via itertools.product    │
              └─────────┬────────────────┘
                        │
              ┌─────────▼────────────────┐
              │  GridSearchRunner         │
              │  For each config point:   │
              │  1. Run subprocess        │
              │  2. Parse accuracy        │
              │  3. Load CSV for stats    │
              └─────────┬────────────────┘
                        │
              ┌─────────▼────────────────┐
              │  Summary DataFrame        │
              │  Ranked by accuracy       │
              │  With cost/latency/tokens │
              └──────────────────────────┘
```

Each config runs as an independent subprocess (`uv run python expt.py
run ...`), which guarantees clean module state. This is necessary
because `@interface` decorators modify global state that cannot be
reset in-process.

## Quickstart

### 1. Define a search space (YAML)

```yaml
# sweep_space.yaml
search_space:
  evaluate.entry_point:
    - answer_question
    - answer_question_workflow
  llm.thinking:
    - "true"
    - "false"
  llm.model:
    - together_ai/deepseek-ai/DeepSeek-V3
    - claude-haiku-4-5-20251001
```

### 2. Run the sweep

```bash
uv run -m secretagent.cli.optimize sweep \
  --command "uv run python expt.py run --config-file conf/murder.yaml" \
  --space-file sweep_space.yaml \
  --cwd benchmarks/musr \
  --timeout 1800 \
  --output sweep_results.csv \
  dataset.n=75 cachier.enable_caching=false
```

Extra args after the options (`dataset.n=75`, etc.) are base overrides
applied to every config.

### 3. View results

```bash
uv run -m secretagent.cli.optimize summary sweep_results.csv
```

Or load in Python:

```python
import pandas as pd
df = pd.read_csv('sweep_results.csv')
print(df.sort_values('accuracy', ascending=False))
```

## Programmatic Usage

```python
from secretagent.optimize import SearchSpace, GridSearchRunner

space = SearchSpace({
    'llm.thinking': [True, False],
    'ptools.answer_question.method': ['simulate', 'direct'],
})

runner = GridSearchRunner(
    command='uv run python expt.py run --config-file conf/murder.yaml',
    space=space,
    base_dotlist=['dataset.n=75', 'cachier.enable_caching=false'],
    cwd='benchmarks/musr',
    timeout=1800,
    metric='correct',
)

summary = runner.run_all()
print(summary)
runner.save_summary('results.csv')
```

## CLI Reference

### `sweep`

Run a grid search over a config space.

```
uv run -m secretagent.cli.optimize sweep [OPTIONS] [DOTLIST_OVERRIDES...]
```

| Option | Default | Description |
|---|---|---|
| `--command` | (required) | Base command to run (quoted string) |
| `--space-file` | (required) | YAML file defining search space |
| `--prefix` | `sweep` | Experiment name prefix |
| `--cwd` | current dir | Working directory for subprocesses |
| `--timeout` | `1800` | Timeout per config in seconds |
| `--metric` | `correct` | Metric column to optimize |
| `--output` | `sweep_summary.csv` | Output summary CSV path |

Extra positional args are treated as base dotlist overrides applied to
all configs.

### `summary`

Display results from a saved sweep.

```
uv run -m secretagent.cli.optimize summary SWEEP_RESULTS.csv [--top-n N]
```

| Option | Default | Description |
|---|---|---|
| `--top-n` | `10` | Number of top results to show |

## Search Space Format

The YAML file can use either format:

```yaml
# Flat format
search_space:
  key1:
    - value1
    - value2
  key2:
    - value3
    - value4
```

```yaml
# Also valid: top-level keys (without search_space wrapper)
key1:
  - value1
  - value2
key2:
  - value3
  - value4
```

All values must be lists. The optimizer generates every combination
(Cartesian product). A space with 3 keys of sizes 2, 3, 2 produces
2 × 3 × 2 = 12 configs.

## Output Format

The sweep produces:

1. **Per-config result directories** — standard format under
   `evaluate.result_dir`, each with `results.csv`, `results.jsonl`,
   and `config.yaml`. Named `{prefix}_{idx:03d}`.

2. **Summary CSV** — one row per config with columns:

| Column | Description |
|---|---|
| `config_idx` | Config index (0-based) |
| `expt_name` | Experiment name (e.g., `sweep_000`) |
| *search dimensions* | One column per search space key |
| `accuracy` | Mean of the `metric` column |
| `elapsed` | Wall clock time in seconds |
| `total_cost` | Sum of per-example costs |
| `cost_per_q` | Mean cost per example |
| `total_latency` | Sum of per-example latencies |
| `latency_per_q` | Mean latency per example |
| `input_tokens_per_q` | Mean input tokens per example |
| `output_tokens_per_q` | Mean output tokens per example |
| `status` | `ok`, `failed`, or `timeout` |

The summary is compatible with pandas and can be loaded directly for
analysis.

## Compatibility

The optimizer works with any benchmark that follows the secretagent
experiment pattern:

1. Uses typer CLI with `allow_extra_args` for dotlist overrides
2. Calls `Evaluator.evaluate()` which prints
   `Accuracy: X% (N/M)` and `saved in <path>`

All existing benchmarks (MUSR, NaturalPlan, sports_understanding,
MedCalc, RuleArena) satisfy these requirements.

## Example: MUSR Murder Sweep

```yaml
# benchmarks/musr/sweep_murder.yaml
search_space:
  evaluate.entry_point:
    - answer_question
    - answer_question_workflow
  llm.thinking:
    - "true"
    - "false"
```

```bash
cd benchmarks/musr
uv run -m secretagent.cli.optimize sweep \
  --command "uv run python expt.py run --config-file conf/murder.yaml" \
  --space-file sweep_murder.yaml \
  --cwd . \
  dataset.n=75 llm.model=together_ai/deepseek-ai/DeepSeek-V3
```

Sample output:

```
SWEEP RESULTS (sorted by accuracy)
============================================================
 expt_name     evaluate.entry_point llm.thinking  accuracy  cost_per_q  latency_per_q
 sweep_001          answer_question        false    0.6400      0.0021           1.8
 sweep_003 answer_question_workflow        false    0.6267      0.0079          20.5
 sweep_000          answer_question         true    0.6133      0.0025           8.1
 sweep_002 answer_question_workflow         true       NaN         NaN           NaN  (timeout)

Best: sweep_001 — 64.0%
  evaluate.entry_point = answer_question
  llm.thinking = false
```

## Module Structure

```
src/secretagent/
    optimize.py              # SearchSpace, GridSearchRunner
    cli/
        optimize.py          # CLI (sweep, summary)
```
