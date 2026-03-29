# CLI Commands

All CLI tools live in `src/secretagent/cli/` and are run with `uv run -m`.

## secretagent.cli.costs

Summarize LLM costs from cachier cache files.

```
uv run -m secretagent.cli.costs [CACHE_DIR] [--config-file FILE]
```

| Argument/Option | Description |
|---|---|
| `CACHE_DIR` | Path to cachier cache directory (optional) |
| `--config-file` | YAML config file to load (uses `cachier.cache_dir` from config) |

Prints total and per-call statistics for input/output tokens, latency, and cost.

## secretagent.cli.results

Analyze experiment results saved by savefile. Experiment directories (or CSV files within them) are passed as extra positional arguments after the subcommand. Results are filtered with `--latest` and `--check`.

### list

Show available experiment directories and row counts.

```
uv run -m secretagent.cli.results list DIRS... [--latest K] [--check KEY=VALUE]
```

### average

Report mean +/- stderr of metrics, grouped by experiment.

```
uv run -m secretagent.cli.results average DIRS... [--latest K] [--check KEY=VALUE] [--metric NAME]
```

Default metric is `cost`. Multiple `--metric` flags are supported.

### pair

Run paired t-tests on metrics across experiments (requires at least 2 experiments).

```
uv run -m secretagent.cli.results pair DIRS... [--latest K] [--check KEY=VALUE] --metric NAME
```

At least one `--metric` is required.

### compare-configs

Show configuration differences between experiments.

```
uv run -m secretagent.cli.results compare-configs DIRS... [--latest K] [--check KEY=VALUE]
```

### validate

Check that experiment directories contain required files (`config.yaml`, `results.csv` by default).

```
uv run -m secretagent.cli.results validate DIRS... [--latest K] [--check KEY=VALUE] [--require FILE] [--norequire FILE] [--purge]
```

| Option | Description |
|---|---|
| `--require` | Add an additional required file |
| `--norequire` | Remove a default required file |
| `--purge` | Interactively delete directories that fail validation |

### delete-obsolete

Delete experiment directories not retained by `filter_paths`.

```
uv run -m secretagent.cli.results delete-obsolete DIRS... [--latest K] [--check KEY=VALUE]
```

Lists directories to keep and delete, then prompts for confirmation.

### Common options

| Option | Default | Description |
|---|---|---|
| `--latest K` | 1 | Keep latest K directories per tag; 0 for all |
| `--check KEY=VALUE` | — | Config constraint filter (repeatable) |
| `--config-file FILE` | — | YAML config file to load |

## secretagent.cli.optimize

Grid search over a discrete space of configuration overrides. See [optimizer.md](optimizer.md) for full documentation.

### sweep

Run a grid search from a YAML search space definition.

```
uv run -m secretagent.cli.optimize sweep \
  --command "uv run python expt.py run --config-file conf/murder.yaml" \
  --space-file sweep_space.yaml \
  [--cwd DIR] [--timeout SECS] [--metric NAME] [--output FILE] \
  [DOTLIST_OVERRIDES...]
```

| Option | Default | Description |
|---|---|---|
| `--command` | (required) | Base command to run (quoted) |
| `--space-file` | (required) | YAML file defining search space |
| `--prefix` | `sweep` | Experiment name prefix |
| `--cwd` | current dir | Working directory for subprocesses |
| `--timeout` | `1800` | Timeout per config in seconds |
| `--metric` | `correct` | Metric column to optimize |
| `--output` | `sweep_summary.csv` | Output summary CSV |

### summary

Display results from a saved sweep.

```
uv run -m secretagent.cli.optimize summary SWEEP_RESULTS.csv [--top-n N]
```

## secretagent.cli.learn

Learn implementations from recorded interface calls.

### rote

Collect training data for a rote (lookup-based) learner from recorded interface calls.

```
uv run -m secretagent.cli.learn rote DIRS... --interface NAME [--latest K] [--check KEY=VALUE] [--train-dir DIR]
```

| Option | Default | Description |
|---|---|---|
| `--interface` | (required) | Interface name to extract, e.g. `consistent_sports` |
| `--latest K` | 1 | Keep latest K directories per tag; 0 for all |
| `--check KEY=VALUE` | — | Config constraint filter (repeatable) |
| `--train-dir DIR` | `/tmp/rote_train` | Directory to store collected training data |
