# DesignBench

DesignBench tests how well a model can rebuild a web page's user interface as code. This benchmark loads DesignBench examples (HTML and metadata), asks the model to write the front-end code, and can then compare screenshots of the result with the reference pages.

## Before you start

- Run `uv sync` from the top folder of the repository.
- You need a local copy of the DesignBench repository. By default it is looked for at `../DesignBench`, next to this repository. To use another place, set `designbench.root=/absolute/path/to/DesignBench` in the config or on the command line.
- To get the visual scores (`clip_similarity`, `mae` and `ssim`), also install DesignBench's evaluation packages.

## Run it

```bash
cd secretagent
uv sync
cd benchmarks/designbench
```

With the default settings:

```bash
uv run python expt.py run
```

On 10 examples only:

```bash
uv run python expt.py run dataset.n=10
```

With a different web framework (the model still comes from `conf/conf.yaml`):

```bash
uv run python expt.py run dataset.framework=react
```

Generate code only, without the visual scores:

```bash
uv run python expt.py run benchmark.skip_eval=true
```

## Makefile shortcuts

- `make model`: run one experiment, with `FRAMEWORK`, `MODEL`, `N` and `EXPT` as options.
- `make list`: list the saved result folders.
- `make avg`: average `correct`, `clip_similarity` and `cost` over each run.
- `make pair`: paired statistical comparisons between runs.
- `make compare`: show how the settings of different runs differ.

## Main settings

| Setting | Default | What it does |
|---|---|---|
| `llm.model` | `Qwen/Qwen3-VL-8B-Instruct` | The model, as named by litellm |
| `evaluate.expt_name` | `designbench_ptool` | Run name, used in the output folder name |
| `evaluate.result_dir` | `results` | Where results are saved |
| `evaluate.entry_point` | `generate_code` | The interface called for each example |
| `dataset.framework` | `vanilla` | Which set of examples to use: `vanilla`, `react`, `vue` or `angular` |
| `dataset.n` | not set | The largest number of examples to run |
| `dataset.max_reference_chars` | `20000` | Long HTML is cut to this many characters in the prompt |
| `benchmark.output_framework` | `null` | Framework for the generated code, if different |
| `benchmark.skip_eval` | `false` | Skip the screenshots and visual scores |

## What a run saves

Each run writes to `results/<timestamp>.<expt_name>/`:

- `results.csv`: one row per example, with its scores and where its files are.
- `results.jsonl`: the full record of each example.
- `config.yaml`: the exact settings used.
- `artifacts/`: the generated code, screenshots and score files for each example.
