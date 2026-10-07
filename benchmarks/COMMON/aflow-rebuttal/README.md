# Running AFlow

This folder holds supporting files for AFlow, the automatic workflow builder we compare against in the paper. AFlow itself is a separate public project, so this folder holds our changes to it rather than a copy of it: task adapters, scorers, the rule for picking a workflow, and one patch file.

If you only want to check the AFlow numbers in the paper, you do not need any of this. Run `python aflow/build_aflow_numbers.py --check` from the top folder of the supplementary package. In the full repository, use `python submission/aflow/build_aflow_numbers.py --check`. New searches need the full repository, the original benchmark data and the full saved results. The zip contains only small result copies.

## What AFlow does

AFlow starts from a generated workflow that makes one model call. Its prompt includes the task description and, in the `matched` setting, the hand-written tool descriptions. A planner model (Gemini 3.1 Pro) writes changed versions. The paper's searches use DeepSeek-V3.1 on 50 validation questions and have 54 numbered rounds, including the starting workflow. Failed proposals still use a round. For the paper, we exclude damaged rounds and pick the best validation score, with ties going to the cheaper round and then the earlier one. We run the chosen workflow once on the test questions.

## Set up a copy of AFlow

1. Get AFlow at the exact version we used:

       git clone https://github.com/FoundationAgents/AFlow.git ../aflow
       git -C ../aflow checkout 3f457218fc716093fe53f6df8a5d5e6379d66346

2. Install AFlow's own `requirements.txt` into `../aflow/.venv`.

3. Copy `upstream/config2.yaml.redacted` to `../aflow/config/config2.yaml` and put your own API keys where it says `<YOUR_KEY>`. Keep that file private. The entry the paper's runs used is `deepseek-v31-atlascloud`: DeepSeek-V3.1 through OpenRouter, sent only to the AtlasCloud provider, with fallback to other providers turned off.

4. From the top folder of this repository, install our changes and the task data into the AFlow copy, then check them:

       uv run python scripts/prepare_aflow.py --aflow-dir ../aflow --apply
       uv run python scripts/run_aflow_table_task.py check --task musr_murder --aflow-dir ../aflow --write-export
       uv run python scripts/prepare_aflow.py --aflow-dir ../aflow

   The first command applies `upstream/aflow_changes.patch` and copies the added files listed in `upstream/table_tasks_install.md`. The second exports the Murder validation and test data into AFlow and updates `upstream/dataset_checksums.txt`. Review any changes to that file and commit them before a paid search; the launcher requires a clean repository. The last command checks the setup: the AFlow version, the patch, the added files, the operator templates and your model config. The launcher checks the dataset checksums before it spends anything.

For the older Sports and FinQA exports and the 50-question Object test subset, the original command is `uv run python scripts/export_aflow_datasets.py --aflow-dir ../aflow`. For the launcher's full 106-question Object test set, use `uv run python scripts/run_aflow_table_task.py check --task musr_object --aflow-dir ../aflow --write-export`. The older `uv run python scripts/run_aflow_table_musr.py --aflow-dir ../aflow --prepare-only` command writes a separate `musrobjectplacements_table_test.jsonl` for that helper's own scoring step.

## Run one search

Look at the inputs first with `--dry-run`:

    uv run python scripts/run_aflow_cell.py --aflow-dir ../aflow --task musr_murder \
        --condition matched --seed 1 --executor deepseek-v31-atlascloud \
        --optimizer gemini-3.1-pro-preview --max-candidates 54 --concurrency 10 --dry-run

Then run the same command without `--dry-run`. The model, round budget and concurrency match the paper's saved settings; the paper uses seeds 1, 2 and 3. Each search gets its own folder under the AFlow checkout's `runs/`. After the search, this launcher checks the validation results and uses `--mode pareto` to test the workflows on the validation Pareto front. This means keeping a workflow unless another has at least as high a score and at most as high a cost, with one strict improvement; exact ties keep the earlier round. This can test several workflows. The paper's tables use one chosen workflow per search, as described above.

| Path | What is in it |
|---|---|
| `runs/<run>/manifest.json` | The inputs of the run: code versions, dataset checksums, prompt checksums and settings. |
| `runs/<run>/<Dataset>/workflows/` | The starting workflow and every version the planner wrote. Files after round 1 are written by the planner model. |
| `runs/<run>/<Dataset>/workflows/results.json` | The validation score and estimated cost of each scored round. |
| `runs/<run>/<Dataset>/workflows/search_usage.json` | The planner's tracked cost estimate, including proposals that failed. This estimate omits Gemini's thinking tokens. |
| `runs/<run>/test_logs/` | The test results, one row per question. |
| `runs/<run>/model_calls/` | The requests and responses from successful calls, under `model_calls/{search,test}/`. Executor responses include the provider and billed cost. |

The rule for picking a workflow is in `upstream/selection.py`, and `upstream/test_pass.py` runs the chosen one on the test questions.

## Replaying a test without calling a model

Because every successful call is saved, you can run a test again from the saved calls. Set `AFLOW_CACHE_MODE=replay` and point `AFLOW_CACHE_DIR` at the run's saved test calls, then run `test_pass.py` from the AFlow checkout. Keep the original `--dataset`, `--workspace`, `--exec-model` and `--round`; use `--out-root` to save replay results in a separate folder. The cache raises an error for a missing or changed request and never contacts a model in replay mode. When a workflow sends identical requests at the same moment, a replay can hand back their saved answers in a different order. That is why 2 of our 15 tests replayed one answer differently.

## Costs

AFlow's own tracker prices calls at fixed rates. The paper's executor costs use the provider's actual bills, read from the saved calls. Planner costs are estimated from token counts. See `aflow/README.md` in the supplementary package or `submission/aflow/README.md` in the full repository.

## Older files in this folder

The folder name comes from an earlier comparison on Sports and FinQA. Their results are not used in the paper. The templates are still used: `SportsUnderstanding` supplies operators for MuSR and NaturalPlan, and `FinQA` supplies the math operators for MedCalc. `benchmarks_finqa.py` remains an import needed by the evaluator and `test_pass.py`.
