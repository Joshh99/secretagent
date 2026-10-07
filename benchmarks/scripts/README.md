# Helper scripts for the code distillation runs

These scripts ran parts of the code distillation experiments, together with the top-level `benchmarks/run_*.sh` scripts. They started as one-off files in `/tmp/*`. Code distillation asks a strong coding model (Claude Opus here) to replace an LLM-based tool, or a whole workflow, with plain Python. The paper uses three settings, called class 1, 2 and 3 below:

- class 1: distill single pseudo-tools inside the hand-built workflow
- class 2: distill the whole workflow over the hand-built tools
- class 3: distill the whole workflow over the learned tools

| File | What it does |
|---|---|
| `musr_obj_team_full.sh` | Runs the full pipeline (phases A to E) for MuSR Object and MuSR Team, the two MuSR tasks the original main pipeline did not cover. For each task: record training runs of the workflow and of ReAct, score the starting workflow on validation, run the class 1, 2 and 3 Opus distillations, then score each result on validation. |
| `tabmwp_full.sh` | The same full pipeline (phases A to E) for TabMWP. It starts from `conf/workflow_incontext.yaml`, a workflow with four pseudo-tools: `identify_operation`, `extract_relevant_values`, `compute_answer` and `format_answer`. |
| `fill_missing_vals_v2.sh` | Scores Opus distillation results that were learned but never scored on validation: `c1_v4` (MuSR Murder, MedCalc, Geometric Shapes, Date Understanding, RuleArena), `c2_v4` (FinQA, RuleArena NBA and Tax) and `c3_v4` (FinQA, Calendar). It skips `c1_v1` on purpose, because that one was intentionally not run. |
| `fix_c2c3_failed.sh` | Reruns class 2 for MuSR Object, MuSR Team and TabMWP, which had failed without an error message. The cause was data in the raw HuggingFace format, which `Dataset.model_validate_json` rejected. The script (1) converts the data with each benchmark's own `expt.load_dataset()`, (2) reruns class 2 one task at a time, and (3) rescores class 3 with the learned tools run as LLM calls (`simulate`) rather than as learned code (`learned_code`). |
| `fix_meeting_v3.sh` | Reruns class 2 for Meeting Planning after converting `golden_plan` from a list to a string. Without the conversion, the distilled code returns a list, which matches the holdout data 90% of the time, but the validation scorer expects a string and marks every answer wrong. With the conversion, about 98% match. |
| `scan_all_vals.py` | Walks every saved validation folder (`val_results_full/`, `val_results/` and archived `results/`), the shared code distillation results and the shared orchestrator test results. For each run it collects the number of questions, accuracy, cost, model, split and run name. It removes the parentheses around BBH answers before scoring, and splits MedCalc into Formulas and Rules when the category is available. It writes `/tmp/all_vals_scan.csv`. |
| `build_master_table.py` | Reads `/tmp/all_vals_scan.csv` and writes a comparison table of accuracy, cost and number of questions to `/tmp/master_table.md` and `benchmarks/COMMON/master_table.md`. It drops empty columns, marks runs that are still going, adds the `v1_*` columns from the first version's write-up, and labels both the orchestrator and orchestrator-induced columns as held out test results. |

## Running them

The scripts expect to be started from the top folder of the repository. Each one sets a `ROOT` variable to the repository folder, which you need to change to the folder on your machine, and reads API keys from `$ROOT/.env` with `set -a; source ...; set +a`. When run in the background, they write progress to `benchmarks/codedistill_logs_v2/` and their main logs to `/tmp/`. Running them calls paid models.
