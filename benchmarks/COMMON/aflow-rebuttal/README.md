# AFlow comparison

External workflow-optimizer baseline for the ICLR submission.

## What is compared

AFlow searches over workflows written as Python code. MODO searches over interface
bindings. Both select on validation and are scored on a held-out split with the same
cases, the same scorer and the same executor model.

The primary AFlow run uses a prompt built from the task's interface text and
three search seeds on each of Sports, FinQA, and MuSR object placements. An
optional second prompt uses only the top-level description. The prompt gap
measures a prompt-condition effect; it does not isolate human effort.

The protocol and run sizes are in `PREREGISTRATION.md`. The September 23 pilot
informed that protocol. `AFLOW-BREAKDOWN.md` explains the repaired defects.

## Prepare a fresh AFlow checkout

Clone [FoundationAgents/AFlow](https://github.com/FoundationAgents/AFlow) at
the pinned commit:

    git clone https://github.com/FoundationAgents/AFlow.git ../aflow
    git -C ../aflow checkout 3f457218fc716093fe53f6df8a5d5e6379d66346

Install its pinned `requirements.txt` in `../aflow/.venv`. Copy
`upstream/config2.yaml.redacted` to
`../aflow/config/config2.yaml` and replace the placeholders with your own keys.
Keep that local config out of Git. From this experiment repo, run:

    uv run python scripts/prepare_aflow.py --aflow-dir ../aflow --apply
    uv run python scripts/export_aflow_datasets.py --aflow-dir ../aflow
    uv run python scripts/prepare_aflow.py --aflow-dir ../aflow

The setup script applies the archived patch and copies the four added AFlow
files. The run launcher checks these files and the six dataset SHA-256 hashes
before spending anything. The exact generated workflows and per-case outputs
must be published with the results. Hosted models can change, so a rerun may
not produce identical answers even with temperature zero.

## Regenerate the table

    uv run python scripts/collect_aflow_results.py --runs-root C:/Users/STUDENT/aflow/runs

Reads every cell's artifacts, writes `results.csv`, and prints the table.

## Run one AFlow search

    uv run python scripts/run_aflow_cell.py --aflow-dir ../aflow --task sports \
        --condition matched --seed 1 --max-candidates 50 --dry-run

Drop `--dry-run` to execute after checking the printed inputs. One run has its
own workspace under `runs/`, with a `manifest.json` recording code revisions,
dataset and prompt hashes, archived operator-template hashes, and commands.

## Where things live

| path | what |
|---|---|
| `runs/<run>/manifest.json` | inputs for one run |
| `runs/<run>/<Dataset>/workflows/` | seed plus every generated candidate |
| `runs/<run>/<Dataset>/workflows/results.json` | per-candidate validation scores |
| `runs/<run>/<Dataset>/workflows/search_usage.json` | optimizer spend, including failed proposals |
| `runs/<run>/test_logs/` | held-out pass, per-case rows |
| `runs/<run>/model_calls/{search,test}/` | requests and responses from live model calls |
| `upstream/` | pinned commit, patch, benchmark classes, selection, checksums |

Files under `workflows/round_*/` after round 1 are written by AFlow's optimizer model.
That is the method working as intended.

Live runs now **record** each successful model call but never read from the
recordings. This preserves the observed answers and token-cost measurements.
Set `AFLOW_CACHE_MODE=replay` and `AFLOW_CACHE_DIR` to the saved `model_calls/test`
directory when rerunning `test_pass.py` on the same workspace; a missing or
changed request fails before contacting a provider. We have unit-tested this
mechanism but have not yet verified a complete warm replay of an experiment.
Model-call records contain benchmark inputs and outputs, so check their sharing
terms before putting them in a public anonymous repository. The earlier Sports
and FinQA runs predate recording and cannot be replayed this way.

The pinned AFlow code originally shared one running token-cost counter across
concurrent examples. The experiment patch now measures each example's own model
calls, including calls made before a workflow error, and sums those case costs.
The earlier clean Sports and FinQA *aggregate* costs were checked against their
independent token-usage totals and match; their old individual case-cost rows
were cumulative and should not be interpreted as per-case costs.

Tables 2 and 3 use all 106 MuSR Object test cases, while the optimizer control
uses a 50-case held-out subset. From this repo, export the full split with
`uv run python scripts/run_aflow_table_musr.py --aflow-dir ../aflow --prepare-only`.
After the search finishes, run
`../aflow/.venv/Scripts/python.exe scripts/run_aflow_table_musr.py --aflow-dir ../aflow --score-only`
(use `.venv/bin/python` on Unix). This selects on validation and reports
accuracy and USD per 100 test examples.

## Earlier pilot

A July run on Sports and FinQA is retained for debugging only. Its pipeline
had a broken `Programmer` operator and a token cap that silenced workflow
proposals. A four-candidate September 23 pilot, after those repairs, improved
Sports and FinQA. Those pilot numbers are not the planned three-seed results.
