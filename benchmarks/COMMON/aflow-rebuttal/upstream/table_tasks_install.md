# Table 2/3 task adapters: what is installed and how to reproduce it

Applied and verified in `C:/Users/STUDENT/aflow-b`. Not yet applied to
`C:/Users/STUDENT/aflow`, because the `table2_musr3` MuSR Object search is still
reading those files. Apply it there once that search finishes.

Reproducing a checkout from scratch is one command, which now installs and
verifies everything below:

```
uv run python scripts/prepare_aflow.py --aflow-dir <dir> --apply
```

## Executor model

Tables 2 and 3 were produced with DeepSeek-V3.1, so the AFlow runs use the same
model. It is reached through OpenRouter, since the Together key no longer works.

OpenRouter routes a model to whichever host is cheapest unless told otherwise,
and its default for this model is DeepInfra serving it at **fp4**. The table
columns were served by Together at fp8, so taking the default would put a more
compressed model in the column it is being compared against. `config2.yaml`
therefore pins SiliconFlow, which is fp8 and carries the largest context of the
fp8 hosts (163,840), which the long RuleArena NBA prompts need:

```yaml
    extra_body:
      provider:
        only: ["SiliconFlow"]
        allow_fallbacks: false
```

Verified on 2026-09-23 that the pin is enforced rather than advisory: unpinned
returns `provider: DeepInfra`, pinned returns `provider: SiliconFlow`, and a
provider that cannot serve the model returns HTTP 404 instead of quietly
rerouting. Fallbacks are off so a throttle fails loudly rather than switching
quantization partway through a run.

`extra_body` joins the request, so it is part of the call-cache key. Changing
the pin correctly invalidates replay instead of reusing calls from another host.

`ModelPricing` carries the pinned host's own rate ($0.27 per 1M prompt, $1.00
per 1M completion). That figure is only right while this pin holds; repinning
means repricing.

## Added files

Installed by `prepare_aflow.py` and verified on every run:

| archived | installed as |
|---|---|
| `scorers_natural_plan.py` | `benchmarks/scorers_natural_plan.py` |
| `scorers_medcalc.py` | `benchmarks/scorers_medcalc.py` |
| `benchmarks_naturalplan_meeting.py` | `benchmarks/naturalplan_meeting.py` |
| `benchmarks_naturalplan_trip.py` | `benchmarks/naturalplan_trip.py` |
| `benchmarks_rulearena_nba.py` | `benchmarks/rulearena_nba.py` |
| `benchmarks_medcalc.py` | `benchmarks/medcalc.py` |
| `benchmarks_finqa.py` | `benchmarks/finqa.py` |

The two scorer files are verbatim copies of `benchmarks/natural_plan/eval_utils.py`
and `benchmarks/medcalc/accuracy.py`. `tests/test_aflow_table_adapters.py` fails
if either drifts, so re-copy rather than edit.

`benchmarks_finqa.py` had been archived since the first pilot but was never
listed in `COPIES`, so a checkout built by `prepare_aflow.py` alone was missing
it and `scripts/evaluator.py` would not import. Found while building `aflow-b`.

## Registered datasets

Six names added to `run.py`, `scripts/evaluator.py` and `test_pass.py`. The
first three are tracked AFlow files and are carried by the rebuilt
`aflow_changes.patch`; `test_pass.py` is an added file and is carried by the
archive above.

| dataset | benchmark class | type | operators |
|---|---|---|---|
| `MuSRMurderMysteries` | `MuSRObjectBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `MuSRTeamAllocation` | `MuSRObjectBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `NaturalPlanMeeting` | `NaturalPlanMeetingBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `NaturalPlanTrip` | `NaturalPlanTripBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `RuleArenaNBA` | `RuleArenaNBABenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `MedCalcTest` | `MedCalcBenchmark` | math | Custom, ScEnsemble, Programmer |

MuSR murder and team reuse the object scorer because all three are an exact
match on the choice index. MedCalc is numeric, so it takes FinQA's math operator
set rather than `AnswerGenerate`.

## MedCalc scores through score_case, not calculate_score

A MedCalc case cannot be graded from its gold value alone. It needs that case's
lower limit, upper limit, output type and category, because formula categories
allow a tolerance while rule categories require an exact match. Those four
fields ride alongside `target` in each exported row and
`MedCalcBenchmark.score_case(problem, prediction)` reads them.
`calculate_score` raises `NotImplementedError` so the difference cannot be
missed. Anything calling benchmarks generically must special-case this dataset.

## Before trusting the MedCalc Rules comparison

The saved MedCalc Rules column was scored on 2026-04-25, before commit
`32f797fa` on 2026-05-01 fixed the rule-category test in `accuracy.py`. Under
the old code the exact-match branch never fired for `risk`, `diagnosis` or
`severity`, so those cases got the 5% tolerance meant for formulas.

Two of 380 cases are affected. The column reads 0.4974 where today's scorer
gives 0.4921. Rescoring needs no model calls, only a recompute from the saved
per-case rows. Do that before putting an AFlow number beside it.
