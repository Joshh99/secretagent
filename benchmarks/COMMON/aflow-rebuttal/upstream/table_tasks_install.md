# What the setup script adds to AFlow

`scripts/prepare_aflow.py --aflow-dir <dir> --apply` turns a plain copy of AFlow (version `3f457218`) into the one we used. Running it again without `--apply` only checks that everything below is in place.

## The executor model

The paper's executor model is DeepSeek-V3.1, so AFlow runs it too. AFlow reaches it through OpenRouter, a service that forwards each request to one of several companies hosting the model. By default, OpenRouter spreads requests across hosts, giving cheaper hosts more weight, and hosts can run differently compressed versions of the same model. To keep every call on one host, the config entry `deepseek-v31-atlascloud` in `config2.yaml` sends requests only to AtlasCloud and turns fallback off:

```yaml
    extra_body:
      provider:
        only: ["AtlasCloud"]
        allow_fallbacks: false
```

With fallback off, a host that cannot serve a request returns an error rather than quietly sending it somewhere else. The provider setting is part of each saved request, so a replay can never reuse an answer from a different host.

AFlow's own cost tracker prices calls at fixed rates ($0.27 per million input tokens and $1.00 per million output tokens). The paper's executor costs use the provider's actual bills instead, read from the saved calls.

## Files added to AFlow

| Archived here | Installed as |
|---|---|
| `scorers_natural_plan.py` | `benchmarks/scorers_natural_plan.py` |
| `scorers_medcalc.py` | `benchmarks/scorers_medcalc.py` |
| `benchmarks_naturalplan_meeting.py` | `benchmarks/naturalplan_meeting.py` |
| `benchmarks_naturalplan_trip.py` | `benchmarks/naturalplan_trip.py` |
| `benchmarks_rulearena_nba.py` | `benchmarks/rulearena_nba.py` |
| `benchmarks_medcalc.py` | `benchmarks/medcalc.py` |
| `benchmarks_finqa.py` | `benchmarks/finqa.py` |

The two scorer files are exact copies of this repository's own scorers, `benchmarks/natural_plan/eval_utils.py` and `benchmarks/medcalc/accuracy.py`, so AFlow is graded the same way as every other method. In the full repository, `tests/test_aflow_table_adapters.py` fails if either copy drifts from its original, so if you change one, copy it again rather than editing the archived file.

## Datasets registered in AFlow

These six names are added to AFlow's `run.py`, `scripts/evaluator.py` and `test_pass.py`. The first two files are AFlow's own and change through `aflow_changes.patch`; `test_pass.py` is one of the added files.

| Dataset | Benchmark class | Type | Operators |
|---|---|---|---|
| `MuSRMurderMysteries` | `MuSRObjectBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `MuSRTeamAllocation` | `MuSRObjectBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `NaturalPlanMeeting` | `NaturalPlanMeetingBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `NaturalPlanTrip` | `NaturalPlanTripBenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `RuleArenaNBA` | `RuleArenaNBABenchmark` | qa | Custom, AnswerGenerate, ScEnsemble |
| `MedCalcTest` | `MedCalcBenchmark` | math | Custom, ScEnsemble, Programmer |

The three MuSR tasks share one scorer, because each is graded by whether the chosen answer number is exactly right. MedCalc uses FinQA's math operator set, including `Programmer`, rather than `AnswerGenerate`. Its scorer also handles date and weeks-and-days answers.

## How MedCalc is graded

A MedCalc answer cannot be graded from the correct value alone. Each exported question carries its lower limit, upper limit, output type and category next to `target`, the correct answer. `MedCalcBenchmark.score_case(problem, prediction)` passes these fields to the shared scorer and uses its `is_within_tolerance` result. Numeric formula answers normally allow 5% error, or an error of at most 0.05 when the correct answer is zero. Numeric rule answers need an absolute error below 0.01, which the scorer calls an exact match. Dates and weeks-and-days answers use the scorer's own exact-match rules. The supplied lower and upper limits are checked separately and do not decide this score. The generic `calculate_score` raises `NotImplementedError`; code that grades several datasets must call `score_case` for MedCalc.

## One caveat about the MedCalc Rules column

The saved MedCalc Rules result of the Engineered Workflow Baseline (its run of 25 April 2026) was scored before a fix on 1 May 2026 (commit `32f797fa` in `benchmarks/medcalc/accuracy.py`) to how the scorer treats rule questions. Before the fix, rule questions (`risk`, `diagnosis` and `severity`) were given the 5% tolerance meant for formula questions. Two of the 380 Rules questions are affected: rescored with today's scorer, that result's 0.4974 becomes 0.4921. Rescoring needs no model calls, only the saved answers. Rescore this saved column before comparing it with an AFlow result graded by the current scorer.
