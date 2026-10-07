# AFlow comparison protocol

Revised 2026-09-23 after the four-candidate pilot. This guides future runs. The
pilot and May MODO results were already observed, so neither is preregistered or
blind. Freeze this version before the next paid run and date any later changes.
Upstream AFlow is pinned at `3f457218`.

September 23 correction: AFlow's shared LLM usage counter made individual
case-cost rows cumulative, not per-example. Its original CSV writer took the
maximum row as the aggregate cost. For the clean Sports and FinQA held-out
passes, that aggregate matches the independent token-usage total, so their
aggregate costs remain valid. The archived patch now isolates usage by example
and sums case costs; earlier case-level cost rows should not be used.

## D1. Starting prompt

MODO's Sports task has four hand-written interfaces (`analyze_sentence`,
`sport_for`, `consistent_sports`, `are_sports_in_sentence_consistent`). The
primary AFlow prompt includes their descriptions. A top-level-only prompt is
an optional second condition.

| condition | AFlow starting prompt |
|---|---|
| `matched` | built from the top-level and LLM-backed sub-interface text and examples |
| `unmatched` | built from the top level interface docstring only |

The matched prompt shares information about the human decomposition, but does not
give AFlow MODO's executable workflow. The pilot showed that prompt framing
affects tasks differently. Treat the matched/unmatched gap as a prompt-condition
effect, not a clean estimate of human effort. Matched is primary; unmatched is
exploratory if time remains.

Seeds are emitted by a script that reads the docstrings out of `ptools.py`. Nothing is hand edited.
A seed prompt typed by a person invalidates the condition it belongs to.

## D2. Selection rule, Pareto fronts

Take the validation Pareto front over each run's candidates as (mean validation
score, mean validation cost per case). Evaluate each selected point once on the
same held-out test cases. The fixed-model MODO front is the AFlow comparison.
The historical mixed-model front is context only.

For a single number table cell, use highest mean validation score with ties broken by lowest mean
validation cost, which is a real criterion rather than an arbitrary one.

The patched `test_pass.py` selects this validation front. The three original
AFlow selection rules disagree: upstream `test_pass.py` breaks ties toward the
earliest candidate, `Optimizer.test` uses only the starting workflow, and
`interface.load_best_round` chooses the second item in its top-candidate pool.

## D3. Replicates, seeded, 3 minimum

Upstream AFlow has no `--seed`. Its only seeding call, `np.random.seed(42)` in
`benchmarks/utils.py:generate_random_indices`, sits in a function nothing imports, so it never runs.

The experiment patch adds `--seed` for both `np.random` and Python's `random`. Both are
needed: `select_round` draws parents with `np.random.choice`, and `DataUtils.load_log` uses
`random.sample` to choose the error examples that go into the mutation prompt.

Use three seeds per task and starting-prompt condition. Report each seed and
their range; three seeds provide only a rough uncertainty estimate.

## D4. Search budget, matched on case evaluations

The primary axis is total case evaluations during search. `Evaluator.graph_evaluate` sets
`va_list = None` on both branches, so AFlow evaluates the entire split each round and per round cost
is `validation_rounds * |validation split|`.

Set `validation_rounds = 1` and disable early stopping. Count the starting
workflow in AFlow's cap: 50 Sports, 43 FinQA, and 54 MuSR object candidates
per seed, each scored on 50 validation cases. These counts match the archived
MODO unique-configuration counts for planning. Report actual successful case
evaluations, failures, and proposals; a cap alone does not prove every case ran.

Report both axes: case evaluations and total search dollars including optimizer
spend. The fixed-model MODO comparison enumerates a smaller space and therefore
has a different search budget. Do not call it an equal-budget optimizer test.

## D5. Cost accounting, cross checked before any plot

The pilot cross-check ran an identical case through both pipelines with the
same model, prompt, and temperature 0. Both recorded 33 input tokens, one
output token, and $0.0000037. Recheck if either accounting path changes.

The Pareto axis is executor inference cost per case. Report search dollars
separately, including optimizer calls, failed proposals, and test-pass spend.
The Gemini OpenAI-compatible endpoint may omit thinking-token usage, so check
provider billing before calling the optimizer total exact.

The patched price table errors on unknown models. `BaseBenchmark.save_results_to_csv`
uses `df["cost"].max()`, valid here only because workflows return cumulative cost.

## D6. Operator set, one upstream bug fixed

The experiment patch fixes `Programmer`. `CodeFormatter.validate_response` returns `{"response": ...}` while
`Programmer.__call__` reads `.get("code")`, so it returns `None` every time and costs up to nine LLM
calls for nothing. The fix is in the runtime operator template archived under
`upstream/templates/FinQA/` and in `upstream/aflow_changes.patch`.

Three further bugs stay unfixed, recorded here so the choice is visible. `MdEnsemble.shuffle_answers`
collapses duplicate candidates through `solutions.index`. `ScEnsemble` raises `KeyError` on an out of
range solution letter. `XmlFormatter.validate_response` swallows its own `FormatError` and loses the
missing field name. Keeping the patch small keeps it auditable.

## D7. Three tasks

Use Sports Understanding, FinQA, and MuSR object placements. They have exported
validation and test cases plus scorers. Check case IDs across both systems. The
pilot used all three; do not describe these tasks as chosen blind.

## D8. Numerical meaning of competitive

This criterion was chosen after the pilot. For each task, set 101 cost budgets
equally spaced in log10 from $0.000001 to $0.01 per case. At each budget select
the highest-validation-accuracy candidate whose validation cost fits (ties:
lower cost, then stable candidate ID). If none fits, assign accuracy zero.
Freeze choices before using test. Let Q be the mean of those 101 test accuracies,
averaged equally over tasks and seeds. The author-chosen practical margin is
`Q_MODO - Q_AFlow >= -0.05`. Show every task's curve and difference as well.

For statistical support, the lower end of a 95% paired case-bootstrap interval
must exceed -0.05. If it crosses the margin, call the result inconclusive.
Three seeds give only coarse information about search variability. Do not
claim dominance when curves cross.

Use paired case bootstrap intervals for direct accuracy differences. Prediction-
powered inference is exploratory only if a cheap predictor and separate
unlabeled sample are defined before that analysis.

## D9. Models

| role | model |
|---|---|
| executor, primary | `gemini-2.5-flash-lite`, temperature 0 |
| optimizer | `gemini-3.1-pro-preview`, temperature 0 |
| optional second executor | `openrouter/openai/gpt-oss-120b`, temperature 0 |

The fixed-model MODO run must also fix every sub-tool model to Flash-Lite. A
second executor is optional after the primary runs. If only one task fits, use
FinQA and state the limited scope. Verify its current price before a cost plot.

## Run matrix

Fixed-model MODO: all methods in the current spaces, 6 Sports + 6 FinQA + 7
MuSR object = 19 configurations, nominally 950 validation-case evaluations.
Matched-prompt AFlow: 3 tasks x 3 seeds = 9 searches, with 147 candidates
per seed across tasks. The unmatched condition and second executor are optional.

For a separate search-method control, run fresh MODO NSGA-II and uniform random
search on the original full model-routing spaces for Sports and FinQA, three
paired seeds. Share the same starting population, validation cases, cache
policy, and actual unique-configuration budget. The archived 50 and 43 counts
are planning values, not guaranteed fresh budgets. MuSR random search is
deferred because its archived runtime is about 12 hours per seed serial.

## What would weaken the claim

If the primary numerical comparison falls below the margin, do not call MODO
competitive. If the random control is incomplete, remove any claim that
NSGA-II beats random search. Keep the historical mixed-model front separate
from the fixed-model comparison. Archive complete configurations, case IDs,
outputs, failures, token counts, costs, prompt hashes, code revisions, and time.
