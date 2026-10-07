# How AFlow works, and what it means for the comparison

Source: FoundationAgents/AFlow at `3f457218`, before the experiment patch. Knowledge graph at `C:/Users/STUDENT/aflow/.ua/knowledge-graph.json`
(179 nodes, 491 edges, 8 layers, 12 tour steps). Every claim here was read in the source.

## The search loop

`run.py` builds an `ExperimentConfig` and passes it to `Optimizer.optimize`, which repeats four steps.

1. Pick a parent. `DataUtils.get_top_rounds(sample)` pools the top `sample` rounds by mean validation
   score, then moves round 1 to the front of that list if it is already in it. Round 1 gets position
   priority, not guaranteed membership. `select_round` then draws stochastically with
   `p = 0.3 * uniform + 0.7 * softmax(0.2 * (s - max s))` over scores scaled by 100.
2. Mutate it. The optimizer LLM rewrites the parent's `graph.py` and `prompt.py`. The reachable
   neighborhood is defined in `WORKFLOW_OPTIMIZE_PROMPT` rather than in code: one detail point per
   step, at most five changed lines, graph complexity capped at 10.
3. Evaluate it. `validation_rounds` repeats on the validation split.
4. Check convergence. `check_convergence(top_k=3)`, with `z` and `consecutive_rounds` left at their
   defaults of 0 and 5.

## Findings that change the experiment design

### 1. Upstream Programmer returns nothing

`CodeFormatter.validate_response` returns `{"response": sanitized_code}` at `formatter.py:176`.
`Operator._fill_node` passes that dict through unchanged. `Programmer.__call__` then reads
`code_response.get("code")`, which is always `None`, and returns
`{"code": None, "output": "No code generated"}`.

This happens on the success path, whatever the response looks like. The earlier RUNBOOK was right
that Programmer was dead, but blamed markdown fences. The cause is a key name mismatch in AFlow
itself. A workflow that selects Programmer pays for up to nine LLM calls and gets nothing back.

The experiment patch fixes the runtime template copy as well as the script copy.

### 2. The July Sports conclusion was invalid

The July run tied several candidates at 0.76 validation and appeared to find no
improvement. The September 23 pilot repaired the runtime `Programmer` operator
and removed a token cap that prevented the proposing model from producing a
workflow. In four candidates, Sports improved from 0.84 to 0.92 on validation
and 0.84 to 0.94 on test; FinQA also improved. The earlier result cannot support
a claim that AFlow's search stalls. The pilot is small and was inspected during
debugging, so full runs still need validation-only selection and independent
seeds.

### 3. Convergence is an equality test

Setting `z = 0` makes the tolerance `z * sigma_delta_y` zero, so the rule becomes: stop after five
consecutive rounds in which the top 3 mean is unchanged. The standard deviation terms never affect
the outcome. The round 9 stop in the earlier run was therefore correct, and the `validation_rounds`
change from 3 to 1 was harmless. Neither `z` nor `consecutive_rounds` is reachable from the CLI.

### 4. Upstream search is not seeded

The one seeding call in the codebase is `np.random.seed(42)` inside
`benchmarks/utils.py:generate_random_indices`. Nothing imports that function or `split_data_set`.
`BaseBenchmark.load_data` reads the JSONL and filters by index without touching numpy, so the seed
never runs on the Sports or FinQA path. The experiment patch adds a `--seed` flag.

Three places consume a global RNG during a search: `data_utils.py:73` uses `np.random.choice` for
parent selection, `data_utils.py:138` uses `random.sample` to pick the error examples that go into
the mutation prompt, and `operators.py:383` uses `random.shuffle` in `MdEnsemble`. The first two are
on every run. There is no `--seed` flag, so independent replicates require adding one that seeds
both numpy and Python at startup.

### 5. Three definitions of "best round"

| where | rule |
|---|---|
| `test_pass.py:best_round` | highest mean validation; `max(sorted(means), ...)` breaks ties to the earliest round |
| `Optimizer.test` | hardcodes `rounds = [1]` |
| `interface.load_best_round` | returns `top_rounds[1]`, index 1, the second entry |

The reported number depends on which one is used, so the choice has to be stated.

### 6. Cost is aggregated with max, not sum

`BaseBenchmark.save_results_to_csv` computes `t_cost = df["cost"].max()`. This works only because
each workflow returns a cumulative `llm.get_usage_summary()["total_cost"]`, so the last row to
finish carries the running total. It is correct here and fragile, which is why the cross-check
against the secretagent harness matters.

`AsyncLLM` counts only `prompt_tokens` and `completion_tokens` against a local price table.
`ModelPricing.get_price` returns 0 for an unlisted model without warning. Thinking tokens have no
representation, which accounts for the optimizer cost undercount disclosed in the earlier run.

### 7. Smaller things that still matter

Upstream declares `--check_convergence` as `type=bool`, so `bool("False") == True` and early
stop cannot be disabled from its CLI. The experiment patch corrects this.
`--validation_rounds` defaults to 1 in `run.py` but 5 in
`Optimizer.__init__`. Both branches of the `is_test` check in `Evaluator.graph_evaluate` set
`va_list = None`, so validation and test both run the entire split, and subset size comes from the
exported file. `max_retries = 1` means the retry block never retries, so one exception burns a round
and logs `score = None`. Exhausted benchmark retries return score 0 and cost 0, so hard failures
lower the mean while staying invisible in cost totals. Round numbering is offset by one, since
`_optimize_graph` evaluates round `self.round + 1`.

## AFlow to MODO

| AFlow | MODO / secretagent |
|---|---|
| workflow `graph.py` plus `prompt.py` | an implementation strategy, interface to implementation bindings |
| operator (`Custom`, `ScEnsemble`, `Review`) | a registered `Implementation.Factory` |
| evaluated candidate | a MODO configuration evaluation |
| `select_round` stochastic parent draw | NSGA-II selection over the population |
| optimizer LLM rewriting the graph | learner producing a new implementation config |
| `results.json` per round means | savefile experiment results |
| validation split, full split evaluation | `Dataset` valid split via `Dataset.configure` |
| `check_convergence` early stop | generation budget |

Both search a space of compositions and select on validation. The structural difference is that an
AFlow candidate is free-form generated code constrained only by a prompt, while a MODO candidate is
a binding drawn from a registered factory set. AFlow explores a larger space with a weaker gradient.
MODO explores a smaller serializable one. That is why matching budget by case evaluations is the
fairness axis that makes sense.
