# Changes - April 11

## Architecture explanations

Added some comments in CLAUDE.md on architectural goals, briefly:

 * "strategy" means "all interface -> implementation bindings"
 * strategies should be serializable in yaml
 * experimental results should be attached to strategy configs and dates
 * "learning" means "adding a new implementation"
 * inputs/outputs of learning should be trackable

## Hypervolume

Added hypervolume computation for a set of experimental results in
`results.py`. (See https://arxiv.org/abs/2005.00515)

 * Note: the hypervolume computation requires a 'reference point',
 which for cost-/correctness metrics is an upper bound on cost and a
 lower bound on correctness. These are computed dynamically from the
 results, so hypervolumes are NOT comparable across different set of
 results.  The awkward workaround is
 * use the `results.py refpoint results/*` command to find the reference point
 for all results
 * pass that into hypervolume as a reference point with the `--ref` option

Here's an example (where `sar` aliases `uv run python -m
secretagent.cli.results`) of correctly measuring the hypervolume
increase associated with (1) results from a model sweep for ptools used
in a fixed workflow (2) using the baseline/pot/workflow/react results
with a fixed model and (3) combining 1 and 2 

```
bash-3.2$ sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` model_sweep_results/*
sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` model_sweep_results/*
Hypervolume: 0.001562
Metrics: cost-, correct
Reference point: [-0.00308, 0.293333]

bash-3.2$ sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` results/*
sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` results/*
Hypervolume: 0.001899
Metrics: cost-, correct
Reference point: [-0.00308, 0.293333]

bash-3.2$ sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` results/* model_sweep_results/*
sar hypervolume --ref `sar refpoint results/*  model_sweep_results/*` results/* model_sweep_results/*
Hypervolume: 0.001904
Metrics: cost-, correct
Reference point: [-0.00308, 0.293333]
```

use the `results.py refpoint results/*` to find the reference point
 for all results

# Changes - April 10

## Code Distillation Learner

Added `CodeDistillLearner` — a new learner that prompts an LLM to
generate Python code from recorded input/output examples. Uses
multi-round refinement (generate → validate → feedback errors →
regenerate) and ensemble (multiple candidates per round, pick best).

 * **Learner base class** — added validation support: after training,
   learners can automatically evaluate on a holdout split. Also stores
   workflow trace context in training data so learners can see how an
   interface is called within the broader pipeline.

 * **Sandbox** — LLM-generated code runs in `LocalPythonExecutor`
   (same sandbox as PoT). Existing rote learner path is unchanged.

 * **CLI** — `uv run -m secretagent.cli.learn codedistill --interface NAME recordings/*`

 * **Makefile targets** — `codedistill_pipeline` in sports_understanding,
   `codedistill_cal_pipeline` in natural_plan.

# Changes - April 3

 * **Global results Makefile** — `benchmarks/Makefile` operates on
   all exported results across benchmarks. Run from `benchmarks/`:

```
make plot            # global cost-vs-correct → results_all.png
make plot_sport      # sport only
make plot_natural    # natural_plan only
make plot_rule       # rulearena only
```

 * **Parse post-processing** — any interface can now add a `parse`
   step via config. The parse interface inherits the original's return
   type and docstring, then wraps the output. Two modes:

   - `simulate` — LLM re-parses the raw output each time
   - `program_of_thought` — LLM generates regex/parsing code, then
     executes it in sandbox (cached across examples)
   - `direct` — user-provided parsing function (`fn=module.func`)

   Implementation: `_add_parse_wrapper()` in `core.py` creates a
   lightweight parse `Interface` on the fly and wraps the original
   `implementing_fn`. No new factories needed.

```
# LLM parse
ptools.calendar_scheduling.parse.method=simulate

# LLM generates code to parse (PoT)
ptools.calendar_scheduling.parse.method=program_of_thought
```

 * **Bug fix** — `cli/results.py` `ttest_rel` crash on boolean
   `correct` column; added `.astype(float)` cast.

# Changes - April 2

 * Refactored implement/ to make creating new Implementation.Factories
   less obscure.  Subclasses now override `setup()` and `__call__()`,
   and theres no build_fn that returns a closure.  **I believe this
   should not affect anyone unless they are implementing factories**.

```
 - setup(**builder_kwargs) configures per-interface state on self
 - __call__(*args, **kw) is the implementing function
```

 * I did this to facilitate fixing a bug: I intended you to be able to
 pass `model` in as a parameter to every Implementation.Factory and
 override the default configured model.  This is now fixed,
 and `sports_understanding/model_sweep.py` is an example.

# Changes - April 1

## Pipeline Optimizer (Phase 1)

Added multi-objective Pareto search to `src/secretagent/optimize/`:

 * `encoder.py` — maps DEAP integer vectors to/from dotlist config
   overrides. `SearchDimension(key, values)` defines one axis of the
   search space. `encode()` / `decode()` / `decode_dict()` handle the
   conversion; `space_size()` returns the Cartesian product size.

 * `pareto.py` — two search modes behind a single `run_nsga2()` entry
   point. Spaces with <= 20 configs get **exhaustive enumeration**
   (guarantees complete coverage). Larger spaces use **NSGA-II** via
   DEAP with categorical operators (uniform crossover, random-reset
   mutation). `EvalCache` wraps subprocess evaluation, caches results
   by chromosome, and assigns worst-case fitness (accuracy=0, cost=inf)
   to failed configs so they can't appear Pareto-optimal.

 * `viz.py` — `plot_pareto_frontier()` renders accuracy-vs-cost scatter
   plots with frontier points as filled stars and dominated points as
   hollow circles. Fixed y-axis (0-1) to prevent misleading auto-scaling.

 * **New dependency**: `deap` (LGPL-3.0, evolutionary algorithms toolkit).

### Design decisions

 * Subprocess-level integration: the optimizer generates dotlist
   overrides and passes them to the existing evaluator via subprocess.
   No changes to core infra, config system, or benchmark code.
 * Exhaustive/NSGA-II auto-selection avoids using evolutionary search
   on spaces small enough to enumerate (a lesson learned during
   development — NSGA-II on a 4-config space wastes budget on
   duplicates).

### Tests

 * `tests/test_encoder.py` — 20 tests covering encode, decode,
   round-trip, out-of-bounds, wrong-length, and space_size.

# Changes - March 29-30

## To help get organized results 

So it's easier to see what people are doing when updating code code,
and so it's easier to reproduce experiments:

 * Changed .gitignore to encourage people to **check in results/ and
   llm_cache** directories.

 * Extended the cli/results.py with an 'export' tool to copy results
 into benchmark/results. **Please export the interesting subset of your
 current results***.  Use these `expt_name`s
  * `structured_baseline`
  * `unstructured_baseline`
  * `workflow`
  * `pot`
  * `react`

If you're not putting subtasks in subdirectories, you should use the
`--as RELATIVE/PATH` option to file your results - see the CLI.md
file.

 * Generalized expt.py and moved it to cli - **Everyone should be able
 to use this now, not specialized ones.**  Discuss with me if you
 can't!

 * To speed up testing, separated out the benchmark tests that
 benchmark/tests, and enabled caching in the orchestrate tests.
 **Thanks to Joshua** for starting out with the benchmark tests, I
 encourage us all to follow!


## Other extensions

 * Extended the results.py with options for finding pareto-optimal
   experimental results (see docs/CLI.md)

 * Extended the cli/results.py with a 'plot' tool to visualize
   cost/performance tradeoffs

## Misc cleanups

 * Moved `sports_understanding`, `penguins_in_a_table`, and
 `geometric_shapes` under bbh (bbh="BIG Bench Hard" the source of all
 of these)

 * Collected all the implementations into an implement/ package, so
 `implement_core` => `implement.core`, and also moved
 `learn/implement_learn.py` => `implement/learnedcode.py`. As part of
 this you no longer need to import specific `implement_*` routines to
 populate the factory registry.

 * Moved Lex's optimze.py into an optimize/ package, and generalized
 the config space that is searched. (Now you can search over configs
 settings that can't be decomposed into independent dotlists, like
 implement_via={'method':direct, 'fn': foobar}


### Learning-related cleanups

 * Renamed the learned factory from `'learned'` to `'learned_code'`

 * Renamed CLI flag `--train-dir` to `--learned-dir`

 * Generalized `learn.base.Learner` to output configurations for a
 learned implementation, to cover Lex's PTP learner (and also to cover
 distillation via ICL)

 * Added `learn/HOWTO.md` for learner contributors

# Moved Changes - March 23

I cleaned up some of the result.py cli methods and fixed a few small
bugs.  CLI.md now summarizes the implemented cli operations.

Some folks asked about where this project is going with learning and
optimization.  Here's the end goal for the system as I see it now
(thanks to the folks that discussed this with me today!)

## The search space

The end goal is to search a space of potential agents for a task.  The
space is defined by a set of Interfaces, and potential Implementations
that can bd bound to these interfaces, as well as other options (like
`llm.model` which can be set for any individual implementation.)

**Status:** now we can specify any element in that space with a config
file.  We want to be able to specify the space itself as well, which
might look something like this:

```
ptools:
  analyze_sentence:
    method: [simulate, simulate_pydantic]
	llm.model: 
	 - together_ai/deepseek-ai/DeepSeek-V3.1
     - together_ai/openai/gpt-oss-20b
     - together_ai/Qwen/Qwen3.5-9B
  sport_for:
    method: [simulate, simulate_pydantic]
  ...
```

## Searching the space

We will add some methods for searching the space under
`src/secretagent/optimize`, presumably based on existing MODO
(multi-objective discrete optimization) packages.  These methods will
take a search space and find a set of Pareto-optimal configs.

To search the space, we need to evaluate a config with respect to
metrics we care about (for now, `correctness` and `cost`).  We can do
this now with CLI tools.

## Extending the space

We will build out the `src/secretagent/learn` package with other ways
of learning and combining implementations.  The first step will be to
run learning methods manually with CLI tools, and the next step will
be to implement some methods that look at the current Pareto-optimal
configs and suggest learning ops for specific implementations.

## Overview of the process

The final process should look like this.

 * Initialize the space with some interfaces and strategies, like
   ReAct and PoT
 * Using the training and validation sets:
   * **repeat** until convergence or budget is exhausted
     * search for Pareto-optimal configs
     * run one or more learning operations to expand the search space
 * Test the Pareto-optimal configs on the test set


# Changes - March 22

There is a very simple and barely-tested example of a learner now, in
the Makefile targets of `benchmarks/sports_understanding`.

To learn a new implementation of some interface `foo` you need to

 * Run an evaluation on the **training** split using some
implementation strategy that includes running some implementation of
`foo`, and the `evaluate.record_details=true` flag set.  This
generates distillation data for `foo` (i.e., input-output pairs).
Those results are filed in a directort specified by
`evaluate.result_dir` (in the Makefile `recordings`).

* Run a cli to do the learning.  That will collect the distillation
data into a working directory specified by --train-dir (in the
Makefile, `training`) and write the result of learning as a Python
function under `training/*foo__rote`.

* Configure `foo` with the learned implementation, by using
`foo.implement_via('learned',learner='rote',backoff=True)` or the
corresponding config options.  This will use rote learning to bypass
the original implementation and back off to the implementation used to
generate distillation data when the rote-learned function returns
None.

# Changes - March 13-Mar 18

## Documentation and examples

In `benchmarks/sports_understanding` is an example of evaluating untrained
models for a simple task, with docs in `benchmarks/HOWTO.md`.


## LLM cost tracking 

New `costs` CLI tool with `extract_cached_stats` for analyzing LLM
spending from the cache
  - For example, running in benchmarks/sports_understanding: uv run python -m secretagent.cli.costs llm_cache
  generates this report:

```
  1091 cached LLM calls

       input_tokens  output_tokens      latency         cost
count   1091.000000    1091.000000  1091.000000  1091.000000
mean    1220.851512     197.437214     2.570375     0.002208
std     2341.485373     147.909843     2.503594     0.003005
min       79.000000      12.000000     0.377901     0.000306
25%      308.000000     125.000000     1.340060     0.000915
50%      343.000000     163.000000     1.677382     0.001105
75%      706.000000     207.000000     2.418103     0.001667
max     8375.000000     750.000000    16.032463     0.012125

Totals:
  input_tokens:  1331949
  output_tokens: 215404
  latency:       2804.3s
  cost:          $2.4090
```

## Analyzing results

Added filtering, `--most-recent` default, dotlist support, and config
comparison.

### Example of getting averages

```
% uv run python -m secretagent.cli.results average --metric correct --metric cost results/*
                                                correct                cost          
                                                   mean       sem      mean       sem
path                                                                                 
results/20260316.183513.workflow               0.973333  0.018728  0.004200  0.000115
results/20260317.131602.pot                    0.960000  0.022780  0.006137  0.000158
results/20260317.141836.zeroshot_structured    0.933333  0.028997  0.001071  0.000018
results/20260317.143224.zeroshot_unstructured  0.933333  0.028997  0.001525  0.000032
results/20260317.145854.react                  0.813333  0.045295  0.013779  0.000475
```

### Finding the differences in configs between experiments

```
% uv run python -m secretagent.cli.results compare-configs results/*
properties of results/20260317.131602.pot:
  evaluate.expt_name=pot
  ptools.are_sports_in_sentence_consistent.method=program_of_thought
  ptools.are_sports_in_sentence_consistent.tools=['ptools.analyze_sentence', 'ptools.sport_for', 'ptools.consistent_sports']
properties of results/20260317.145854.react:
  evaluate.expt_name=react
  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic
  ptools.are_sports_in_sentence_consistent.tools=['ptools.analyze_sentence', 'ptools.sport_for', 'ptools.consistent_sports']
...
```


### Running paired tests

```
% uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*
                                    _comparison   n  correct_t  correct_p     cost_t        cost_p
0                                  pot vs react  75   3.238021   0.001803 -15.994221  1.011472e-25
1                               pot vs workflow  75  -1.000000   0.320569  20.859531  1.019114e-32
2                    pot vs zeroshot_structured  75   0.704730   0.483190  32.051952  3.901321e-45
3                  pot vs zeroshot_unstructured  75   0.704730   0.483190  29.922488  4.408926e-43
4                             react vs workflow  75  -3.754363   0.000344  20.763785  1.363903e-32
5                  react vs zeroshot_structured  75  -2.111829   0.038077  26.699630  9.997003e-40
6                react vs zeroshot_unstructured  75  -2.111829   0.038077  25.270505  3.938396e-38
7               workflow vs zeroshot_structured  75   1.136089   0.259585  26.704393  9.878003e-40
8             workflow vs zeroshot_unstructured  75   1.136089   0.259585  23.529151  4.356200e-36
9  zeroshot_structured vs zeroshot_unstructured  75   0.000000   1.000000 -13.698474  5.518676e-22
```

### Other software changes

- **Program of Thoughts** (PoT) factory
- **Linting/typing** — Integrated ruff and mypy, fixed lint errors and
  type annotation bugs
- **Configuration of tools is consistent** for PoT and Pydantic React agent
- **More configuration can be done just from a config files**
- **New MUSR benchmark** — Added a benchmark harness with 3 splits and zero-shot chain-of-thought baseline
- **More tests** - up to 157 tests, now more nc lines of test code than nc lines of code in the core system
