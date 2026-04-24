# Tasks/Known bugs

## Experimental improvements

 * musr checked in experiments are with deepseek v3, not v3.1
 * some bbh issues
   * geometric_shapes has examples in the ptool code - should it?
 * saved configs
   * maybe evaluate.interface should be in conf.yaml - right now it's hard to find
   * 
 * add `result.py rename --to '%O_oss2b' results/*` - to help cleanup results
 * look at pot failures and see if there is an easy way to improve them - 
 * Current 2026-04-24  Several easy POT losses appear to be plumbing fixes rather than reasoning failures: eg. sandbox/code-extraction issues - eg. typing imports being blocked (penguins), fixable by replaying the cached generated code with typing allowed. Some runs can often return tuples like ("E", "04/11/1985") (especially in datetime tasks) when the evaluator wants just (E). There are smaller similar issues from blocked json/fractions imports and no-code-block outputs. The low hanging fruits seem to be generic PoT robustness fixes: allow a few safe imports, improve code-block extraction, and normalize final answer shape. MUSR, NatPlan and Medcalc Rule failures look like strategy misses, as opposed to plumbing
 
 * look at finding ICL examples for pot, workflow, ... ?

## Tracking extensions

 * What's the use case for llm streaming in llm_util?

## Cleanups

 * Done 2026-04-24: fix ptools.py loading dependencies in benchmarks/tests.
   Root cause was two independent forms of cross-test contamination —
   (a) `sys.path` / `sys.modules` collisions between benchmark dirs that
   share top-level module names (ptools, expt, evaluator, ...), so bare
   `import ptools` inside `_import_*()` helpers could resolve to the
   wrong benchmark's file; and (b) `config.GLOBAL_CONFIG` leakage across
   `_run_eval` calls, which `OmegaConf.merge` kept accumulating. Fixed
   by adding a `load_benchmark_modules()` helper in
   `benchmarks/tests/conftest.py` that re-pins `sys.path[0]`, purges
   colliding `sys.modules` entries, and chdirs for import-time template
   reads; plus `GLOBAL_CONFIG = OmegaConf.create()` resets in
   rulearena/tabmwp/sports_understanding `_run_eval` (matching the
   pattern already in designbench/natural_plan). Also fixed a stale
   `zeroshot.txt` path in natural_plan's unstructured dotlist and made
   designbench smoke tests skip when the dataset is unbuilt.
   Full `benchmarks/tests` suite now runs 75 passed / 10 skipped / 0
   failed in one pytest process; previously ~17 failed when mixed.
 * move subprocess out of optimizer and use expt
 * cleanup learn/examples.py, and traces.py
   - It should be a Learner
   - maybe a Learner should output an implementation config? that's more general
   - need to add filtering for iscorrect examples
 * Done 2026-04-24: make orchestrate a Learner — OrchestrationLearner in
   learn/orchestrate_learner.py, CLI slimmed to a thin wrapper, implementation.yaml
   emitted and consumable via `direct` + `fn: __learned__.<attr>, learner: orch_learner`
 * should check if rulearena tests drop's stuff in results_dir or not

## Core issues/bugs

 * boxes should wrap text

## Caching

 * Done 2026-04-24: checked command-line caching overrides with a one-example
   `secretagent.cli.expt` run using Gemini and isolated temp cache dirs.
   `cachier.enable_caching=true` wrote 1 cache file; `false` wrote 0, so
   disabling caching from the command line works.

## Code quality/etc

 * More guidance for claude/devs on defensive programming

## Known minor bugs

 * Running the simulate_pydantic with tools leads to a bunch of
   litellm task warnings, which are meaningless but annoying and
   seemingly hard to fix.
