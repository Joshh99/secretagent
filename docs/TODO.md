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
 * Current 2026-04-24  Several easy POT losses appear to be plumbing fixes rather than reasoning failures: eg. sandbox/code-extraction issues - eg. typing imports being blocked, fixable by replaying the cached generated code with typing allowed. Some runs can often return tuples like ("E", "04/11/1985") when the evaluator wants just (E). There are smaller similar issues from blocked json/fractions imports and no-code-block outputs. The low hanging fruits seem to be generic PoT robustness fixes: allow a few safe imports, improve code-block extraction, and normalize final answer shape. MUSR, NatPlan and Medcalc Rule failures look like strategy misses, as opposed to plumbing
 
 * look at finding ICL examples for pot, workflow, ... ?

## Tracking extensions

 * What's the use case for llm streaming in llm_util?

## Cleanups

 * fix ptools.py loading dependencies in benchmarks/tests
   * tests run independently but running all together in one pytest
	 call causes problems.
 * move subprocess out of optimizer and use expt
 * cleanup learn/examples.py, and traces.py
   - It should be a Learner
   - maybe a Learner should output an implementation config? that's more general
   - need to add filtering for iscorrect examples
 * make orchestrate a Learner
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
