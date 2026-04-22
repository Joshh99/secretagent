# Tasks/Known bugs

## Experimental improvements

 * add `result.py rename --to '%O_oss2b' results/*` - to help cleanup results
 * look at pot failures and see if there is an easy way to improve them
 * add `result.py replay --case-name NAME results/*` - like quick-test
   but loads the config, turns up all the logging, and reruns that case.
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
 * should move common LLM values out of CLAUDE.me

## Core issues/bugs

 * output processing code is so fragile and messy, is there a way to fix that?
 * boxes should wrap text

## Caching

 * Check if disabling caching from the command-line works

## Code quality/etc

 * More guidance for claude/devs on defensive programming

## Known minor bugs

 * Running the simulate_pydantic with tools leads to a bunch of
   litellm task warnings, which are meaningless but annoying and
   seemingly hard to fix.
