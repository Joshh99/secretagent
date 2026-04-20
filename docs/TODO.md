# Tasks/Known bugs

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

## Learning methods

Thoughts on distilling react traces.

Add a learn/distill_pot.py
 * takes every pot function for every example
 * uses simulated interfaces to convert to a 'canonical' functional form
   "def workflow(...) -> ..."
 * uses ast to rename all the variables to v01, v02, ...
 * hashes them to get a smaller set of functions
 * computes coverage of each function (correct/incorrect)
 * does some sort of greedy set cover or incrementally calls a
   simulated interface to refactor the workflows into one program

## Code quality/etc

 * More guidance for claude/devs on defense programming
 * Standardize implementation strategies: [un]structured_baseline, pot, workflow, react

## Known minor bugs

 * Running the simulate_pydantic with tools leads to a bunch of
   litellm task warnings, which are meaningless but annoying and
   seemingly hard to fix.
