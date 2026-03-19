# Setting up a benchmark

This is a work-in-progress!

## Directory structure

Your directory structure will look something like this:

├── conf
│   └── conf.yaml      # configuration settings used in all or most experiments
├── data
│   ├── ...
│   ├── BACKSTORY.md   # provenance of the data
│   ├── test.json      # should have train/test/validation splits
│   ├── train.json
│   └── valid.json
├── expt.py            # your driver program
├── llm_cache          # caches llm calls
├── Makefile           # optional, for repeated tasks
├── prompt_templates   # optional, if you have hand-constructed prompts
│   └── zeroshot.txt
├── ptools.py          # ptools and code implementations of ptools for this benchmark
└── results            # results from runs of expt.py
    ├── 20260316.183513.workflow
    │   ├── config.yaml
    │   ├── results.csv
    │   └── results.jsonl
    ├└── 20260316.183519.pot
    ... ├── config.yaml
        ├── results.csv
        └── results.jsonl

## Defining the interfaces

Use `ptools.py` to define the top-level Interface for problems in this
dataset, and any ptools that will used.  Also put hand-coded tools or
workflows here

## Setting up the experiment driver

`benchmarks/sports_understanding/expt.py` is an example.  You need to

 * define how to evaluate responses, by subclassing
   evaluate.Evaluator.  The minimal implementation computes one metric
   by comparing the predicted_output and expected_output.
 * define a way to load the dataset - in the example, there's a
   load_dataset(split) function.  This needs to include an example
   id (example.name) for each example, the list of inputs that will
   be passed to the top-level Interface, and an expected output
   for those inputs.
 * define how to run an experiment, which needs to perform these steps:
   * load the shared configuration from conf/conf.yaml, and any any
   experiment-specific configuration params from the command line.
	* One necessary experiment-specific config is
      'evaluate.expt_name', which is where the results of the
      evaluation will be filed.
    * You will also probably need to bind the top-level interface to
      an implementation for each experiment.
    * The shared config params should include the result directory and
	  the cache directory.
 * load the dataset 
 * configure implementations for all the ptools, using implement_via_config
 * create an instance of your custom Evaluator and run
   evaluator.evaluate(dataset, <top-level-interface>)

## Viewing results

 * To see the most recent experiment for every expt_name, run
   * `uv run python -m secretagent.cli.results average --metric <YOUR METRIC> --metric cost results/*`
 * Other options for the `cli.results` tool are
   * `average` - show average values for each expt_name
   * `pair` - run paired tests (the p values should be < 0.05 for differences to be significant)
   * `compare` - review the config options that differ in the selected runs

All of the `cli.results` calls end with a list of `results` directories, which you can specify with 
a `results/*` glob or with a more specific file list.  They can also be modified by arguments before
the list directories:

The argument `--check config.param=value` restricts the list to ones with the specified config params.
Multiple `--check` args can be used to check multiple values.

The argument `--latest k` means to consider the `k` most recent
directories for each expt_name, instead of the single most recent one.
