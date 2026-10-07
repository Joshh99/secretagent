# How to add a learner

A learner is a class that subclasses `learn.base.Learner` and has three methods: `fit`, `save_implementation` and `report`. Put each learner in its own Python file if you can. `RoteLearner` in `baselines.py` is a simple example to copy from.

- `fit` does the actual learning and is usually the slow part. It works out and saves whatever the learned implementation needs.
- `save_implementation` saves what is needed to use the result: a small YAML example showing how to configure the learned implementation, and every file that implementation needs.
- `report` returns a short, readable summary that helps you judge how well the learning went.

## Keeping track of where the training data came from

Use `collect_distillation_data` to collect training data from recorded runs. It saves the data in the learner's own output folder, together with a record of which runs it came from. If you need more, for example keeping only the runs that got the right answer, add it to the code in `base` so every learner can use it.

## Running generated code

If a learner produces code that might be unsafe to run, run it inside `LocalPythonExecutor`.
