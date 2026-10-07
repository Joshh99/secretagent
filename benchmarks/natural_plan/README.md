# NaturalPlan

Three planning tasks: calendar scheduling, meeting planning and trip planning. Each task is run with five methods, so there are 15 experiments in all.

## Set up

```bash
cd secretagent && uv sync
export TOGETHER_AI_API_KEY="your-key"
cd benchmarks/natural_plan
```

The configs use DeepSeek-V3.1 served by Together AI, which is why the key is needed. You can also put API keys in `secretagent/.env`; the Makefile loads that file for you.

## Run the experiments

```bash
# One experiment
make cal_workflow

# All 15 experiments (3 tasks, 5 methods each), 50 questions each by default
make run_all_15

# A quick test with 5 questions each
uv run python scripts/run_all_15.py -n 5

# The same, with five worked examples in each prompt
uv run python scripts/run_all_15.py -n 5 --prompt-mode 5shot

# Save the prompts as well
make run_all_15_trace
```

## Look at the results

```bash
make report          # writes report.md
make plot            # writes plot_calendar.png, plot_meeting.png, plot_trip.png
make export          # copies the results to benchmarks/COMMON/results/natural_plan/
```

## Tests

```bash
make test
```

This runs 15 tests, one for each task and method, on 2 questions each. It works the same way as `test_sports_understanding.py`.

## Settings

| Setting | Default | What it does |
|-------|---------|-------------|
| `llm.model` | `together_ai/deepseek-ai/DeepSeek-V3.1` | The model, as named by litellm |
| `dataset.split` | none | `calendar`, `meeting` or `trip` |
| `dataset.n` | 4 | How many questions to run |
| `dataset.prompt_mode` | 5shot | `5shot` or `0shot` |
| `dataset.stratified` | false | Use stratified sampling |
| `dataset.sample_n` | 50 | How many questions to sample when `dataset.stratified` is true |
| `ptools.{entry}.method` | none | How a pseudo-tool is carried out: `simulate`, `direct`, `prompt_llm`, `program_of_thought` or `simulate_pydantic` |
