# Path to the RuleArena repo clone — only needed to run 'make prepare'
RULEARENA_DIR ?= ../../../RuleArena

#
# One-time data preparation (run before any experiment)
#

prepare:
	uv run python data/prepare.py $(RULEARENA_DIR)

#
# Oracle: Python calculators with ground-truth parameters, no LLM
#

oracle_airline:
	uv run expt.py run evaluate.expt_name=oracle_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

oracle_nba:
	uv run expt.py run evaluate.expt_name=oracle_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

oracle_tax:
	uv run expt.py run evaluate.expt_name=oracle_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0_oracle_workflow

#
# Unstructured baseline: single LLM call per instance, chain-of-thought, no structure
#

unstructured_baseline_airline:
	uv run expt.py run evaluate.expt_name=unstructured_baseline_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

unstructured_baseline_nba:
	uv run expt.py run evaluate.expt_name=unstructured_baseline_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

unstructured_baseline_tax:
	uv run expt.py run evaluate.expt_name=unstructured_baseline_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow

#
# Structured baseline (workflow): PTool extraction + Python calculation
#

structured_baseline_airline:
	uv run expt.py run evaluate.expt_name=structured_baseline_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

structured_baseline_nba:
	uv run expt.py run evaluate.expt_name=structured_baseline_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

structured_baseline_tax:
	uv run expt.py run evaluate.expt_name=structured_baseline_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

#
# ReAct: autonomous agent (simulate_pydantic with tools)
#

react_airline:
	uv run expt.py run evaluate.expt_name=react_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_airline_params,ptools.compute_airline_calculator]"

react_nba:
	uv run expt.py run evaluate.expt_name=react_nba dataset.domain=nba \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_nba_params]"

react_tax:
	uv run expt.py run evaluate.expt_name=react_tax dataset.domain=tax \
	  ptools.compute_rulearena_answer.method=simulate_pydantic \
	  "ptools.compute_rulearena_answer.tools=[ptools.extract_tax_params,ptools.compute_tax_calculator]"

#
# PoT: Program of Thought (LLM generates Python code, no tools)
#

pot_airline:
	uv run expt.py run evaluate.expt_name=pot_airline dataset.domain=airline \
	  ptools.pot_airline.method=program_of_thought \
	  ptools.pot_airline.inject_args=true \
	  "ptools.pot_airline.tools=[]" \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.pot_workflow

pot_nba:
	uv run expt.py run evaluate.expt_name=pot_nba dataset.domain=nba \
	  ptools.pot_nba.method=program_of_thought \
	  ptools.pot_nba.inject_args=true \
	  "ptools.pot_nba.tools=[]" \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.pot_workflow

pot_tax:
	uv run expt.py run evaluate.expt_name=pot_tax dataset.domain=tax \
	  ptools.pot_tax.method=program_of_thought \
	  ptools.pot_tax.inject_args=true \
	  "ptools.pot_tax.tools=[]" \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.pot_workflow

#
# Legacy aliases (old L-level names → new standardized names)
#

l0_airline: oracle_airline
l0_nba: oracle_nba
l0_tax: oracle_tax
l0f_airline: unstructured_baseline_airline
l0f_nba: unstructured_baseline_nba
l0f_tax: unstructured_baseline_tax
l1_airline: structured_baseline_airline
l1_nba: structured_baseline_nba
l1_tax: structured_baseline_tax
l3_airline: react_airline
l3_nba: react_nba
l3_tax: react_tax

#
# Learning pipeline (airline domain, structured_baseline extract workflow)
#

RECORD_OPTS = dataset.split=train evaluate.record_details=true evaluate.result_dir=recordings

learn_pipeline: record_structured_baseline_airline learn_extract_airline evaluate_learned_structured_baseline_airline avg

record_structured_baseline_airline:
	uv run expt.py run evaluate.expt_name=structured_baseline_airline dataset.domain=airline \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow \
	  $(RECORD_OPTS)

learn_extract_airline:
	uv run python -m secretagent.cli.learn rote --interface extract_airline_params \
	  --learned-dir training recordings/*

evaluate_learned_structured_baseline_airline:
	uv run expt.py run evaluate.expt_name=structured_baseline_airline_learned dataset.domain=airline \
	  learn.train_dir=training \
	  ptools.extract_airline_params.method=learned_code \
	  ptools.extract_airline_params.learner=rote \
	  ptools.extract_airline_params.backoff=True \
	  ptools.compute_rulearena_answer.method=direct \
	  ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow

# Legacy aliases for learning pipeline
record_l1_airline: record_structured_baseline_airline
evaluate_learned_l1_airline: evaluate_learned_structured_baseline_airline

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric correct_tolerance --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

costs:
	uv run python -m secretagent.cli.costs llm_cache

#
# Pareto sweep (exhaustive or NSGA-II: workflows x models per domain)
#

N_PARETO ?= 5
TIMEOUT_PARETO ?= 600

pareto_airline:
	uv run python run_pareto.py --domain airline --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_nba:
	uv run python run_pareto.py --domain nba --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_tax:
	uv run python run_pareto.py --domain tax --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_all: pareto_airline pareto_nba pareto_tax
