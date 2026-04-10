-include ../../.env
export

#
# Run experiments — 5 conditions x 3 tasks
#
# Stratified 50 samples (AgentProject-compatible): add dataset.stratified=true dataset.sample_n=50 dataset.n=50
# Sample 50 more (seed 43): add dataset.sample_seed=43
#

# ── Calendar ──

cal_zeroshot_structured:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_zs_struct \
	  ptools.calendar_scheduling.method=simulate

cal_zeroshot_unstructured:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_zs_unstruct \
	  ptools.calendar_scheduling.method=prompt_llm \
	  ptools.calendar_scheduling.prompt_template_file=prompt_templates/zeroshot.txt

cal_workflow:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_workflow \
	  ptools.calendar_scheduling.method=direct \
	  ptools.calendar_scheduling.fn=ptools_calendar.calendar_workflow

cal_pot:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_pot \
	  ptools.calendar_scheduling.method=program_of_thought \
	  ptools.calendar_scheduling.tools=[ptools_calendar.parse_schedules,ptools_calendar.find_available_slots,ptools_calendar.select_and_format]

cal_react:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_react \
	  ptools.calendar_scheduling.method=simulate_pydantic \
	  ptools.calendar_scheduling.tools=[ptools_calendar.parse_schedules,ptools_calendar.find_available_slots,ptools_calendar.select_and_format]

# ── Meeting ──

meet_zeroshot_structured:
	uv run python expt.py run --config-file conf/meeting.yaml \
	  evaluate.expt_name=meet_zs_struct \
	  ptools.meeting_planning.method=simulate

meet_zeroshot_unstructured:
	uv run python expt.py run --config-file conf/meeting.yaml \
	  evaluate.expt_name=meet_zs_unstruct \
	  ptools.meeting_planning.method=prompt_llm \
	  ptools.meeting_planning.prompt_template_file=prompt_templates/zeroshot.txt

meet_workflow:
	uv run python expt.py run --config-file conf/meeting.yaml \
	  evaluate.expt_name=meet_workflow \
	  ptools.meeting_planning.method=direct \
	  ptools.meeting_planning.fn=ptools_meeting.meeting_workflow

meet_pot:
	uv run python expt.py run --config-file conf/meeting.yaml \
	  evaluate.expt_name=meet_pot \
	  ptools.meeting_planning.method=program_of_thought \
	  ptools.meeting_planning.tools=[ptools_meeting.parse_meeting_info,ptools_meeting.plan_visit_order,ptools_meeting.build_meeting_plan]

meet_react:
	uv run python expt.py run --config-file conf/meeting.yaml \
	  evaluate.expt_name=meet_react \
	  ptools.meeting_planning.method=simulate_pydantic \
	  ptools.meeting_planning.tools=[ptools_meeting.parse_meeting_info,ptools_meeting.plan_visit_order,ptools_meeting.build_meeting_plan]

# ── Trip ──

trip_zeroshot_structured:
	uv run python expt.py run --config-file conf/trip.yaml \
	  evaluate.expt_name=trip_zs_struct \
	  ptools.trip_planning.method=simulate

trip_zeroshot_unstructured:
	uv run python expt.py run --config-file conf/trip.yaml \
	  evaluate.expt_name=trip_zs_unstruct \
	  ptools.trip_planning.method=prompt_llm \
	  ptools.trip_planning.prompt_template_file=prompt_templates/zeroshot.txt

trip_workflow:
	uv run python expt.py run --config-file conf/trip.yaml \
	  evaluate.expt_name=trip_workflow \
	  ptools.trip_planning.method=direct \
	  ptools.trip_planning.fn=ptools_trip.trip_workflow

trip_pot:
	uv run python expt.py run --config-file conf/trip.yaml \
	  evaluate.expt_name=trip_pot \
	  ptools.trip_planning.method=program_of_thought \
	  ptools.trip_planning.tools=[ptools_trip.parse_trip_constraints,ptools_trip.find_valid_route,ptools_trip.build_trip_plan]

trip_react:
	uv run python expt.py run --config-file conf/trip.yaml \
	  evaluate.expt_name=trip_react \
	  ptools.trip_planning.method=simulate_pydantic \
	  ptools.trip_planning.tools=[ptools_trip.parse_trip_constraints,ptools_trip.find_valid_route,ptools_trip.build_trip_plan]

#
# Stratified 50 samples (same IDs as AgentProject seed=42)
#

cal_50_workflow:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_workflow \
	  ptools.calendar_scheduling.method=direct \
	  ptools.calendar_scheduling.fn=ptools_calendar.calendar_workflow \
	  dataset.stratified=true dataset.sample_n=50 dataset.n=50

cal_50_more_workflow:
	uv run python expt.py run --config-file conf/calendar.yaml \
	  evaluate.expt_name=cal_workflow_more \
	  ptools.calendar_scheduling.method=direct \
	  ptools.calendar_scheduling.fn=ptools_calendar.calendar_workflow \
	  dataset.stratified=true dataset.sample_n=50 dataset.sample_seed=43 dataset.n=50

# Quick view: table by task (accuracy, cost, time, reproduce cmd) -> report.md
report:
	uv run python scripts/report.py -o report.md

# Run all 15 (3 tasks × 5 levels) with n=5 each
run_all_15:
	uv run python scripts/run_all_15.py

# Same + save prompt_trace.jsonl per run for inspection
run_all_15_trace:
	uv run python scripts/run_all_15.py --trace

#
# View results of past experiments
#

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*

export:
	uv run python -m secretagent.cli.results export results/*

plot: plot_cal plot_meet plot_trip

plot_cal:
	uv run python -m secretagent.cli.results plot --metric cost- --metric correct --output plot_calendar.png results/run_*/*cal_*

plot_meet:
	uv run python -m secretagent.cli.results plot --metric cost- --metric correct --output plot_meeting.png results/run_*/*meet_*

plot_trip:
	uv run python -m secretagent.cli.results plot --metric cost- --metric correct --output plot_trip.png results/run_*/*trip_*

test:
	uv run python -m pytest ../../benchmarks/tests/test_natural_plan.py -v

#
# Pareto sweep (exhaustive: methods x models per domain)
#

N_PARETO ?= 5
TIMEOUT_PARETO ?= 600

pareto_calendar:
	uv run python run_pareto.py --domain calendar --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_meeting:
	uv run python run_pareto.py --domain meeting --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_trip:
	uv run python run_pareto.py --domain trip --dataset-n $(N_PARETO) --timeout $(TIMEOUT_PARETO)

pareto_all: pareto_calendar pareto_meeting pareto_trip
