#
# Run experiments - all on the validation data
#

# zero-shot structured queries - i.e. the old ptool LLM / PTP call
EVALUATOR = --evaluator ptools.DateUnderstandingEvaluator

structured_baseline:
	uv run python -m secretagent.cli.expt run --interface ptools.answer_date_question $(EVALUATOR) evaluate.expt_name=structured_baseline \
	  ptools.answer_date_question.method=simulate

# orchestrated: LLM auto-generates a workflow composing all sub-tools
orchestrated:
	uv run python -m secretagent.cli.expt run --interface ptools.answer_date_question_orchestrated $(EVALUATOR) \
	  evaluate.expt_name=orchestrated \
	  echo.orchestrate=true \
	  orchestrate.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
	  orchestrate.max_retries=3 \
	  ptools.answer_date_question.method=simulate \
	  ptools.answer_date_question_orchestrated.method=orchestrate

# quick test of orchestrated on a single example
quick-test-orchestrated:
	uv run python -m secretagent.cli.expt quick-test --interface ptools.answer_date_question_orchestrated \
	  echo.orchestrate=true \
	  orchestrate.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
	  orchestrate.max_retries=3 \
	  ptools.answer_date_question.method=simulate \
	  ptools.answer_date_question_orchestrated.method=orchestrate

# zero-shot unstructured queries with a vanilla prompt
unstructured_baseline:
	uv run python -m secretagent.cli.expt run --interface ptools.answer_date_question $(EVALUATOR) evaluate.expt_name=unstructured_baseline \
	  ptools.answer_date_question.method=direct \
	  ptools.answer_date_question.fn=ptools.zeroshot_unstructured_workflow

#
# Summarize results of past experiments
#

CHECK ?=

list:
	uv run python -m secretagent.cli.results list $(CHECK) results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost $(CHECK) results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost $(CHECK) results/*

compare:
	uv run python -m secretagent.cli.results compare-configs $(CHECK) results/*

#
# track costs so far
#

costs:
	uv run python -m secretagent.cli.costs llm_cache
