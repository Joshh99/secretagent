#
# Run experiments - all on the validation data
#

# zero-shot structured queries - i.e. the old ptool LLM / PTP call
structured_baseline: 
	uv run expt.py run evaluate.expt_name=structured_baseline \
	  ptools.are_sports_in_sentence_consistent.method=simulate

# zero-shot unstructured queries with a vanilla prompt
# technically the vanilla prompt returns a string, which is then coerced to a bool
# in the second stage of a very simple workflow

unstructured_baseline: 
	uv run expt.py run evaluate.expt_name=unstructured_baseline \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.zeroshot_unstructured_workflow

# the hand-coded workflow
workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow

# program of thoughts
pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.are_sports_in_sentence_consistent.method=program_of_thought \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

# ReAct using simulate_pydantic (which runs ReAct if there are tools)
react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

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

#
# Learning is a pipeline, documented below  
#

# 1) 'record' detailed traces of an implementation strategy, put them in recordings
#    Recordings are just like evaluations but keep some extra information around.
#

learn_pipeline: record_workflow learn_consistent_sports evaluate_learned_consistent_sports avg

RECORD_OPTS = dataset.split=train evaluate.record_details=true evaluate.result_dir=recordings

record_workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow \
	  $(RECORD_OPTS)	

# 2) Then run the learner - right now that's a rote learner - for some
# interface.  A learner loads the recordings and saves stuff to a
# training directory, train-dir, in this case filed under
# consistent_sports__rote

learn_consistent_sports:
	uv run python -m secretagent.cli.learn rote --interface consistent_sports --train-dir training recordings/*

# 3) Now we can configure the learned implementation 

evaluate_learned_consistent_sports:
	uv run expt.py run evaluate.expt_name=workflow_with_learning \
	  learn.train_dir=training \
	  ptools.consistent_sports.method=learned \
	  ptools.consistent_sports.learner=rote \
	  ptools.consistent_sports.backoff=True \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow
