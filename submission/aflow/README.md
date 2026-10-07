# AFlow comparison: data and scripts

In the paper we compare our methods with AFlow, a tool that builds LLM workflows by itself. A workflow here is a fixed series of steps, some of them LLM calls and some plain Python, that solves one kind of task. This folder lets you rebuild every AFlow number in the paper on your own computer in a few seconds. You do not need an API key, and no model gets called.

## Run it

You need Python 3.9 or newer. Nothing else has to be installed.

```
python build_aflow_numbers.py --check
```

The last line should say `323 of 323 numbers match the paper.` The script also writes two files to `outputs/`. `aflow_report.md` is the one to read. `aflow_numbers.json` holds every number at full precision.

## What we did

AFlow starts from the workflow we built by hand for a task. A planner model, Gemini 3.1 Pro, then keeps writing changed versions of that workflow. Each version is run with DeepSeek-V3.1 on 50 validation questions, and its score is saved. One search is 54 rounds of this.

At the end of a search we take the version with the best validation score. If two versions tie, the cheaper one wins, and if they still tie, the earlier one. We saved this choice before looking at any test result. Then we ran the chosen version once on the test questions.

We did this for five tasks, three times each, with a different seed each time. A seed is the number that controls the random choices inside a search, so three seeds give three separate searches. That makes 15 searches. Two network outages broke six other attempts, so we threw those out and reran them from the start. That is why there are 21 attempts in total.

## What the script does

1. For each search, it picks the winning round again from the saved validation scores and checks that it matches the choice we saved at the time.
2. It works out the test accuracy and cost of each chosen workflow. These numbers make up the AFlow row of the main tables and the per-seed table in the appendix. For Murder, both accuracy and cost use only the 50 held out questions, like every other row in the Murder column.
3. For the Murder task, it scores the other methods on the 50 test questions that AFlow never saw during its search.
4. It compares AFlow with each other method on Murder, one question at a time, with a 95% bootstrap interval. To get that interval, the script resamples the questions 10,000 times and sees how much the difference moves. It shows how much of a gap could come from chance.
5. It rebuilds the table of all 21 attempts and the cost totals.

With `--check`, the script compares each number with `expected.json`, which holds the numbers as printed in the paper. We typed that file in from the paper's tables by hand, so it is independent of the script.

## Files

| File | What is in it |
|---|---|
| `build_aflow_numbers.py` | The script you run. |
| `expected.json` | The numbers printed in the paper, used by `--check`. |
| `extract_from_logs.py` | How we made `data/` from the full run logs. You cannot run it, because the logs are not included. It is here so you can see where each data file comes from. |
| `data/attempts.json` | One entry per search attempt: the validation score and cost of every round, which rounds failed and why, what the attempt cost, the planner's token counts and the run settings. |
| `data/selected_tests/*.csv` | Test results of the 15 chosen workflows. One row per question: its ID, 1 if the answer was right and 0 if not, and AFlow's own cost estimate for that question. |
| `data/selected_test_usage.json` | Token counts and billed cost of those 15 test runs. |
| `data/medcalc_billed_by_question.csv` | For the three MedCalc test runs: what each question's calls cost, and whether the question counts under Formulas or Rules. |
| `data/murder_baselines.csv` | The other methods' Murder results, one row per question: ID, right or wrong, and cost. |
| `data/murder_baselines_sources.json` | Which saved result file each of those methods comes from, with its checksum. A checksum is a fingerprint of a file's contents, so you can tell if the file has changed. |
| `data/murder_bills_by_question.csv` | For the three Murder test runs: what each question cost and how many tokens it used. |
| `murder_bills_by_question.py` | How we worked out those Murder costs from the call logs. Like `extract_from_logs.py`, you cannot run it without the logs. |
| `data/murder_heldout_ids.json` | The 50 Murder test questions AFlow never saw during its search. |
| `data/selection_records/` | The round each search picked, saved before its test result was read. |

The data holds only IDs, scores, costs and counts. It has no question text and no model answers.

## Terms used above

Validation and test questions: a search uses validation questions to compare versions. Test questions are kept aside and used once at the end, so the final score is fair.

Held out: for Murder, half of the 100 test questions were also AFlow's validation questions. The other 50 are held out, meaning AFlow never saw them. The paper's Murder numbers use only those 50. The two CodeDist rows are the one exception, because they were tested on a different set of questions; the paper marks them with a dagger.

Executor and optimizer: the executor, DeepSeek-V3.1, answers the questions. The optimizer, Gemini 3.1 Pro, writes the new workflow versions.

Damaged round: a round where the network dropped some answers. A damaged round can never be picked.

Parse failure: a round where the planner wrote code that Python could not read. It still uses up the round.

Original and amended rule: the rule for deciding whether a search counts changed once, before the later searches started. The paper's appendix explains both rules, and the attempts table shows each attempt under both.

## How the costs are counted

Cost in the paper: what we actually paid to test each chosen workflow, per 100 test questions. OpenRouter passed every call to AtlasCloud, the company that ran DeepSeek-V3.1 for us, and each saved call records what it was charged. The script adds those charges up. For MedCalc, every call was matched to the question it answered, so the Formulas and Rules columns each get the bill for their own questions (`data/medcalc_billed_by_question.csv`).

Cost at Together AI prices: the other rows of the paper's cost table were priced from their token counts at Together AI's prices. To compare like with like, the script also prices AFlow's token counts that way (USD 0.60 per million input tokens and USD 1.70 per million output tokens). AFlow's bills came to 34% to 52% of that price. So on the same footing as most other rows, AFlow would cost about twice what the table shows.

Optimizer cost: an estimate. It is the planner's token count times Gemini 3.1 Pro list prices (USD 2 per million input tokens and USD 12 per million output tokens). AFlow's own cost tracker misses Gemini's thinking tokens, so we counted tokens from the saved responses instead.

MedCalc: the paper splits MedCalc into two columns, Formulas and Rules. Each billed call was matched to the question it answered, so each column gets the bill for its own questions.

Murder: the bill is recorded per call, not per question. `murder_bills_by_question.py` ties each call to its question by finding the question's text inside the call. About one short call per question cannot be tied that way: AFlow's voting step, which only sees the candidate answers and picks the most common one. AFlow saved its own cost estimate for every question, so the cost those voting calls left unaccounted for is known for each question, and their bill is split the same way. Voting calls are about 1% of each bill, so this split barely changes anything.

Other methods on Murder: their cost is what they spent on the 50 held out questions, divided by 50. One method (Orch-WfSeed) timed out on two of those questions. Those two add no cost, just as they count as wrong answers.

## Limits

This folder covers the AFlow side and the Murder column. The other numbers in the main tables come from the repository's main table scripts.

The billed costs come from adding up about 3 GB of saved model calls. Those calls are not included here, so the billed totals are given as data and not recomputed.

Running AFlow again would not give exactly the same answers, because the model provider does not accept a random seed.

MedCalc's test set has one pair of questions that are identical in every way (`test.0217` and `test.0218`). They get their IDs in the order they appear. They are the same question with the same answer, so this cannot change any score.
