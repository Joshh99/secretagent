# TabMWP Benchmark Report

## Dataset

TabMWP: 38K grade-school math problems over small tables (avg 6 rows, 2 cols). Free-text (75%) and multi-choice (25%), grades 1-8.

Example: *"What is the mean of the numbers?"* + table of names/coin counts -> answer: `84`.

Source: Lu et al., ICLR 2023. License: CC BY-NC-SA 4.0.

## Experiments

All experiments use `together_ai/deepseek-ai/DeepSeek-V3.1`, caching enabled, full rollout recordings saved.


| Config               | LLM calls | Table access   | Ptool context                | PTP stubs      | Description                                       |
| -------------------- | --------- | -------------- | ---------------------------- | -------------- | ------------------------------------------------- |
| `zeroshot`           | 1         | in prompt      | n/a                          | No             | Single simulate call, no decomposition            |
| `guided_ptp`         | 1         | in prompt      | n/a                          | **Yes**        | Single call with Python stub trace as scaffolding |
| `workflow_broad`     | 2         | in prompt      | full (question+table)        | Yes (simulate) | `extract_and_compute` -> `format_answer`          |
| `workflow_rich`      | 4         | in prompt      | full (question+table)        | Yes (simulate) | 4 ptools, each receives question + table          |
| `workflow_incontext` | 4         | in prompt      | **isolated** (own args only) | Yes (simulate) | 4 ptools, each sees only its own arguments        |
| `workflow_tools`     | 4         | via tool calls | **isolated**                 | Yes (simulate) | Same as above, table fetched by tool interfaces   |
| `orchestrated`       | 4         | in prompt      | isolated (auto-wired)        | Yes (simulate) | LLM auto-composes pipeline from available ptools  |
| `react`              | variable  | via tool calls | shared (agent state)         | Yes (simulate) | ReAct agent decides which tools to call           |
| `pot`                | 1         | as dict        | n/a                          | No             | LLM generates Python code executed in sandbox     |


**Naming note:** `workflow_incontext` refers to the table being *in the prompt context* (vs `workflow_tools` which fetches it via tool calls). It does NOT mean the ptools have full context — in fact, each ptool in this config sees only its own arguments (isolated).

## Results (n=50, test1k split, seed=42)


| Config                | Accuracy | +/- SE | Cost/ex | LLM calls | Ptool context  | Key insight                                               |
| --------------------- | -------- | ------ | ------- | --------- | -------------- | --------------------------------------------------------- |
| guided (v1, informal) | 98%      | 2.0%   | $0.0005 | 1         | n/a            | Informal step hints                                       |
| **guided_ptp**        | **96%**  | 2.8%   | $0.0015 | 1         | n/a            | **PTP stubs as scaffolding**                              |
| react                 | 90%      | 4.3%   | $0.0067 | ~5-10     | shared (agent) | Agent flexibility, 17x cost                               |
| workflow_broad        | 90%      | 4.3%   | $0.0007 | 2         | full           | Fewer steps = less error propagation                      |
| zeroshot              | 88%      | 4.6%   | $0.0004 | 1         | n/a            | Strong baseline, cheapest                                 |
| workflow_rich         | 82%      | 5.5%   | $0.0015 | 4         | full           | Context helps, 4 steps still compound                     |
| orchestrated          | 50%      | 7.1%   | $0.0012 | 4         | isolated       | Auto-composed same flawed structure                       |
| workflow_incontext    | 46%      | 7.1%   | $0.0010 | 4         | **isolated**   | Each ptool sees only its own args                         |
| workflow_tools        | 42%      | 7.1%   | $0.0010 | 4         | **isolated**   | Tool access ~= in-context on small tables                 |
| pot                   | 10%      | 4.3%   | $0.0010 | 1         | n/a            | Sandbox blocks float()/int(); issue #7                    |
| pot (sandbox fixed)   | 60%      | 7.0%   | $0.0010 | 1         | n/a            | With builtins injected; branch `fix/pot-sandbox-builtins` |


**Total cost for all n=50 experiments: ~$2**

**Note on guided v1 vs v2 (guided_ptp):** v1 used informal English steps ("Step 1 — Identify the operation"). v2 uses actual Python function stubs with type signatures and docstrings, presented as a program trace to simulate — the PTP approach.

## Results (n=1000, test1k split)

*To be run.*

## Analysis

**PTP-style trace simulation works.** `guided_ptp` (96%) strongly outperforms zeroshot (88%) by presenting Python stubs as reasoning scaffolding in a single LLM call. This validates PTP-style decomposition for tabular reasoning, not just the BBH tasks in the original paper. 4x cheaper than react at higher accuracy.

**Context isolation is the primary failure mode for modular workflows.** The original 4-step workflow (46%) fails because each ptool sees only its own arguments. Adding full context to every step (`workflow_rich`, 82%) nearly doubles accuracy. The difference: `compute_answer("difference", ["0.78", "0.54"])` vs `compute_answer_rich(question, table, "difference", ["0.78", "0.54"])`.

**Fewer steps also helps.** `workflow_broad` (90%) > `workflow_rich` (82%). Even with full context, 4 handoff points still allow error compounding.

**The orchestrator inherits the isolation problem.** It composed nearly the same 4-step pipeline as the hand-coded workflow (50% vs 46%).

**PoT is broken by sandbox restrictions, but fixable.** 45/50 examples crash because `float()` and `int()` are blocked by smolagents' executor. A 7-line fix injecting safe builtins brings PoT from 10% to 60% (branch `fix/pot-sandbox-builtins`). This affects other benchmarks too (natural_plan, medcalc). Filed as issue #7.

## Checks for Later

- Error analysis by `ques_type`, `ans_type`, `grade`
- Per-ptool accuracy from rollout recordings
- n=1000 run for tighter confidence intervals
- Phase 7: grid search over models/methods per ptool
- PoT re-run after sandbox fix (issue #7)

## Key Decisions

1. `**table_id` in interface signature.** All configs receive `(question, table, table_id, choices)`. Keeps interface uniform; zeroshot ignores `table_id`.
2. **PoT receives `table_for_pd` (dict).** Sandbox restricts `float()` and `pandas`. Standard practice — original TabMWP paper provides this format for programmatic approaches.
3. **React uses `extract_answer` post-processing.** Follows MUSR pattern (`raw_answer` -> `extract_index`).
4. **Evaluator strips `$`, `%` and uses tolerance-based numeric matching.** Integers within 0.5, decimals within 0.01.
5. **Model:** `together_ai/deepseek-ai/DeepSeek-V3.1` for all experiments.

