# RuleArena Benchmark Report

## Dataset

RuleArena (Sun et al., 2024): rule-based reasoning across three domains, each
with complexity levels 0/1/2 controlling the number of interacting rules.

| Domain  | Task | Answer type | N (valid split) |
|---------|------|-------------|-----------------|
| Airline | Compute total baggage + ticket cost from AA fee rules | integer (USD) |  60 |
| Tax     | Compute federal income tax from filled IRS forms       | float (USD)   |  60 |
| NBA     | Detect CBA salary cap violations in proposed trades    | binary (0/1)  |  42 |

Source: vendored from `external/RuleArena/`. Deterministic Python calculators
provide ground truth for airline and tax; NBA ground truth is author-labeled.

## Experiment Levels

| Level | Method | LLM calls | Description |
|-------|--------|-----------|-------------|
| L0    | Oracle | 0 | Ground-truth params fed to Python calculators. Accuracy ceiling. |
| L0F   | CoT    | 1 | Single LLM call with chain-of-thought prompt. No decomposition. |
| L1    | PTool  | 1 | LLM extracts structured params via prompt_llm template, Python computes answer. |
| L3    | Agent  | variable | Autonomous agent with extraction and calculator tool interfaces. |

All experiments use `together_ai/deepseek-ai/DeepSeek-V3`, valid split only.

## Results (valid split, N=60/42)

### Airline

| Level | Accuracy | Tol. Acc | Cost/ex | Latency | N | Status |
|-------|----------|----------|---------|---------|---|--------|
| L0    | 100.0%   | 100.0%   | $0      | 0s      | 60 | full |
| L0F   |  41.7%   |  40.0%   | $0.019  | 46.1s   | 60 | full |
| L1    | **98.3%**|  96.7%   | $0.015  | 13.9s   | 60 | full |
| L3    |  76.7%   |  73.3%   | $0.048  | 19.3s   | 60 | full |

### Tax

| Level | Accuracy | Tol. Acc | Cost/ex | Latency | N | Status |
|-------|----------|----------|---------|---------|---|--------|
| L0    | 100.0%   | 100.0%   | $0      | 0s      | 60 | full |
| L0F   |  56.7%   |  48.3%   | $0.014  | 44.0s   | 60 | full |
| L1    |**100.0%**| 100.0%   | $0.015  | 27.3s   | 60 | full |
| L3    |  61.7%   |  61.7%   | $0.053  | 69.0s   | 60 | full |

### NBA

| Level | Accuracy | F1 macro | Cost/ex | Latency | N | Status |
|-------|----------|----------|---------|---------|---|--------|
| L0F   |  66.7%   |  0.57    | $0.030  | 31.7s   | 42 | full |
| L1    | **83.3%**|  0.45    | $0.028  |  8.5s   | 42 | full |
| L3    |  83.3%   |  0.55    | $0.066  | 19.1s   | 42 | full |

No L0 for NBA (binary classification, no calculator). Class distribution is
36 True / 6 False (86% positive), so accuracy overstates performance.
F1 macro is the better metric here; L1's 0.45 matches ptp's 0.44 on the
same task.

## Replication Targets

Reference numbers from ptp-behavior-distillation (separate codebase, same
model DeepSeek-V3, **mixed train+valid+test splits, N=300/216**):

| Level | Airline (N=300) | Tax (N=300) | NBA (N=216) |
|-------|----------------:|------------:|------------:|
| L0    |          100.0% |      100.0% |          -- |
| L0F   |           48.3% |       35.3% |       64.4% |
| L1    |           77.0% |       99.7% |       80.1% |
| L3 (react) |      63.3% |       78.3% |       82.4% |

Source: `ptp-behavior-distillation/benchmark_results/rulearena/rq2_summary.csv`,
runs dated 2026-02-27 to 2026-03-05.

**Comparison caveats:**
- ptp used all splits (train+valid+test); our results are valid-split only
- N differs (300 vs 60 for airline/tax, 216 vs 42 for NBA)
- Direct accuracy comparison is directional, not exact

## Analysis

**L1 extraction replicates ptp on tax.** secretagent L1 achieves 100% on
valid-split tax (N=60), matching ptp's 99.7% on mixed-split (N=300). This
required migrating from the generic `simulate` factory to task-specific
`prompt_llm` templates with direct framing. The simulate factory scored 43.3%
on the same data -- a 56.7pp gap caused entirely by prompt scaffolding
differences. See `findings_prompt_scaffolding.md` for the full ablation.

**L1 airline exceeds ptp reference, but on a smaller/different split.**
98.3% vs ptp's 77.0%. Possible explanations: valid-split may be easier than
mixed, small N variance, or our region-normalization helper fixes edge cases
ptp didn't handle. Needs full mixed-split run to confirm.

**L1 NBA accuracy matches ptp, but F1 reveals class imbalance.** 83.3%
accuracy vs ptp's 80.1%, but F1 macro is 0.45 (ptp: 0.44). The valid split
is 86% positive (36T/6F), so a model that always predicts True would score
86% accuracy. L1 predicts True on 40/42 cases (TP=34, FP=6, FN=2), meaning
it barely distinguishes compliant from violating operations.

**L3 agent underperforms L1 on airline and tax.** L3 airline (76.7%) and L3
tax (61.7%) are well below their L1 counterparts, while costing 3-4x more
per example. L3 NBA matches L1 (both 83.3%). This pattern -- agents
underperforming structured extraction on deterministic tasks -- is consistent
with ptp's findings.

**L0F CoT is the weakest LLM-based method across all domains.** This is
expected: a single unstructured prompt with no decomposition or tool use.

**Cost efficiency.** L1 is the cheapest LLM method ($0.015-0.028/ex) while
achieving the highest accuracy. L3 costs 2-4x more with lower accuracy on
airline and tax. L0F is comparable in cost to L1 but much less accurate.

## Known Issues

- **Airline L1 failures:** 1/60 wrong (airline_2_91: predicted 4759, expected
  5324). 1 additional tolerance failure.
- **NBA false-positive bias:** 6 of 7 L1 NBA errors predicted violation when
  answer was compliant. Suggests the prompt may prime toward finding violations.
- **L3 tax: agent overhead without accuracy gain.** L3 tax (61.7%) performs
  at roughly the same level as L0F CoT (56.7%) but at 4x the cost ($0.053
  vs $0.014/ex). On complex extraction tasks, the multi-turn agent loop adds
  latency and cost without improving over a single well-structured prompt.

## Implementation Notes

- L1 uses `prompt_llm` factory with domain-specific templates
  (`prompt_templates/{airline,tax,nba}_extract.txt`) and `answer_pattern=None`
  for raw string output, parsed by `_parse_json_result()`.
- L3 uses `simulate_pydantic` (pydantic-ai JSON function-calling agent).
  secretagent currently has one L3 backend; ptp had two: `l3_react`
  (text-based ReAct loop) and `l3_pydantic` (pydantic-ai function-calling).
  The L3 numbers in the Replication Targets table are from ptp's `l3_react`,
  which is a mechanistically different agent architecture. A text-based
  `simulate_react` factory for secretagent is under discussion.
- L0F uses `prompt_llm` with CoT templates (`prompt_templates/{airline,tax,nba}_cot.txt`).
- All levels share the same Python calculators (`calculators/airline.py`, `calculators/tax.py`).

## Checks for Later

- Full mixed-split runs (N=300/216) for direct ptp comparison
- L1 airline error analysis on full split to validate or discount the 98.3%
- NBA prompt rebalancing to reduce false-positive bias
- L3 with prompt_llm extraction (hybrid: structured extraction + agent reasoning)
- Multi-model comparison (Qwen, GPT-oss, Haiku)
- Cost-accuracy Pareto analysis across levels and models
