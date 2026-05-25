# RQ1+RQ2 x Optimize: Combination Experiments

**Owner**: Joshua
**Status**: planning
**Last updated**: 2026-05-01

## What this is

This is the "Combined with Optimize" column of Prof's RQ1/RQ2 matrix. It
evaluates how *learned* workflows (Aditya's `orch`, Jerry's `codedistill`)
compose with *learned* ptools (Jerry's code-distilled, Lex's
react-induced) — running each pairing through the NSGA-II optimizer
(`secretagent.cli.optimize`) to produce per-benchmark Pareto frontiers
and a cross-benchmark summary.

## The matrix (per benchmark)

```
                            PT
            +------+---------------+---------+
            | eng  | code-distill  | learned |
   +--------+------+---------------+---------+
   |   eng  |  ok  |      ok       |    -    |   <- excluded
WF |  orch  |  ok  |      ok       |    ok   |
   | codeD  |  ok  |      ok       |    ok   |
   +--------+------+---------------+---------+
```

Formal:

> **WF[eng, orch, codedistill] x PT[eng, code-distilled]
>   +  WF[orch, codedistill] x PT[learned]**

- Term 1: 3 x 2 = 6 pairings
- Term 2: 2 x 1 = 2 pairings
- **Total: 8 (workflow, ptools) pairings per benchmark**

The `+` is set union, not arithmetic. The skipped cell
`(eng workflow, learned ptools)` is excluded by design: Lex's induced
ptools assume a learned-style planner above them; pairing them with a
hand-engineered workflow is not structurally meaningful.

Each pairing is one full evaluation: a single workflow + a single ptools
set are bound, the benchmark runs end-to-end. There is no runtime
combinatorics — the 8 is just how many subprocess evals to launch per
benchmark.

## Scope

| Benchmark | Sub-tasks | Pairings | Eval count |
|---|---|---|---|
| MUSR | 3 (murder, team, object) | 8 | 24 |
| NatPlan | 2 | 8 | 16 |
| RuleArena/NBA | 1 | 8 | 8 |
| MedCalc | 2 | 8 | 16 |
| **Total** | | | **64 evals** |

**Model**: `together_ai/deepseek-ai/DeepSeek-V3.1` only — fixed across
every cell. Per Paul: every collaborator's prior runs (Aditya, Jerry,
Lex) used V3.1. The LLM cache key is `(prompt, model)`, so:

- Workflow LLM calls overlap with Aditya's prior orch runs -> cache hit
- Ptools LLM calls overlap with Jerry's / Lex's prior runs -> cache hit

Switching models would force every call to re-pay. Sticking to V3.1
collapses logical eval count into a small fraction of new API spend.

## Cache merging convention

Cross-benchmark outputs from each component team land under
`benchmarks/COMMON/`:

```
benchmarks/COMMON/
    codedistill-workflow-results/   # Jerry's workflow learning
    codedistill-ptools-results/     # Jerry's ptools learning
    optimize-results/               # this experiment (and merged caches)
    results/                        # per-benchmark canonical results
```

Workflow before kickoff:

1. Pull `codedistill-workflow-results/<bench>/llm_cache/` and
   `codedistill-ptools-results/<bench>/llm_cache/`.
2. Pull Aditya's orch caches (location TBD — confirm with Aditya).
3. Pull Lex's react-induced ptools caches (location TBD — confirm with Lex).
4. Merge into the per-benchmark `llm_cache/` directory before launching
   the sweep. Cache files are hash-keyed; duplicates are no-ops.

## Setup in the optimizer

Each pairing maps to one named entry in the benchmark's `nsga2.yaml`:

```yaml
methods:
  wf_eng__pt_eng:                   [overrides for eng wf + eng pt]
  wf_eng__pt_codedistill:           [overrides for eng wf + code-distilled pt]
  wf_orch__pt_eng:                  [...]
  wf_orch__pt_codedistill:          [...]
  wf_orch__pt_learned:              [...]
  wf_codedistill__pt_eng:           [...]
  wf_codedistill__pt_codedistill:   [...]
  wf_codedistill__pt_learned:       [...]
```

Each value is a list of dotlist overrides that bind the workflow
function and per-sub-interface implementations. The encoder treats them
as a single compound categorical gene
(`src/secretagent/optimize/encoder.py:60-91`).

### Code-generated combinations

Hand-writing 8 entries x 4 benchmarks = 32 entries is bug-prone. Use a
helper that emits the methods dict from workflow templates + ptools
templates + the exclusion rule:

```python
# benchmarks/COMMON/optimize-results/combo_methods.py
def build_combo_methods(
    workflows: dict[str, list[str]],
    ptools: dict[str, list[str]],
) -> dict[str, list[str]]:
    """WF[eng,orch,codedistill] x PT[eng,code-distilled]
       + WF[orch,codedistill] x PT[learned]."""
    out = {}
    for wf in ["eng", "orch", "codedistill"]:
        for pt in ["eng", "codedistill"]:
            out[f"wf_{wf}__pt_{pt}"] = workflows[wf] + ptools[pt]
    for wf in ["orch", "codedistill"]:
        out[f"wf_{wf}__pt_learned"] = workflows[wf] + ptools["learned"]
    return out  # -> 8 entries
```

Per-benchmark YAML declares only the templates; the helper expands them.
When Aditya renames `orch_workflow -> orch_planner`, only the template
changes — not 32 hand-written method entries.

### Per-benchmark inputs needed

For each benchmark, collect six dotlist override snippets:

| Source | Owner | Status |
|---|---|---|
| Eng workflow | (existing in repo) | ready |
| Orch workflow | Aditya | confirm snippet per benchmark |
| Codedistill workflow | Jerry | confirm snippet per benchmark |
| Eng ptools | (existing in repo) | ready |
| Codedistill ptools | Jerry | confirm snippet per benchmark |
| Learned (react-induced) ptools | Lex | confirm snippet per benchmark; airline `nsga2_airline.yaml` has a working `react_learned` example to copy from |

## Running

```bash
# Per benchmark (8 configs total -> auto-falls back to exhaustive,
# since space_size <= 20 threshold; see pareto.py:449)
uv run -m secretagent.cli.optimize nsga2 \
    --space-file benchmarks/musr/nsga2_combo.yaml \
    --cwd benchmarks/musr \
    --pop-size 8 --n-gen 1 \
    llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
    dataset.split=<split> dataset.n=<N>
```

`--pop-size 8 --n-gen 1` is nominal — with only 8 configs the auto-
exhaustive path runs every pairing once and ignores the evolutionary
parameters.

```bash
# Cross-benchmark report after all 4 done
uv run -m secretagent.cli.optimize cross-summary \
    benchmarks/musr/results/nsga2_summary.csv \
    benchmarks/natural_plan/results/nsga2_summary.csv \
    benchmarks/rulearena/nba/results/nsga2_summary.csv \
    benchmarks/medcalc/results/nsga2_summary.csv \
    --output benchmarks/COMMON/optimize-results/combo_summary.md
```

### Outputs

- Per benchmark: `results/nsga2_summary.csv`, `results/nsga2.png`
- Cross-benchmark: `optimize-results/combo_summary.md` with
  per-benchmark hypervolume + method-frequency tables

## Research questions answered by this column

1. **Do learned workflows + learned ptools compound or interfere?**
   Compare `(orch, code-distilled)` and `(codedistill, learned)` against
   single-axis baselines `(orch, eng)` and `(eng, code-distilled)`. If
   the diagonal beats both off-diagonals, learning compounds.
2. **Is one source of "learning" universally better?**
   Method-frequency on the Pareto frontier across the 4 benchmarks
   shows whether `orch` or `codedistill` dominates as a workflow source,
   and whether `code-distilled` or `learned` dominates as a ptools
   source.
3. **Where on the accuracy/cost frontier do learned components win?**
   Engineered components are usually cheaper but sometimes weaker;
   learned components add cost and sometimes accuracy. The Pareto plot
   makes the tradeoff explicit per benchmark.

## Open blockers

- [ ] Dotlist snippet for orch workflow per benchmark (Aditya)
- [ ] Dotlist snippet for codedistill workflow per benchmark (Jerry)
- [ ] Dotlist snippet for codedistill ptools per benchmark (Jerry)
- [ ] Dotlist snippet for Lex's induced ptools per benchmark (Lex)
- [ ] Cache directory locations for Aditya's and Lex's outputs (for rsync-merge)
- [ ] Confirm `dataset.n` and split per benchmark — should match what
      Aditya/Jerry/Lex used so cache overlap is maximal
- [ ] Confirm sub-task counts: MUSR (3), NatPlan (2), NBA (1), MedCalc (2)?

## Notes

- The 8-pairing space is small enough that the optimizer's NSGA-II logic
  is unused; it auto-falls back to exhaustive enumeration. The optimizer
  is still the right tool because it gives us the same outputs format
  (Pareto CSV, hypervolume, cross-summary report) used by the rest of
  the team.
- If the model axis is later expanded beyond V3.1 (e.g., add gpt-oss-120b
  for sensitivity analysis), the same yaml will scale: 8 pairings x 2
  models = 16 configs, still under the exhaustive threshold of 20.
- Cache merging is the load-bearing optimization here. Without it,
  re-running the workflow LLM calls from scratch in 4 benchmarks is the
  bulk of the cost. With it, this column should complete in hours, not
  days.
