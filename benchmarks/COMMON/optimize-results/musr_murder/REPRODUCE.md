# musr_murder NSGA-II sweep snapshot

Frozen on 2026-05-04 from commit dccdcac8.

## Command

```
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/musr/nsga2.yaml \
  --cwd benchmarks/musr \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  <dataset overrides — see ../_archive/20260506_paper_submission/EXPERIMENT_CMDS.md Phase 1 for the exact split>
```

## Methods searched

`structured_baseline`, `workflow`, `pot`, `react`, `react_learned`
(RQ1; if applicable), `wf_orch` (RQ2). See `benchmarks/musr/nsga2.yaml` for the exact
dotlist expansions per method.

## Files

- `nsga2_summary.csv` — one row per evaluated config
- `nsga2_generations.csv` — per-generation convergence stats
- `nsga2.png` — Pareto plot (cost vs correctness)
- `nsga_runs/<TS>.nsga_NNN/` — per-config rollout dirs (53 total)

## Caveat — split bug

**This sweep scored on `murder_mysteries_test`, not `_val`.** The Pareto
frontier was therefore selected by peeking at test data, so its configs
should not be reported as test numbers in the paper.

To produce honest numbers, re-run NSGA-II on `murder_mysteries_val`, then run
the test pass on the new frontier:

```bash
# 1. Re-run optimizer on the valid split (~3 hr wall, mostly cache-warm)
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/musr/nsga2_murder.yaml \
  --cwd benchmarks/musr \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=murder_mysteries_val dataset.n=50

# 2. Snapshot the new sweep into a fresh dir, then add musr_murder to
#    test_pass.py BENCHES with test_split="murder_mysteries_test", and run:
uv run python benchmarks/COMMON/optimize-results/test_pass.py musr_murder
```
