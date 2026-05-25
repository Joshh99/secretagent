# tabmwp NSGA-II sweep snapshot

Frozen on 2026-05-02 from commit 6b0a48f.

## Command

```
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/tabmwp/nsga2.yaml \
  --cwd benchmarks/tabmwp \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  <dataset overrides — see ../_archive/20260506_paper_submission/EXPERIMENT_CMDS.md Phase 1 for the exact split>
```

## Methods searched

`structured_baseline`, `workflow`, `pot`, `react`, `react_learned`
(RQ1; if applicable), `wf_orch` (RQ2). See `benchmarks/tabmwp/nsga2.yaml` for the exact
dotlist expansions per method.

## Files

- `nsga2_summary.csv` — one row per evaluated config (dev1k split)
- `nsga2_generations.csv` — per-generation convergence stats
- `nsga2.png` — Pareto plot (cost vs correctness)
- `nsga_runs/<TS>.nsga_NNN/` — per-config rollout dirs (42 total)
- `test_pass_summary.csv` — paired (valid, test) numbers per Pareto config

## Test pass

Optimizer scored on `dev1k`; the unbiased generalization numbers come from
re-running each Pareto-optimal config once on `test1k`:

```bash
uv run python benchmarks/COMMON/optimize-results/test_pass.py tabmwp
```

See `benchmarks/COMMON/optimize-results/test_pass_README.md` for details.

```bash
# pull the test numbers via the standard CLI
uv run -m secretagent.cli.results average \
  --check evaluate.expt_name=test_pass_* \
  benchmarks/tabmwp/results
```
