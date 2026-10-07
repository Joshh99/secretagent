# finqa NSGA-II sweep snapshot

Frozen on 2026-05-03 from commit a56e7ed1.

## Command

```
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/finqa/nsga2.yaml \
  --cwd benchmarks/finqa \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  <dataset overrides — see ../_archive/20260506_paper_submission/EXPERIMENT_CMDS.md Phase 1 for the exact split>
```

## Methods searched

`structured_baseline`, `workflow`, `pot`, `react`, `react_learned`
(RQ1; if applicable), `wf_orch` (RQ2). See `benchmarks/finqa/nsga2.yaml` for the exact
dotlist expansions per method.

## Files

- `nsga2_summary.csv` — one row per evaluated config (valid split)
- `nsga2_generations.csv` — per-generation convergence stats
- `nsga2.png` — Pareto plot (cost vs correctness)
- `nsga_runs/<TS>.nsga_NNN/` — per-config rollout dirs (0 total)

## Test pass

**Not yet run for finqa.** Earlier notes here said the FinQA test set was
private. That was wrong. The official repository ships a labeled public test
split at `dataset/test.json`: 1,147 rows, every one carrying `qa.exe_ans`,
with ids disjoint from dev. `private_test.json` is a different file and is the
one held back for the leaderboard.

So every number in `nsga2_summary.csv` is a **validation** number, selected on
validation. Label it that way wherever it appears. A public test pass is
possible and is pending: run the validation-selected candidates once on the
300-case export, changing nothing else.

Raw provenance:

    raw/dev.json    883 rows  sha256 a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51
    raw/test.json  1147 rows  sha256 831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc
