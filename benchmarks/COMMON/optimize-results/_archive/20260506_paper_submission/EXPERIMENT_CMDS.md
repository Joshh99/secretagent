# EXPERIMENT_CMDS.md — copy-paste runbook for the deadline sweep

**Owner**: Joshua
**Deadline**: 2026-05-06 (paper) — experiments target 2026-05-04
**Last updated**: 2026-05-02
**Branch**: `main` → push to **upstream** (not origin; origin is the Joshh99 fork)
**Companion doc**: `EXPERIMENT_PLAN.md` (team-blocked 8-pairing matrix)

This file is the *exact* copy-paste runbook. The companion `EXPERIMENT_PLAN.md` is
the strategic context. Always run from repo root unless noted.

---

## Status at start (2026-05-02)

Five nsga2 yamls were patched (uncommitted at base commit `28047a5`) to add
`react_learned` (RQ1) and/or `wf_orch` (RQ2) wherever the learned artifacts
exist:

| Yaml | RQ1 added | RQ2 added | Coverage |
|---|---|---|---|
| `benchmarks/bbh/sports_understanding/nsga2.yaml` | ✅ | ✅ | RQ1+RQ2+RQ4 |
| `benchmarks/medcalc/nsga2.yaml` | ✅ | ✅ | RQ1+RQ2+RQ4 |
| `benchmarks/musr/nsga2_murder.yaml` | ✅ | ✅ | RQ1+RQ2+RQ4 |
| `benchmarks/finqa/nsga2.yaml` | — (no induction) | ✅ | RQ2+RQ4 |
| `benchmarks/tabmwp/nsga2.yaml` | — (no induction) | ✅ | RQ2+RQ4 |

NBA already had `react_learned` from a prior NSGA-II run (results frozen at
`benchmarks/COMMON/optimize-results/rulearena_nba/`); `wf_orch` for NBA is
blocked because rulearena's orch_learner was trained on
`compute_rulearena_answer`, not `compute_nba_answer`.

For musr/murder, the latest seeded murder orch run (`20260427.012653.orch_learner`,
3.5 MB) was copied into `benchmarks/musr/learned/murder/` so the loader's glob
disambiguates murder from object/team. Same recipe needed later for object/team.

---

## Phase 0 — Pre-flight (one-time, ~2 min)

```bash
# Confirm the patched yamls compile and the loader resolves the new bindings.
# Smoke tests already passed for sports (wf_orch) and musr/murder (wf_orch).
# Just verify your env is set:
echo "TOGETHER_API_KEY=${TOGETHER_API_KEY:+set}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:+set}"
# Both must be 'set'. ANTHROPIC_API_KEY not required for these sweeps.

# Pull latest from upstream (cache merges from teammates land here):
git fetch upstream
git status   # confirm no local conflicts before sweeps
```

---

## Phase 1 — Run the 5 patched sweeps

**Each sweep is independent**: each benchmark has its own `llm_cache/`, so you
can run them in **parallel terminals** safely (no shelve write conflicts across
benchmarks). Within a single benchmark, only one sweep at a time.

Run from **repo root** in each command:

### 1a. Sports (re-run with new RQ1+RQ2 entries)

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/bbh/sports_understanding/nsga2.yaml \
  --cwd benchmarks/bbh/sports_understanding \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=valid dataset.n=50
```
Expected wall: **~60-90 min** (cache from prior sweep helps; new methods are fresh).

### 1b. MedCalc

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/medcalc/nsga2.yaml \
  --cwd benchmarks/medcalc \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=train dataset.stratified=true dataset.shuffle_seed=42 dataset.n=50
```
Expected wall: **~150-200 min** (62 MB cache already partially warm; medcalc has
8 genes / >10K configs — bigger search).

### 1c. FinQA

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/finqa/nsga2.yaml \
  --cwd benchmarks/finqa \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=valid dataset.n=50
```
Expected wall: **~80-120 min** (17 MB cache from prior run helps).

### 1d. TabMWP

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/tabmwp/nsga2.yaml \
  --cwd benchmarks/tabmwp \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=dev1k dataset.n=50
```
Expected wall: **~90-120 min** (cold cache; first sweep on this benchmark).

### 1e. MuSR / murder

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/musr/nsga2_murder.yaml \
  --cwd benchmarks/musr \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=murder_mysteries_test dataset.n=50
```
Expected wall: **~150-200 min** (V3-react was 6 min/case in smoke test; cache will
fill fast since smaller search space).

**Total Phase 1 wall clock**:
- Sequential: **~10-12 hours**
- All 5 in parallel terminals: **~3.5 hours** (bound by the slowest, MuSR/MedCalc)

---

## Phase 2 — Snapshot + commit + push (per completed sweep)

Do this **immediately after each sweep finishes** so reruns don't overwrite a
paper-bound artifact. The snapshot is permanent under
`benchmarks/COMMON/optimize-results/<bench>/`.

### Template (replace `<BENCH>` and `<CWD>`)

```bash
# Variables to set per sweep:
BENCH=sports_understanding         # snapshot folder name
CWD=benchmarks/bbh/sports_understanding   # the benchmark's cwd
DEST=benchmarks/COMMON/optimize-results/$BENCH

# 1. Verify the sweep produced the three expected artifacts
ls -la "$CWD/results/nsga2_summary.csv" "$CWD/results/nsga2_generations.csv" "$CWD/results/nsga2.png"

# 2. Snapshot to COMMON
mkdir -p "$DEST/nsga_runs"
cp "$CWD/results/nsga2_summary.csv" "$DEST/"
cp "$CWD/results/nsga2_generations.csv" "$DEST/"
cp "$CWD/results/nsga2.png" "$DEST/"

# 3. Copy per-config dirs from THIS sweep only (filter by today's date prefix)
TODAY=$(date +%Y%m%d)
cp -r $CWD/results/$TODAY.*.nsga_*/ "$DEST/nsga_runs/" 2>/dev/null || true

# 4. Write a one-page REPRODUCE.md so the snapshot is self-citable
cat > "$DEST/REPRODUCE.md" <<EOF
# $BENCH NSGA-II sweep snapshot

Frozen on $(date -u +%Y-%m-%d) from commit $(git rev-parse --short HEAD).

## Command
\`\`\`
uv run -m secretagent.cli.optimize nsga2 \\
  --space-file $CWD/nsga2.yaml \\
  --cwd $CWD \\
  --pop-size 12 --n-gen 5 --timeout 1200 \\
  <dataset overrides — see below>
\`\`\`

## Methods searched
\`structured_baseline\`, \`workflow\`, \`pot\`, \`react\`, \`react_learned\` (RQ1),
\`wf_orch\` (RQ2). See nsga2.yaml in this directory for the exact dotlist
expansions.

## Files
- \`nsga2_summary.csv\` — one row per evaluated config
- \`nsga2_generations.csv\` — per-generation convergence stats
- \`nsga2.png\` — Pareto plot (cost vs correctness)
- \`nsga_runs/<TS>.nsga_NNN/\` — per-config rollout dirs
EOF

# 5. Commit + push (cache + snapshot together so teammates can pull both)
git add "$CWD/llm_cache/" "$DEST/" "$CWD/nsga2.yaml"
git commit -m "$BENCH: NSGA-II sweep with react_learned + wf_orch (RQ1+RQ2+RQ4)"
git push upstream main
```

### NEVER on these commits

- **No `Co-Authored-By: Claude` trailer** — user dislikes; always omit.
- **No `--no-verify`** — let pre-commit hooks run; fix issues if any fail.
- **Push to `upstream`, not `origin`** — origin is the Joshh99 fork, upstream is
  wwcohen canonical.

### Per-benchmark variables for Phase 2

| BENCH | CWD |
|---|---|
| `sports_understanding` | `benchmarks/bbh/sports_understanding` |
| `medcalc` | `benchmarks/medcalc` |
| `finqa` | `benchmarks/finqa` |
| `tabmwp` | `benchmarks/tabmwp` |
| `musr_murder` | `benchmarks/musr` |

Note: musr writes results into `benchmarks/musr/results/` (not into a
`murder/` subdir), so the snapshot dir name `musr_murder` distinguishes it
from upcoming object/team snapshots.

---

## Phase 3 — Add MuSR object + team (after Phase 1 finishes; ~30 min setup + 2x sweep time)

### 3a. Copy task-specific orch dirs into `learned/<task>/` (mirrors murder fix)

```bash
# object: 20260427.020456 = seeded answer_question_workflow run on object_placements
cp -r benchmarks/musr/results/orchestration_learner/20260427.020456.orch_learner \
      benchmarks/musr/learned/object/

# team: 20260427.024705 = seeded answer_question run on team_allocation
cp -r benchmarks/musr/results/orchestration_learner/20260427.024705.orch_learner \
      benchmarks/musr/learned/team/
```

### 3b. Create `nsga2_object.yaml` and `nsga2_team.yaml`

PLACEHOLDER — model on `nsga2_murder.yaml`. Key differences:
- **object**: replace `dataset.split=murder_mysteries_test` with
  `object_placements_test`; orch entry binds `answer_question_workflow` to
  `__learned__.answer_question_workflow_orchestrated_seed`.
- **team**: replace split with `team_allocation_test`; orch entry binds
  `answer_question` (no `_workflow`!) to
  `__learned__.answer_question_orchestrated_seed` (the team orch was trained
  on the direct `answer_question` interface, not the workflow variant).

Sweep commands once yamls exist:

```bash
uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/musr/nsga2_object.yaml \
  --cwd benchmarks/musr \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=object_placements_test dataset.n=50

uv run -m secretagent.cli.optimize nsga2 \
  --space-file benchmarks/musr/nsga2_team.yaml \
  --cwd benchmarks/musr \
  --pop-size 12 --n-gen 5 --timeout 1200 \
  dataset.split=team_allocation_test dataset.n=50
```

Expected wall: **~150-200 min each**, similar to murder.

Snapshot to `benchmarks/COMMON/optimize-results/musr_object/` and
`musr_team/` using the Phase 2 template.

---

## Phase 4 — V3.1 insurance pass (after Phase 1; ~30 min/benchmark, mostly cached)

Goal: guarantee that the 8-pairing EXPERIMENT_PLAN run will hit cache for every
`(method, V3.1)` pair, even ones NSGA-II didn't sample. One quick run per method
per benchmark with model fixed to V3.1.

Each command below is **stand-alone** — it runs a single (method, V3.1) eval at
`dataset.n=50` (matching Phase 1). Cache is shared with the corresponding
sweep, so after that benchmark's Phase 1 completes, most calls below will hit
cache. Run from the **benchmark's cwd** as noted (matches the `--cwd` used by
the optimizer in Phase 1).

Per-method dotlists are lifted verbatim from each patched `nsga2.yaml`'s
`methods:` block; only `llm.model` is pinned to V3.1 and `evaluate.expt_name`
is set to `v31_<method>`.

### 4a. Sports (`benchmarks/bbh/sports_understanding`)

```bash
cd benchmarks/bbh/sports_understanding

# structured_baseline
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=simulate \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_structured_baseline

# unstructured_baseline
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=direct \
  ptools.are_sports_in_sentence_consistent.fn=ptools.zeroshot_unstructured_workflow \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_unstructured_baseline

# workflow
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=direct \
  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_workflow

# react
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
  "ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]" \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_react

# react_learned
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
  ptools.are_sports_in_sentence_consistent.tool_module=__learned__ \
  ptools.are_sports_in_sentence_consistent.learner=ptool_editer \
  ptools.are_sports_in_sentence_consistent.tools=__all__ \
  learn.train_dir=learned \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_react_learned

# wf_orch
uv run python -m secretagent.cli.expt run \
  --interface ptools.are_sports_in_sentence_consistent \
  llm.model=together_ai/deepseek-ai/DeepSeek-V3.1 \
  ptools.are_sports_in_sentence_consistent.method=direct \
  ptools.are_sports_in_sentence_consistent.fn=__learned__.are_sports_in_sentence_consistent_orchestrated_seed \
  ptools.are_sports_in_sentence_consistent.learner=orch_learner \
  learn.train_dir=results/orchestration_learner \
  dataset.split=valid dataset.n=50 \
  evaluate.expt_name=v31_wf_orch
```

### 4b. MedCalc (`benchmarks/medcalc`)

```bash
cd benchmarks/medcalc

DS_OVR="dataset.split=train dataset.stratified=true dataset.shuffle_seed=42 dataset.n=50"
LLM_OVR="llm.model=together_ai/deepseek-ai/DeepSeek-V3.1"

# structured_baseline
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=simulate \
  $DS_OVR evaluate.expt_name=v31_structured_baseline

# workflow
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=direct \
  ptools.calculate_medical_value.fn=ptools.workflow \
  $DS_OVR evaluate.expt_name=v31_workflow

# pot
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=direct \
  ptools.calculate_medical_value.fn=ptools.pot_workflow \
  ptools.pot_medical_value.method=program_of_thought \
  "ptools.pot_medical_value.tools=[ptools.extract_clinical_values,ptools.compute_calculation]" \
  $DS_OVR evaluate.expt_name=v31_pot

# react
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=simulate_pydantic \
  "ptools.calculate_medical_value.tools=[ptools.identify_calculator,ptools.extract_clinical_values,ptools.compute_calculation]" \
  $DS_OVR evaluate.expt_name=v31_react

# react_learned
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=simulate_pydantic \
  ptools.calculate_medical_value.tool_module=__learned__ \
  ptools.calculate_medical_value.learner=ptool_inducer \
  ptools.calculate_medical_value.tools=__all__ \
  learn.train_dir=learned \
  $DS_OVR evaluate.expt_name=v31_react_learned

# wf_orch
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.calculate_medical_value.method=direct \
  ptools.calculate_medical_value.fn=__learned__.calculate_medical_value \
  ptools.calculate_medical_value.learner=orch_learner \
  learn.train_dir=results/orchestration_learner \
  $DS_OVR evaluate.expt_name=v31_wf_orch
```

### 4c. FinQA (`benchmarks/finqa`)

No `react_learned` in finqa (no induction artifacts). 6 methods: structured/unstructured baselines, workflow, pot, react, wf_orch.

```bash
cd benchmarks/finqa

LLM_OVR="llm.model=together_ai/deepseek-ai/DeepSeek-V3.1"
BASE="uv run python -m secretagent.cli.expt run --interface ptools.answer_finqa --evaluator evaluator.FinQAEvaluator"
DS_OVR="dataset.split=valid dataset.n=50"

# structured_baseline
$BASE $LLM_OVR \
  ptools.answer_finqa.method=simulate \
  $DS_OVR evaluate.expt_name=v31_structured_baseline

# unstructured_baseline
$BASE $LLM_OVR \
  ptools.answer_finqa.method=direct \
  ptools.answer_finqa.fn=ptools.unstructured_baseline_workflow \
  $DS_OVR evaluate.expt_name=v31_unstructured_baseline

# workflow
$BASE $LLM_OVR \
  ptools.answer_finqa.method=direct \
  ptools.answer_finqa.fn=ptools.answer_finqa_workflow \
  $DS_OVR evaluate.expt_name=v31_workflow

# pot
$BASE $LLM_OVR \
  ptools.answer_finqa.method=program_of_thought \
  "ptools.answer_finqa.tools=[ptools.parse_table,ptools.compute]" \
  ptools.answer_finqa.inject_args=true \
  "ptools.answer_finqa.additional_imports=[re]" \
  $DS_OVR evaluate.expt_name=v31_pot

# react
$BASE $LLM_OVR \
  ptools.answer_finqa.method=simulate_pydantic \
  "ptools.answer_finqa.tools=[ptools.call_parse_table,ptools.call_lookup_cell,ptools.call_compute,ptools.call_extract_reasoning_plan]" \
  $DS_OVR evaluate.expt_name=v31_react

# wf_orch
$BASE $LLM_OVR \
  ptools.answer_finqa.method=direct \
  ptools.answer_finqa.fn=__learned__.answer_finqa_orchestrated_seed \
  ptools.answer_finqa.learner=orch_learner \
  learn.train_dir=results/orchestration_learner \
  $DS_OVR evaluate.expt_name=v31_wf_orch
```

### 4d. TabMWP (`benchmarks/tabmwp`)

No `react_learned` in tabmwp (no induction artifacts). 5 methods.

```bash
cd benchmarks/tabmwp

LLM_OVR="llm.model=together_ai/deepseek-ai/DeepSeek-V3.1"
DS_OVR="dataset.split=dev1k dataset.n=50"

# structured_baseline
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.tabmwp_solve.method=simulate \
  $DS_OVR evaluate.expt_name=v31_structured_baseline

# workflow
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.tabmwp_solve.method=direct \
  ptools.tabmwp_solve.fn=ptools.tabmwp_workflow \
  $DS_OVR evaluate.expt_name=v31_workflow

# pot
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  evaluate.entry_point=tabmwp_solve_pot \
  ptools.tabmwp_solve_pot.method=program_of_thought \
  "ptools.tabmwp_solve_pot.additional_imports=[pandas]" \
  $DS_OVR evaluate.expt_name=v31_pot

# react
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  evaluate.entry_point=tabmwp_react \
  ptools.tabmwp_react.method=simulate_pydantic \
  "ptools.tabmwp_react.tools=[ptools.get_table_schema,ptools.lookup_value,ptools.query_column,ptools.query_table,ptools.extract_answer]" \
  $DS_OVR evaluate.expt_name=v31_react

# wf_orch
uv run python expt.py run --config-file conf/conf.yaml \
  $LLM_OVR \
  ptools.tabmwp_solve.method=direct \
  ptools.tabmwp_solve.fn=__learned__.tabmwp_solve_orchestrated_seed \
  ptools.tabmwp_solve.learner=orch_learner \
  learn.train_dir=results/orchestration_learner \
  $DS_OVR evaluate.expt_name=v31_wf_orch
```

### 4e. MuSR / Murder (`benchmarks/musr`)

7 methods (includes `zs_cot`).

```bash
cd benchmarks/musr

LLM_OVR="llm.model=together_ai/deepseek-ai/DeepSeek-V3.1"
DS_OVR="dataset.split=murder_mysteries_test dataset.n=50"

# structured_baseline
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  ptools.answer_question.method=simulate \
  $DS_OVR evaluate.expt_name=v31_structured_baseline

# zs_cot
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  ptools.answer_question.method=prompt_llm \
  ptools.answer_question.prompt_template_file=prompt_templates/zero_shot_cot.txt \
  $DS_OVR evaluate.expt_name=v31_zs_cot

# workflow
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  evaluate.entry_point=answer_question_workflow \
  ptools.answer_question_workflow.method=direct \
  $DS_OVR evaluate.expt_name=v31_workflow

# pot
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  ptools.answer_question.method=program_of_thought \
  ptools.answer_question.tools=__all__ \
  llm.max_tokens=16384 \
  $DS_OVR evaluate.expt_name=v31_pot

# react
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  ptools.answer_question.method=direct \
  ptools.answer_question.fn=ptools_common.react_answer_impl \
  ptools.react_solve.method=simulate_pydantic \
  "ptools.react_solve.tools=[ptools_common.search,ptools_common.lookup,ptools_common.finish]" \
  $DS_OVR evaluate.expt_name=v31_react

# react_learned
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  ptools.answer_question.method=direct \
  ptools.answer_question.fn=ptools_common.react_answer_impl \
  ptools.react_solve.method=simulate_pydantic \
  ptools.react_solve.tool_module=__learned__ \
  ptools.react_solve.learner=ptool_inducer \
  ptools.react_solve.tools=__all__ \
  learn.train_dir=learned/murder \
  $DS_OVR evaluate.expt_name=v31_react_learned

# wf_orch
uv run python expt.py run --config-file conf/murder.yaml \
  $LLM_OVR \
  evaluate.entry_point=answer_question_workflow \
  ptools.answer_question_workflow.method=direct \
  ptools.answer_question_workflow.fn=__learned__.answer_question_workflow_orchestrated_seed \
  ptools.answer_question_workflow.learner=orch_learner \
  learn.train_dir=learned/murder \
  $DS_OVR evaluate.expt_name=v31_wf_orch
```

### Notes for execution

- Each pass is **~5-10 min/method** if cache warm (post-Phase-1), **~30
  min/method** if cold.
- Total Phase 4: **~30 min/benchmark x 5 = ~2.5 hours** if mostly cached.
- These commands write into each benchmark's `results/` dir under
  `evaluate.expt_name=v31_<method>` — they don't snapshot to COMMON. Cache is
  the load-bearing artifact; commit `llm_cache/` after the pass completes.
- After Phase 4 commits, the EXPERIMENT_PLAN 8-pairing run will inherit all
  V3.1 cache hits.

---

## Phase 5 — Paper drafting (run in parallel with Phase 1, after first 2-3 sweeps land)

Two distinct sections to write into `paper_draft.txt`:

1. **§4.5 "Can modular agentic systems be effectively optimized?"** — your
   contribution. Lead with cross-model heterogeneous-LLM Pareto frontiers per
   benchmark (the 5 sweeps). Frame: *"Across 5 diverse benchmarks, NSGA-II
   discovers configurations that mix learned and engineered components and
   assign different LLMs per stage; the cost-vs-correctness Pareto …"*

2. **Subsection of §4.5 (or appendix): EXPERIMENT_PLAN cross-pairing study** —
   placeholder until Aditya/Jerry unblock. Will use V3.1-only, the 8 cells.

Build the per-benchmark Pareto figure block:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.32\textwidth]{benchmarks/COMMON/optimize-results/sports_understanding/nsga2.png}
\includegraphics[width=0.32\textwidth]{benchmarks/COMMON/optimize-results/medcalc/nsga2.png}
\includegraphics[width=0.32\textwidth]{benchmarks/COMMON/optimize-results/finqa/nsga2.png}
\includegraphics[width=0.32\textwidth]{benchmarks/COMMON/optimize-results/tabmwp/nsga2.png}
\includegraphics[width=0.32\textwidth]{benchmarks/COMMON/optimize-results/musr_murder/nsga2.png}
\caption{Per-benchmark NSGA-II Pareto frontiers …}
\label{fig:pareto-grid}
\end{figure}
```

Cross-benchmark summary (after all sweeps done):
```bash
uv run -m secretagent.cli.optimize cross-summary \
  benchmarks/COMMON/optimize-results/sports_understanding/nsga2_summary.csv \
  benchmarks/COMMON/optimize-results/medcalc/nsga2_summary.csv \
  benchmarks/COMMON/optimize-results/finqa/nsga2_summary.csv \
  benchmarks/COMMON/optimize-results/tabmwp/nsga2_summary.csv \
  benchmarks/COMMON/optimize-results/musr_murder/nsga2_summary.csv \
  --output benchmarks/COMMON/optimize-results/cross_summary.md
```

---

## Total time projection

| Phase | Sequential | Parallel (5 terminals) |
|---|---|---|
| Phase 0 (preflight) | 5 min | 5 min |
| Phase 1 (5 sweeps) | 10-12 hr | 3.5 hr |
| Phase 2 (snapshot+push, 5x) | 30-50 min | spread across Phase 1 ends |
| Phase 3 (object+team) | 5-7 hr | 3.5 hr |
| Phase 4 (V3.1 insurance, 5x) | 2-3 hr | mostly cached, ~1 hr |
| Phase 5 (paper drafting) | concurrent | concurrent |

**Best-case parallel total: ~9 hours of wall clock** (one solid day) for the
unblocked work, with paper drafting happening throughout.

**Sequential safe estimate: ~22 hours** spread across 2-3 days.

Cache pushes after each sweep are critical — teammates rebasing on `upstream`
get the cache automatically and amortize cost across the team.
