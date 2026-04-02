# MODO Knowledge Reference: Multi-Objective Discrete Optimization for secretagent

## 1. The Core Problem

We have a system (secretagent) that can run benchmark evaluations with different configurations. Each evaluation returns two numbers: **accuracy** and **cost**. These objectives conflict: higher accuracy typically requires more expensive LLM calls, more agent steps, or bigger models.

The goal: find the set of configurations that represent the best possible tradeoffs between accuracy and cost. This set is the **Pareto frontier**.

### 1.1 Key Definitions

**Pareto dominance**: Config A dominates config B if A is better or equal on *every* objective. For us: A has >= accuracy AND <= cost compared to B.

**Pareto frontier (non-dominated set)**: All configs that are not dominated by any other config in the evaluated set. These are the "rational choices" where picking one means you've consciously decided which objective matters more.

**Dominated config**: A config where some other config is strictly better on all objectives. There is no reason to ever use a dominated config.

**Example from actual results (DeepSeek-V3, airline domain):**

| Config | Accuracy | Cost/q | Dominated by | On frontier? |
|--------|----------|--------|--------------|-------------|
| L1 PTool | 77.0% | $0.0007 | nothing | Yes |
| L0F CoT | 48.3% | $0.0054 | L1 (better acc AND cost) | No |
| L3 ReAct | 63.3% | $0.0027 | L1 (better acc AND cost) | No |

Airline frontier = {L1}. L1 dominates everything on this domain.

**NBA domain (different story):**

| Config | F1 | Cost/q | Dominated by | On frontier? |
|--------|-----|--------|--------------|-------------|
| L1 PTool | 0.44 | $0.0009 | nothing (cheapest) | Yes |
| L0F CoT | 0.50 | $0.0071 | nothing (best F1) | Yes |
| L3 ReAct | 0.48 | $0.0045 | L0F (better F1 AND cost) | No |

NBA frontier = {L1, L0F}. Neither dominates the other. This is the "boundary condition" finding.

### 1.2 Why the Frontier Matters for the Paper

The Pareto frontier IS the deliverable. It answers: "Given a cost budget, what's the best architecture?" or "Given an accuracy target, what's the cheapest way to get there?" The optimizer is the tool that produces this artifact systematically across benchmarks.


## 2. Two Distinct Optimization Problems

### Problem A: Config Search (hyperparameter optimization)

Fixed set of pipeline architectures. Search over their settings (model, n_demos, temperature). Find the Pareto-optimal settings.

- Search space: discrete grid of categorical choices
- Tools: NSGA-II via DEAP or jMetalPy
- Scope: implementable in days
- Risk: low

### Problem B: Architecture Search (workflow discovery)

Don't just choose from pre-designed pipelines. Discover new pipeline topologies automatically. The optimizer invents workflows nobody designed by hand.

- Search space: all possible DAGs over available interfaces
- Tools: MCTS (AFLOW), RL over graphs (GPTSwarm)
- Scope: weeks of work, research contribution in itself
- Risk: high (might not beat hand-designed pipelines)

**Decision**: Implement Problem A first. Use Problem B framing in the paper (cite GPTSwarm/AFLOW). Problem B connects to RQ3 (meta-router) and the `orchestrate` factory.


## 3. Relevant Papers

### 3.1 GPTSwarm (ICML 2024 Oral)

**What it does**: Represents LLM agent pipelines as computational graphs. Nodes = operations (LLM calls, tool calls, Python functions). Edges = information flow. Optimizes via:
- Node optimization: prompt refinement within each node
- Edge optimization: RL (REINFORCE) over graph connectivity — add/remove/reweight edges

**Key idea**: Agents are graphs. Graphs are differentiable (edge weights are probabilities). You can optimize graph structure with gradient-free RL.

**Relevance to us**: The graph representation maps cleanly onto secretagent's `@interface` system. Each interface is a node, workflow functions define edges. GPTSwarm's edge optimization is analogous to our "which method binds to which interface" question. However, GPTSwarm optimizes *agent-to-agent collaboration topology*, which is a different problem than optimizing *config knobs on fixed pipelines*.

**Limitation noted by AFLOW authors**: Graph structure can't represent conditional logic, loops, or retry patterns.

**Citation**: Zhuge et al., "Language Agents as Optimizable Graphs," ICML 2024.

### 3.2 AFLOW (ICLR 2025)

**What it does**: Represents agentic workflows as code (not just graphs). Each workflow is a program where LLM invocations are parameterized nodes (model, prompt, temperature, output format). Searches over this code space using Monte Carlo Tree Search (MCTS).

**Key innovations**:
- **Operators**: Reusable building blocks (Ensemble, Review & Revise, Self-Refine, Multi-Agent Debate). Composable primitives that reduce the search space.
- **MCTS search**: Tree-structured exploration. Each tree node represents a workflow variant. Selection via UCB (exploration/exploitation balance), expansion via LLM-driven code modification, evaluation via benchmark execution, backpropagation of results.
- **Code representation**: More expressive than graphs — supports conditionals, loops, error handling.

**Results**: 5.7% average improvement over SOTA baselines across 6 benchmarks. Smaller models + AFLOW-discovered workflows can outperform GPT-4o at 4.55% of inference cost.

**Relevance to us**: AFLOW's node abstraction (model + prompt + temperature + output format) maps directly onto our `@interface` + `implement_via()` system. The `orchestrate` factory is essentially a rudimentary version of AFLOW (LLM composes pipeline from available ptools). If we ever extend to Problem B, AFLOW is the template.

**Citation**: Zhang et al., "AFlow: Automating Agentic Workflow Generation," ICLR 2025.

### 3.3 Connection to secretagent

| AFLOW/GPTSwarm concept | secretagent equivalent |
|---|---|
| Node | `@interface` decorated function |
| Node parameters (model, prompt) | `ptools.<name>.method`, `llm.model` |
| Edge | Workflow function calling interfaces in sequence |
| Operator (Ensemble, Self-Refine) | `self_consistency` factory, potential new factories |
| Code-based workflow | `orchestrate` factory output |
| Workflow search | What the Pipeline Optimizer does |

Lauhitya's suggestion: represent our pipelines as graphs/code and let the optimizer discover new topologies, not just pick from {L0F, L1, L3, PoT}. This is valid but is Problem B — implement after Problem A.


## 4. NSGA-II Mechanics

### 4.1 Algorithm Overview

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the standard algorithm for multi-objective optimization with 2-3 objectives. Two key innovations over basic genetic algorithms:

**Non-dominated sorting**: Sorts the population into fronts by dominance rank.
- Front 1: non-dominated individuals (current Pareto frontier)
- Front 2: non-dominated after removing Front 1
- Front 3: non-dominated after removing Fronts 1 and 2
- ...

**Crowding distance**: When two individuals are on the same front and one must be eliminated, keep the one that's more isolated (farther from its neighbors along the frontier). This maintains diversity and prevents the frontier from collapsing to one region.

### 4.2 The Loop

```
1. Initialize population of N random configs
2. Evaluate all N (run benchmarks → accuracy + cost)
3. For each generation (repeat G times):
   a. Create N children via selection + crossover + mutation
   b. Evaluate all N children
   c. Combine parents + children → 2N individuals
   d. Non-dominated sort the 2N into fronts
   e. Fill next generation of size N:
      - Add all of Front 1 (if it fits)
      - Add all of Front 2 (if it fits)
      - When a front doesn't fit entirely:
        sort by crowding distance, take the most spread-out
   f. Back to N individuals. Repeat from (a).
4. Return Front 1 of final population = Pareto frontier
```

### 4.3 Operators for Categorical Variables

Standard NSGA-II uses SBX crossover and polynomial mutation, which assume continuous variables. For our categorical config space:

**Uniform crossover**: For each gene position, randomly pick from parent A or parent B with 50/50 probability.
```
Parent A: [L1,     DeepSeek-V3, 0]
Parent B: [PoT,    Haiku,       5]
Child:    [L1,     Haiku,       5]  (took gene 0 from A, genes 1-2 from B)
```

**Random reset mutation**: Pick a random gene position, replace with a uniformly random valid value for that dimension.
```
Before: [L1,  Haiku,  5]
After:  [L1,  Haiku,  3]  (gene 2 mutated: n_demos 5 → 3)
```

### 4.4 Considerations for Expensive Evaluations

Standard NSGA-II assumes cheap evaluations. Ours cost real money (API calls). Adaptations:

- **Small populations**: 10-20 individuals, not 100
- **Few generations**: 5-10, not 100
- **Evaluation cache**: If a config was already evaluated, return cached result
- **Optional**: Surrogate-assisted optimization (build a cheap model of the objective function, use it to pre-screen candidates). Not needed initially.


## 5. Package Comparison

### 5.1 The Three Candidates

| Dimension | DEAP | jMetalPy | Airbus d-o |
|---|---|---|---|
| **Core design** | General EA toolkit, you build everything | MOO framework with batteries included | Discrete optimization (OR problems) |
| **Representation** | Anything (list, array, custom) | Float/Integer/Binary/Permutation | Problem-specific (TSP, knapsack) |
| **Categorical support** | Native (you define your own) | Encode as integers, fight abstractions | Not designed for this |
| **NSGA-II** | `tools.selNSGA2` | Full algorithm class | Basic |
| **Other MOO algos** | SPEA2 | NSGA-III, MOEA/D, SMPSO, SMS-EMOA, IBEA | NSGA only |
| **Quality indicators** | External (pymoo) | Built-in (hypervolume, IGD, IGD+, epsilon) | No |
| **Statistical testing** | No | Yes (Wilcoxon, Friedman, Bayesian) | No |
| **Visualization** | No (use matplotlib) | Yes (real-time + static Pareto plots) | Basic |
| **Parallelism** | multiprocessing, SCOOP | Spark, Dask | No |
| **GitHub stars** | ~5,800 | ~400 | ~52 |
| **License** | LGPL-3.0 | MIT | Apache 2.0 |
| **Python** | 2 & 3 | 3.6+ | 3.10+ |
| **Maturity** | Very mature (used by TPOT) | Academic, actively developed | Niche |

### 5.2 Recommendation: DEAP

**Why DEAP wins for our problem:**

1. **Categorical variables are first-class**. You define your own individual type and operators. No encoding hacks.
2. **Minimal abstraction overhead**. The toolbox pattern is transparent — you see exactly what's happening.
3. **Battle-tested for pipeline optimization**. TPOT (AutoML tool) uses DEAP for exactly this kind of problem: searching over ML pipeline configurations.
4. **Sufficient MOO support**. `selNSGA2` and `selSPEA2` cover what we need. Quality indicators can come from pymoo if needed.
5. **Easy to extend**. Custom evaluation functions, caching, early stopping — all straightforward.

**What DEAP lacks (and how to fill gaps):**
- Hypervolume indicator → `pymoo.indicators.hv` or 30 lines of code
- Pareto front plotting → matplotlib (we'd write this anyway)
- Statistical testing → scipy (already a dependency)


## 6. secretagent Infrastructure (from Claude Code report)

### 6.1 What Already Exists

**Config system** (`config.py`): OmegaConf-based, untyped, fully dynamic. Key API:
- `configure(yaml_file, dotlist)` — merge config sources
- `configuration(**kw)` — context manager for temporary config scope
- `get(key)` / `require(key)` — dot-notation access

**Interface system** (`core.py`):
- `@interface` — declares a function stub (type signature + docstring, no body)
- `implement_via(method, **kwargs)` — binds an implementation via a factory
- `implement_via_config(module, config)` — bulk-bind from YAML config

**9 registered factories** (methods):
| Name | What it does |
|---|---|
| `direct` | Execute a Python function |
| `simulate` | LLM predicts function output from signature + docstring |
| `prompt_llm` | Custom prompt template |
| `program_of_thought` | LLM generates Python code, executes it |
| `simulate_pydantic` | pydantic-ai ReAct agent with tools |
| `ptp` | Program Trace Prompting (simulate with trace examples) |
| `self_consistency` | Majority vote over N samples of an inner method |
| `orchestrate` | LLM composes a pipeline from available ptools |
| `learned` | Load a function from training data |

**Evaluator** (`evaluate.py`): Runs interface over dataset, returns per-case results with `correct`, `cost`, `input_tokens`, `output_tokens`, `latency`. Saves to timestamped directory as CSV + JSONL + config snapshot.

**Grid search** (`optimize.py`):
- `SearchSpace` — Cartesian product over config dimensions
- `GridSearchRunner` — sequential subprocess-per-config evaluation
- `run_single()` returns: `{accuracy, total_cost, cost_per_q, total_latency, latency_per_q, ...}`

**Result analysis** (`cli/results.py`): `average` (mean +/- stderr), `pair` (paired t-tests), `compare-configs`, `list`, `validate`.

### 6.2 What's Missing (Gaps)

1. **No Pareto analysis** — no frontier computation, no dominance sorting, no multi-objective comparison anywhere in the repo
2. **No cross-run result store** — only per-run directories; comparison requires loading multiple CSVs
3. **No programmatic evaluate-and-return** — `run_single()` uses subprocess; no in-process evaluation loop
4. **Interface bindings are global mutable state** — `implement_via()` mutates global Interface objects; not safe for parallel in-process evaluation
5. **No incumbent tracking or convergence detection**
6. **No plotting code** — no matplotlib, seaborn, or visualization anywhere

### 6.3 Integration Seam

**Best approach: subprocess-level integration (current `run_single()`).**

The optimizer sits on top of the existing infrastructure. It generates config dotlists, passes them to `GridSearchRunner.run_single()`, reads back `{accuracy, cost_per_q}`. No changes to benchmark code, evaluator, config system, or CLI tools.

```
DEAP NSGA-II
    |
    | individual = [method_idx, model_idx, ndemos_idx]
    v
Config Encoder/Decoder
    |
    | dotlist = ["ptools.compute_answer.method=simulate", "llm.model=deepseek-v3", ...]
    v
GridSearchRunner.run_single(config_idx, dotlist)
    |
    | subprocess: loads config, binds interfaces, runs evaluator, saves results
    v
Returns: {accuracy: 0.77, cost_per_q: 0.0007, ...}
    |
    v
DEAP fitness: (0.77, -0.0007)  # maximize accuracy, minimize cost
```

### 6.4 PtoolCatalog Hooks

`catalog.py` already has placeholder fields for optimizer-driven selection:
```python
class PtoolInfo:
    avg_cost: float | None = None
    avg_latency: float | None = None
    success_rate: float | None = None
    lift: float | None = None
```
These are never populated. They are future hooks for connecting optimizer results back into the system.


## 7. Integration Plan

### 7.1 Components to Build

| # | Component | Description | ~Lines | Difficulty |
|---|---|---|---|---|
| 1 | Config encoder/decoder | Maps DEAP integer vectors ↔ secretagent dotlist strings | ~50 | Easy |
| 2 | Evaluation wrapper + cache | Calls `run_single()`, handles failures, caches results | ~80 | Easy |
| 3 | Categorical operators | Uniform crossover, random-reset mutation | ~30 | Easy |
| 4 | NSGA-II main loop | DEAP boilerplate: creator, toolbox, main loop | ~60 | Easy |
| 5 | Frontier extraction + viz | Decode frontier individuals, plot accuracy vs cost | ~80 | Easy |
| 6 | Per-benchmark search space defs | Valid (method, model, n_demos) combos per benchmark | ~100 | Medium |
| **Total** | | | **~400** | **Days** |

### 7.2 Where It Lives

```
src/secretagent/optimize/
    config_space.py      # existing (ConfigSpace for in-process)
    __init__.py          # existing (SearchSpace, GridSearchRunner)
    pareto.py            # NEW: NSGA-II optimizer, frontier extraction
    operators.py         # NEW: categorical crossover/mutation
    viz.py               # NEW: Pareto front plotting
```

### 7.3 Design Decision: Per-Benchmark Frontiers

Run NSGA-II separately for each benchmark. Reasons:
- Search spaces differ by benchmark (different interfaces, different valid methods)
- The frontiers themselves are findings ("computational tasks favor L1, compliance tasks have a different frontier shape")
- Cross-benchmark optimization conflates domains with fundamentally different evaluation metrics (accuracy vs F1)

### 7.4 Execution Order

1. Get DEAP working on a toy example (minimize x^2, maximize -x^2) to verify setup
2. Build encoder/decoder for rulearena airline (smallest, best-understood domain)
3. Wire up to `run_single()` with caching
4. Run NSGA-II on rulearena airline, verify frontier matches exhaustive grid
5. Extend to all 3 rulearena domains
6. Generalize to other benchmarks
7. Add visualization and reporting

### 7.5 Problem B Implementation Plan (Architecture Search)

Problem B is the extension after Problem A is complete. NSGA-II is NOT replaced; it becomes a subroutine inside the outer MCTS loop. The two layers compose as bi-level optimization:

```
Outer loop (MCTS): discovers topology candidates
  |
  |  topology = [extract -> verify -> calculate]
  v
  Inner loop (NSGA-II): optimizes configs for this topology
    |
    |  Pareto frontier for this topology
    v
    Return (frontier quality, topology) back to MCTS as evaluation signal
```

#### 7.5.1 What MCTS Does (Outer Loop)

MCTS explores a tree where each node represents a workflow variant (a DAG of interfaces):

- **Selection**: Pick which existing workflow variant to refine (UCB balances exploration vs exploitation)
- **Expansion**: Use an LLM to propose a code modification to the workflow (add a node, remove a node, change wiring). This is what AFLOW does.
- **Evaluation**: Run the modified workflow through NSGA-II (Problem A) or a quick pilot evaluation to get accuracy + cost
- **Backpropagation**: Update the tree with results so future selections prefer promising regions

#### 7.5.2 Topology Representation

Two options, both compatible with secretagent:

**Option 1: Code-based (AFLOW style)**. A workflow is a Python function body that chains `@interface` calls. The LLM generates/modifies this code. Maps to the `orchestrate` factory.

```python
# Example topology discovered by MCTS:
def discovered_workflow(problem_text, rules):
    params = extract_params(problem_text)        # LLM node
    verified = verify_extraction(params, rules)   # NEW: LLM verification node
    if verified.confidence < 0.8:
        params = re_extract_params(problem_text, verified.feedback)  # retry
    return calculator(params)                     # Python node
```

**Option 2: Graph-based (GPTSwarm style)**. A workflow is a directed graph of (interface, method) pairs. Edges are data flow. Optimizer adds/removes edges via learned probabilities. Maps to existing `@interface` + `implement_via()` system.

Recommendation: code-based (Option 1) because secretagent's `orchestrate` factory already produces code-based workflows, and code can express conditionals/loops that graphs cannot.

#### 7.5.3 Components to Build (on top of Problem A)

| # | Component | Description | Depends on |
|---|---|---|---|
| 1 | Topology representation | Data structure for workflow DAGs (list of nodes + edges, or code string) | Nothing |
| 2 | Topology mutator | LLM-driven: given a workflow + its performance, propose a modified workflow | Component 1 |
| 3 | MCTS tree | Tree nodes = topologies, edges = mutations. UCB selection, LLM expansion. | Components 1-2 |
| 4 | Bi-level evaluator | For a given topology: run NSGA-II (Problem A) to get its frontier, return frontier quality (e.g., hypervolume) as MCTS reward | Problem A (complete) |
| 5 | Topology catalog | Store discovered topologies + their frontiers for analysis | Component 1 |

#### 7.5.4 What Carries Over from Problem A

Everything built for Problem A is reused:
- Config encoder/decoder (unchanged)
- Evaluation wrapper + cache (unchanged)
- Categorical operators (unchanged)
- NSGA-II main loop (called as subroutine)
- Frontier extraction + viz (reused per topology)
- Per-benchmark search space defs (extended with new interfaces)

#### 7.5.5 New Interfaces for Problem B

MCTS may discover that new `@interface` functions are needed (e.g., `verify_extraction`, `decompose_problem`). These would be:
- Defined as new `@interface` stubs with type signatures
- Added to the benchmark's ptools module
- Automatically available to the `orchestrate` factory and MCTS mutator
- Bindable via the same `implement_via()` system

#### 7.5.6 Connection to RQ3 (Meta-Router)

The meta-router asks: "given a new problem instance, which topology should I use?" Problem B produces a catalog of topologies with their Pareto frontiers. The meta-router is a classifier trained on this catalog:

```
Input: problem instance features
Output: which topology (and which config on its frontier) to use
```

This could be a simple rule ("if computational domain, use L1 topology; if compliance domain, use L0F topology") or a learned classifier. The oracle gain analysis from Problem A tells you whether per-instance routing has enough headroom over static domain-level routing to justify the complexity.

#### 7.5.7 Execution Order for Problem B

1. Complete Problem A across all benchmarks (prerequisite)
2. Build topology representation + mutator for one benchmark (rulearena airline)
3. Wire up MCTS with NSGA-II as inner loop
4. Run on rulearena airline, check if MCTS discovers anything better than hand-designed L0F/L1/L3
5. If yes: extend to other benchmarks. If no: report as negative result (still publishable — "hand-designed pipelines are hard to beat automatically")
6. Build topology catalog + meta-router


## 8. Glossary

| Term | Definition |
|---|---|
| **MODO** | Multi-Objective Discrete Optimization |
| **Pareto frontier** | Set of non-dominated solutions (no solution is better on ALL objectives) |
| **Pareto dominance** | A dominates B if A >= B on every objective with strict inequality on at least one |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II. Standard MOO algorithm. |
| **Non-dominated sorting** | Sorting population into fronts by dominance rank |
| **Crowding distance** | Measure of how isolated a point is along the frontier. Higher = more worth keeping. |
| **Hypervolume indicator** | Area/volume between the Pareto front and a reference point. Higher = better frontier. |
| **IGD (Inverted Generational Distance)** | Average distance from reference front points to nearest found point. Lower = better. |
| **Tournament selection** | Pick k random individuals, best one wins. Standard parent selection in GAs. |
| **Uniform crossover** | For each gene, randomly pick from parent A or B. Works for categoricals. |
| **Random reset mutation** | Replace one gene with a random valid value. Works for categoricals. |
| **Surrogate-assisted optimization** | Build a cheap model of the objective function to pre-screen candidates. Saves evaluations. |
| **MCTS** | Monte Carlo Tree Search. Tree-based search with selection, expansion, simulation, backpropagation. Used by AFLOW. |
| **REINFORCE** | Policy gradient RL algorithm. Used by GPTSwarm for edge optimization. |
| **GPTSwarm** | Framework representing agents as optimizable computational graphs (ICML 2024). |
| **AFLOW** | Framework for automated workflow generation via MCTS over code (ICLR 2025). |
| **Problem A** | Config search: optimize settings of fixed pipeline architectures |
| **Problem B** | Architecture search: discover new pipeline topologies |
| **Interface** | secretagent: function stub with type signature + docstring, bound to implementation at runtime |
| **Factory** | secretagent: creates an implementation for an interface (direct, simulate, etc.) |
| **PTool** | A function whose output is simulated by an LLM |
| **Workflow** | A top-level function chaining ptools + calculators |
