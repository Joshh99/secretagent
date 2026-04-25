# RuleArena infrastructure fixes

Cross-domain framework-level changes encountered while building per-domain benchmarks. Not yet applied. Living list — finalize with Prof.

Each entry: **What** (symptom) / **Where** (file:line) / **Fix** / **Risk** / **Evidence**.

---

## 1. `simulate` / `prompt_llm` answer coercion fails on comma-formatted floats

**What.** LLM emits `<answer>-25,502.0</answer>` (natural for dollar amounts). Parser calls `float('-25,502.0')` → `ValueError: could not convert string to float`. Case is lost as `**exception raised**`. Affects any float-output domain.

**Where.** Two sites, same pattern:
- `src/secretagent/implement/core.py:159-160` — `simulate.parse_output`: `if return_type in [int, str, float]: return return_type(final_answer)`
- `src/secretagent/implement/core.py:274-275` — `_extract_answer` (used by `prompt_llm`): same line, same risk.

**Fix.** Add a numeric-normalization helper before coercion: strip commas and `$`, accept ` -123` etc. Apply at both call sites. ~3 lines + 1 helper.

```python
def _coerce_numeric(s: str, t):
    if t in (int, float):
        s = s.strip().replace(',', '').replace('$', '')
    return t(s)
```

**Risk.** Touches framework code currently in use by the live airline run; defer until airline finishes. Net safe: airline's outputs are integer dollars (no commas naturally emitted, no decimals); tax benefits; nba unaffected (verdict bool).

**Evidence.** `benchmarks/rulearena/tax/FINDINGS.md` § "simulate factory parser doesn't strip commas" + `tax/results/20260425.044520.structured_baseline/` (case `tax_0_95`).

---

## 2. PoT pydantic / dict type-asymmetry

**What.** `extract_*_params(query) -> *Params` returns a pydantic model; `compute_*_calculator(params: dict) -> ...` declares dict in its annotation. PoT prompts the LLM with both signatures, so generated code naturally writes `params["x"] = …` after extraction — which crashes on the pydantic instance (`TypeError: 'TaxParams' object does not support item assignment`). ~1/3 plumbing failures observed in airline n=3.

**Where.** Domain ptools (`airline/ptools.py`, `tax/ptools.py`, future `nba/ptools.py`). Mirrored intentionally for cross-domain parity; deferred for unified fix.

**Possible fixes (pick one cross-domain):**
- **A.** Type `compute_*_calculator(params: <DomainParams>)` so PoT code is generated against the pydantic API (e.g. `params.x` access, `model_copy(update=...)`). Cleanest API.
- **B.** Have `extract_*_params` return a dict (lose pydantic validation; cheapest at API surface).
- **C.** Inject `params = params.model_dump() if hasattr(params, 'model_dump') else params` preamble into the PoT sandbox before the generated code executes. Fix is in the `program_of_thought` factory, not in domain code. Most domain-agnostic.
- **D.** Strengthen `_*_calc_fn`'s 4-convention shim to short-circuit on pydantic instances earlier (already partly done via `hasattr(raw, 'model_dump')`) — but PoT's generated code may crash on `params["x"] = ...` *before* reaching the shim.
- **E. (NEW)** Inject the **pydantic schema (field names + types + descriptions)** into the PoT prompt explicitly. Without this, the LLM hallucinates dict keys based on the prompt's surface form (e.g. `"Schedule C (Form 1040)_Line 1 - Gross receipts or sales"` instead of the actual field `gross_receipts`). C alone doesn't fix this — the lookup would silently return defaults. See tax FINDINGS § "PoT: schema hallucination on rich pydantic models". Severity scales with schema size: airline (5 fields) ≪ tax (70 fields).

**Risk.** A and B change the API across all 3 domains; coordinated change. C touches the framework. D is safest but doesn't fully solve. E touches the framework and may also reduce fragility on unstructured prompts. C+E together is probably the right combo.

**Evidence.** Memory note `project_pot_pydantic_dict_asymmetry.md`; tax `results/20260425.045958.pot/` (case `tax_1_17` — has `self_employed=True`, triggers both `.get()` failure and hallucinated key); airline ~1/3 plumbing failures.

---

## 3. (placeholder)

Add new entries here as further infra issues surface during smoke / production runs.
