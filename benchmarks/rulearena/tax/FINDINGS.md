# Tax benchmark findings

Living document. Add findings as they're confirmed; update when stronger evidence arrives.

---

## Gemini-2.5-flash burns `max_tokens` budget on hidden reasoning

**What.** With `gemini/gemini-2.5-flash` on `unstructured_baseline`, `output_tokens` reflects *thinking + visible* tokens together. Visible text fills only 25-100% of the budget; the rest is invisible reasoning. On complex cases the cap hits before `<answer>` is emitted; the parser falls back to the last `$`-amount and grabs an intermediate calculation value.

**Evidence** (`results/20260425.041752.unstructured_baseline/`, `max_tokens=16384`):

| Case | Level | output_tokens | visible chars | chars/token | Outcome |
|------|-------|---------------|---------------|-------------|---------|
| `tax_0_95` | L0 | 7901 | 6375 | 1.24 | clean: `<answer>14475.68</answer>` ✓ |
| `tax_1_44` | L1 | 16380 (cap) | 7019 | 0.43 | truncated; pred `2.00` (grabbed `$2,` from mid-number) |
| `tax_1_17` | L1 | 16380 (cap) | 4039 | 0.25 | truncated; pred `104062.00` (intermediate gross-receipts line) |

Hidden-thinking ratio scales with case complexity, not input length: simplest case used ~0 hidden tokens; harder L1 case used ~3× visible.

**Caveats.**
- Specific to thinking models (`gemini-2.5-flash`, presumably `gemini-2.5-pro`).
- `gemini-2.5-flash-lite` did not exhibit this on a 1-case workflow smoke (insufficient evidence; revisit).
- Production model `together_ai/deepseek-ai/DeepSeek-V3.1` does NOT have this asymmetry. Airline ran `unstructured_baseline` cleanly at `max_tokens=8192` — tax should be similar; verify on the real run.

**Mitigations measured (n=3 valid: tax_0_95 L0, tax_1_44 L1, tax_1_17 L1):**

| Variant | `reasoning_effort` | `max_tokens` | Correct | output_tokens (per case) | Total cost | Results dir |
|---|---|---|---|---|---|---|
| A (initial) | default | 8192 | 1/3 | 8188 / 8188 / 8188 (all cap) | $0.069 | `20260425.040350.unstructured_baseline/` |
| B | default | 16384 | 1/3 | 7901 / 16380 / 16380 (L1 cap) | $0.110 | `20260425.041752.unstructured_baseline/` |
| C | **low** | 16384 | **3/3** | 3465 / 7164 / 6764 | **$0.052** | `20260425.043240.unstructured_baseline_loweffort_16k/` |
| D | default | 32768 | 3/3 | 7505 / 14190 / 17232 | $0.105 | `20260425.043431.unstructured_baseline_default_32k/` |

**Conclusion.** `reasoning_effort=low` resolves truncation at 2× lower cost than just raising `max_tokens`. C and D produce bit-identical predictions (incl. refunds to the cent) — the extra hidden thinking in D adds no correctness, only spend. `effort=low` also makes Gemini behave more like production DeepSeek (no hidden-thinking budget), so smoke results are a more faithful proxy.

**Recommended smoke-test settings for Gemini.**
```
llm.model=gemini/gemini-2.5-flash llm.reasoning_effort=low llm.max_tokens=16384
```

**Production caveat.** This whole finding concerns Gemini-2.5-flash. Production runs use `together_ai/deepseek-ai/DeepSeek-V3.1`. DeepSeek's `output_tokens` is visible-text-only; the airline run is succeeding at `max_tokens=8192`, and tax should follow. **Verify on the first real DeepSeek run** before assuming.

---

## `simulate` factory parser doesn't strip commas before `float()`

**What.** With `compute_tax_answer.method=simulate` (`structured_baseline`), the LLM may emit comma-formatted dollar amounts inside `<answer>...</answer>` (e.g. `-25,502.0`). `simulate`'s parser calls `float()` on the extracted string, which raises `ValueError: could not convert string to float: '-25,502.0'`. The whole case is lost as an exception.

**Evidence** (`results/20260425.044520.structured_baseline/`):
- `tax_0_95`: `raw_response` = 28 chars literally `<answer>\n-25,502.0\n</answer>`. All 1,770 chars of computation went to `reasoning_content`. Parser raised. Predicted value bubbled up as `**exception raised**: could not convert string to float: '-25,502.0'`. Failure mode = `extraction_failure`.
- `tax_1_44` and `tax_1_17` parsed fine (model happened to format without commas) but `tax_1_17` was ~13% off in pure model quality (-7458.11 vs -6600.99).

**Caveats.**
- Airline does NOT exhibit this because its outputs are integer dollars (no commas naturally emitted, and no `.X` decimal).
- Tax outputs are floats with cents — comma formatting is the LLM default. Affects any float-output domain.
- DeepSeek-V3.1 may not default to comma-formatting in this scaffold; verify on real run.

**Possible fixes (not yet applied):**
- Patch `compute_tax_answer`'s docstring to forbid commas in the answer (cheapest, tax-local).
- Patch `secretagent.implement.core.parse_output` to strip commas before float coercion (right fix, framework-wide; touches code currently in use by the live airline run, so deferred).

**Cross-strategy implication (for the paper).** As strategies shift from explicit-prompt (unstructured) toward scaffold-driven (simulate, react), responsibility for output formatting moves from the prompt designer to the framework. Each handoff exposes a new parser-LLM mismatch surface. Tax surfaces one that airline doesn't.

---

## PoT: schema hallucination on rich pydantic models

**What.** PoT's generated code accesses extracted-params using **form-line surface names** from `forms_text`, not the **pydantic field names**. With `TaxParams` (70 fields, many named after IRS-form labels), the LLM hallucinates dict-style keys that don't exist.

**Evidence** (`results/20260425.045958.pot/`, case `tax_1_17`, has `self_employed=True`):
- Generated code: `tax_params.get("Schedule C (Form 1040)_Line 1 - Gross receipts or sales", 0.0)`
- Actual field name: `gross_receipts`
- Two stacked failures: (a) `.get()` on pydantic raises (the predicted asymmetry); (b) even if dict, the key wouldn't match — silently returns `0.0`, corrupting arithmetic.
- L0 and the simple L1 refund cases (`tax_0_95`, `tax_1_44`) succeeded because their generated code used attribute access or didn't need Schedule C. Complex cases reliably fail.

**Caveats.**
- Surface area is domain-dependent: airline (5 fields, terse names) is much harder to hallucinate against than tax (70 fields, IRS-form labels).
- DeepSeek-V3.1 may behave differently; verify on real run.
- Cost of failure is not zero — `tax_1_17` consumed 17,893 output tokens (highest of any smoke case so far) before giving up.

**Mitigation candidates.** See `benchmarks/rulearena/INFRA_FIXES.md` § 2 (pot pydantic asymmetry); the additional schema-fidelity issue argues for **explicit pydantic-schema injection into the pot prompt**, not just relying on type annotations.

---

## React + Gemini-flash + `extract_*_params.method=simulate` is broken

**What.** Under `make react` with `gemini/gemini-2.5-flash` and the locked-in #8 scoped override `extract_tax_params.method=simulate`, all 3 smoke cases fail at the framework layer (NaN stats — failure before aggregation).

**Evidence** (`results/20260425.050635.react/`):
- `tax_0_95`, `tax_1_44`: `**exception raised**: cannot find final answer` — `simulate.parse_output` at `src/secretagent/implement/core.py:191`. pydantic-ai's Agent calls the extract tool → tool invokes `simulate` → Gemini doesn't emit the scaffold `simulate` expects → parse raises.
- `tax_1_17`: `**exception raised**: No choices returned from LiteLLM` — Gemini API returned empty (transient flake or content filter on this case).

**Conflict between two locked-in decisions, exposed by Gemini smoke:**
- **#8** says: scope `extract_*_params.method=simulate` in react because pydantic-ai can't nest pydantic-ai tools.
- **Empirical from #6 sub-step 5.6 + this run**: `simulate` + Gemini-flash is unreliable — Gemini doesn't always emit `simulate`'s `<answer>` scaffold when invoked through pydantic-ai's tool dispatch.

The two collide only for **(Gemini-flash × tax × react)**. DeepSeek-V3.1 (production) reliably emits the scaffold; airline ran clean with the same config.

**Action: defer.** Documented as known smoke-only failure. Production with DeepSeek is the real test. Revisit if production also fails.

---

## Cache key is `(prompt, model)` only

Tuning `max_tokens` / `reasoning_effort` / `temperature` via DOTPAIRS does NOT bust the cache (key is computed in `src/secretagent/llm_util.py:163` from `_llm_impl(prompt, model)` args only). Telltale of a stale-cache hit: re-run completes in <1s with bit-identical predictions and stats. Force a fresh call with `cachier.enable_caching=false`.
