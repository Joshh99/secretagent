# Changes - April 1

## Tax extraction: return type and schema overhaul

- Changed `extract_tax_params` return type from `dict` to `str` to avoid
  `ast.literal_eval` failures on JSON booleans (`true`/`false`), trailing
  commas, and `//` comments. `l1_extract_workflow` now parses the raw
  string via `json.loads` after stripping JS-style comments.
- Expanded the `extract_tax_params` docstring with the full field schema
  (~70 fields), line references, and extraction rules. The docstring is
  the LLM prompt — this makes extraction more reliable.

## Bugfixes

- `SimulateFactory.parse_output` (framework-level, `implement/core.py`):
  raised `AttributeError` for `str` return types when the LLM omitted
  `<answer>` tags. Now falls back to returning the raw text.
- `calculators/airline.py`: cast return value to native `int`. The
  underlying fee tables produce `numpy.int64`, which broke pydantic-ai
  tool-result serialization in L3 experiments.

## Pilot verification

- Ran n=5 pilots (n=3 for L3) across all 14 experiment configurations.
  All pass. See `rulearena_secretagent_execution_guide.md` in the
  project root for a full execution reference.

# Changes - March 28

## Prompt fixes

- Standardized all three CoT prompt templates (airline, NBA, tax) to use
  `<answer>` tags, matching the secretagent `prompt_llm` factory convention.
- Removed duplicate answer-format instruction in the tax domain. The
  `_TAX_PROMPT_TEMPLATE` in `ptools.py` is now the single source of truth;
  `tax_cot.txt` just passes through `$forms_text`.
- Removed rules-text truncation (`[:6000]` in NBA, `[:3000]` in L1 airline)
  so all complexity levels see the full rule set.

## Evaluator improvements

- Added `correct_tolerance` metric using `np.isclose` (tight tolerance)
  alongside the existing 1% relative tolerance `correct`.
- Added `failure_mode` field to `compare_predictions`: `none`,
  `calculation_error`, or `extraction_failure`. Correctly handles the
  `**exception raised**` string from the evaluator base class.

## Tax calculator robustness

- `_taxpayer_defaults()` auto-fills missing TaxPayer fields with
  type-appropriate zero values, preventing crashes when the LLM omits
  optional pydantic fields.

## Tests

- Added `tests/test_rulearena.py` (28 tests):
  - `TestConfig` (3) -- YAML loading, dotlist overrides, invalid domain
  - `TestCalculators` (4) -- known anchors: airline=1275, tax=4747.5
  - `TestSchema` (4) -- Case/Dataset structure, load_dataset
  - `TestMetrics` (9) -- tolerance, isclose, failure_mode classification
  - `TestIntegrationL0` (2) -- oracle baseline, no LLM
  - `TestIntegrationL0F` (3) -- CoT, 1 LLM call/example
  - `TestIntegrationL1` (3) -- extraction + calculator
  - L3 excluded from tests (no `max_steps` cap in pydantic-ai agent)
