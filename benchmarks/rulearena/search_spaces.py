"""Search space definitions for rulearena benchmark domains.

Each domain defines:
  - dims: list of SearchDimensions (model only — the axis NSGA-II searches over)
  - fixed: empty (method overrides are in DOMAIN_METHODS, applied by run_pareto.py)
  - methods: dict of method_name -> list of dotlist overrides

Methods are selected via dotlist overrides (outer loop in run_pareto.py),
matching the Natural Plan pattern. Each method's overrides are copied
exactly from the Makefile targets.
"""

from secretagent.optimize.encoder import SearchDimension

# -- Models --

MODELS = [
    "together_ai/deepseek-ai/DeepSeek-V3",
    "together_ai/deepseek-ai/DeepSeek-V3.1",
    "together_ai/openai/gpt-oss-20b",
    "together_ai/openai/gpt-oss-120b",
    "together_ai/Qwen/Qwen3.5-9B",
    "together_ai/google/gemma-3n-E4B-it",
    # "claude-haiku-4-5-20251001",  # needs Anthropic API key
    # NOTE: Qwen3.5-9B and gemma-3n-E4B-it do not support tool use.
    # react (simulate_pydantic) will fail with these models (scored 0%).
]

# -- Methods per domain --
# Each method maps to a list of dotlist overrides, copied exactly from the
# Makefile targets. run_pareto.py iterates methods in the outer loop.

AIRLINE_METHODS = {
    "unstructured_baseline": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow",
    ],
    "workflow": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow",
    ],
    "pot": [
        "ptools.pot_airline.method=program_of_thought",
        "ptools.pot_airline.inject_args=true",
        "ptools.pot_airline.tools=[]",
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.pot_workflow",
    ],
    "react": [
        "ptools.compute_rulearena_answer.method=simulate_pydantic",
        "ptools.compute_rulearena_answer.tools=[ptools.extract_airline_params,ptools.compute_airline_calculator]",
    ],
}

NBA_METHODS = {
    "unstructured_baseline": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow",
    ],
    "workflow": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow",
    ],
    "pot": [
        "ptools.pot_nba.method=program_of_thought",
        "ptools.pot_nba.inject_args=true",
        "ptools.pot_nba.tools=[]",
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.pot_workflow",
    ],
    "react": [
        "ptools.compute_rulearena_answer.method=simulate_pydantic",
        "ptools.compute_rulearena_answer.tools=[ptools.extract_nba_params]",
    ],
}

TAX_METHODS = {
    "unstructured_baseline": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l0f_cot_workflow",
    ],
    "workflow": [
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.l1_extract_workflow",
    ],
    "pot": [
        "ptools.pot_tax.method=program_of_thought",
        "ptools.pot_tax.inject_args=true",
        "ptools.pot_tax.tools=[]",
        "ptools.compute_rulearena_answer.method=direct",
        "ptools.compute_rulearena_answer.fn=ptools.pot_workflow",
    ],
    "react": [
        "ptools.compute_rulearena_answer.method=simulate_pydantic",
        "ptools.compute_rulearena_answer.tools=[ptools.extract_tax_params,ptools.compute_tax_calculator]",
    ],
}


# -- Search space builders --

def airline_space() -> tuple[list[SearchDimension], list[str]]:
    """Model dimension only; methods are applied as dotlist overrides."""
    dims = [
        SearchDimension(key="llm.model", values=MODELS),
    ]
    return dims, []


def nba_space() -> tuple[list[SearchDimension], list[str]]:
    dims = [
        SearchDimension(key="llm.model", values=MODELS),
    ]
    return dims, []


def tax_space() -> tuple[list[SearchDimension], list[str]]:
    dims = [
        SearchDimension(key="llm.model", values=MODELS),
    ]
    return dims, []


DOMAIN_SPACES = {
    "airline": airline_space,
    "nba": nba_space,
    "tax": tax_space,
}

DOMAIN_METHODS = {
    "airline": AIRLINE_METHODS,
    "nba": NBA_METHODS,
    "tax": TAX_METHODS,
}
