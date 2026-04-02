"""Search space definitions for rulearena benchmark domains.

Each domain defines:
  - dims: list of SearchDimensions (the axes NSGA-II searches over)
  - fixed: list of dotlist strings applied to every config point
"""

from secretagent.optimize.encoder import SearchDimension

# -- Workflows available per domain --

AIRLINE_WORKFLOWS = [
    "ptools.l1_extract_workflow",
    "ptools.l0f_cot_workflow",
    # "ptools.l3_react_workflow",  # TODO: add when L3 is wired up for sweep
]

NBA_WORKFLOWS = [
    "ptools.l1_extract_workflow",
    "ptools.l0f_cot_workflow",
]

TAX_WORKFLOWS = [
    "ptools.l1_extract_workflow",
    "ptools.l0f_cot_workflow",
]

# -- Models --

MODELS = [
    "together_ai/deepseek-ai/DeepSeek-V3",
    "together_ai/deepseek-ai/DeepSeek-V3.1",
    # "claude-haiku-4-5-20251001",  # needs Anthropic API key
]

# -- Fixed overrides (not searched) --

FIXED = [
    "ptools.compute_rulearena_answer.method=direct",
]


# -- Search space builders --

def airline_space() -> tuple[list[SearchDimension], list[str]]:
    dims = [
        SearchDimension(
            key="ptools.compute_rulearena_answer.fn",
            values=AIRLINE_WORKFLOWS,
        ),
        SearchDimension(
            key="llm.model",
            values=MODELS,
        ),
    ]
    return dims, list(FIXED)


def nba_space() -> tuple[list[SearchDimension], list[str]]:
    dims = [
        SearchDimension(
            key="ptools.compute_rulearena_answer.fn",
            values=NBA_WORKFLOWS,
        ),
        SearchDimension(
            key="llm.model",
            values=MODELS,
        ),
    ]
    return dims, list(FIXED)


def tax_space() -> tuple[list[SearchDimension], list[str]]:
    dims = [
        SearchDimension(
            key="ptools.compute_rulearena_answer.fn",
            values=TAX_WORKFLOWS,
        ),
        SearchDimension(
            key="llm.model",
            values=MODELS,
        ),
    ]
    return dims, list(FIXED)


DOMAIN_SPACES = {
    "airline": airline_space,
    "nba": nba_space,
    "tax": tax_space,
}
