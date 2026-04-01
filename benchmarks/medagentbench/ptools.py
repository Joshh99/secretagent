"""MedAgentBench ptools: interface for solving medical EHR tasks via FHIR API.

Supports four experiment levels:
  L0 paper_baseline: multi-turn text loop (GET/POST/FINISH)
  L1 structured_tools: pydantic-ai structured tool calling
  L2 pot: single-pass code generation with FHIR tools
  L3 codeact: iterative code generation with error feedback
"""

from secretagent.core import interface


@interface
def solve_medical_task(instruction: str, context: str) -> list[str]:
    """You are an expert in using FHIR functions to assist medical professionals.
    You are given a question and a set of possible functions.
    Based on the question, you will need to make one or more function/tool calls
    to achieve the purpose.

    You have access to two tools:
    - fhir_get(url): Send a GET request to the FHIR server. The url should be
      the full FHIR endpoint with query parameters.
    - fhir_post(url, payload): Send a POST request. url is the FHIR endpoint,
      payload is a JSON string with the resource data.

    The context argument contains the FHIR API base URL, available FHIR
    endpoint definitions, and any task-specific context (timestamps, lab codes,
    dosing instructions, etc.).

    Use the tools to interact with the FHIR server and solve the given task.

    IMPORTANT: Return ONLY the exact values requested, nothing else.
    - For patient lookups: just the MRN (e.g. ["S6534835"])
    - For numeric values: just the number (e.g. ["28"] or ["2.3"])
    - For dates/times: just the ISO timestamp
    - Do NOT include explanations, descriptions, or units in the answer list.
    - If no value is found, return ["-1"].
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# FHIR tool interfaces for PoT (L2)
#
# PoTFactory only shows Interface stubs in the prompt. Plain callables
# are invisible. These wrappers ensure fhir_get/fhir_post signatures
# appear so the LLM knows what tools to call in generated code.
# ──────────────────────────────────────────────────────────────────────

@interface
def fhir_get_iface(url: str) -> str:
    """Send a GET request to the FHIR server.

    The url should be the full FHIR endpoint with query parameters,
    e.g. "http://localhost:8080/fhir/Patient?family=Smith&birthdate=1990-01-01"

    The response is returned as a JSON string. Use json.loads() to parse it.
    Returns an error message string if the request fails.
    """
    ...


@interface
def fhir_post_iface(url: str, payload: str) -> str:
    """Send a POST request to create a FHIR resource.

    The url is the FHIR endpoint, e.g. "http://localhost:8080/fhir/MedicationRequest".
    The payload must be a JSON string with the resource data.

    Returns "POST request accepted and executed successfully..." on success,
    or an error message if the JSON payload is invalid.
    """
    ...
