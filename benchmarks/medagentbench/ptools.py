"""MedAgentBench ptools: interfaces for solving medical EHR tasks via FHIR API.

Three tiers of tools for different experiment strategies:
  Low-level:  fhir_get_iface, fhir_post_iface (raw HTTP)
  Mid-level:  find_patient, get_observations, record_vital, create_order (domain operations)
  High-level: parse_task, extract_answer (LLM reasoning)

Experiment levels:
  L0 paper_baseline: multi-turn text loop (GET/POST/FINISH)
  L1 structured_tools: pydantic-ai structured tool calling
  L2 pot: single-pass code generation with FHIR tools
  L3 codeact: iterative code generation with error feedback
  L4 orchestrate: auto-generated workflow composing all tiers
"""

from secretagent.core import interface


# ──────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────

@interface
def solve_medical_task(instruction: str, context: str) -> list[str]:
    """Solve a medical EHR task given an instruction and FHIR context.

    The instruction describes what to do (lookup patient, record vital,
    order medication, etc.). The context contains the FHIR API base URL
    and task-specific details (timestamps, lab codes, dosing instructions).

    Return ONLY the exact values requested as a list:
    - Patient lookups: ["S6534835"]
    - Numeric values: [28] or [2.3]
    - Write tasks (record/order): []
    - Not found: [-1]
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# Low-level: raw FHIR HTTP (for PoT / direct code)
# ──────────────────────────────────────────────────────────────────────

@interface
def fhir_get_iface(url: str) -> str:
    """Send a GET request to the FHIR server.

    The url should be the full FHIR endpoint with query parameters,
    e.g. "http://localhost:8080/fhir/Patient?family=Smith&birthdate=1990-01-01"

    Returns the JSON response as a string. Use json.loads() to parse it.
    Returns an error message string if the request fails.
    """
    ...


@interface
def fhir_post_iface(url: str, payload: str) -> str:
    """Send a POST request to create a FHIR resource.

    The url is the FHIR endpoint, e.g. "http://localhost:8080/fhir/MedicationRequest".
    The payload must be a JSON string with the resource data.

    Returns a success message on success, or an error message if invalid.
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# Mid-level: domain operations (deterministic Python)
# ──────────────────────────────────────────────────────────────────────

@interface
def find_patient(fhir_base: str, family: str, given: str, birthdate: str) -> str:
    """Search for a patient by family name, given name, and date of birth.

    Returns the patient's MRN (e.g. "S6534835") or "Patient not found".
    """
    ...


@interface
def get_patient_dob(fhir_base: str, mrn: str) -> str:
    """Look up a patient's date of birth by MRN.

    Returns ISO date string (e.g. "1963-01-15") or "" if not found.
    """
    ...


@interface
def get_observations(fhir_base: str, mrn: str, code: str, hours: int) -> str:
    """Get lab/vital observations for a patient as a JSON string.

    Args:
        fhir_base: FHIR API base URL
        mrn: patient MRN
        code: observation code (e.g. "MG", "K", "GLU", "A1C")
        hours: look back this many hours (0 = all time)

    Returns a JSON string: list of {"value": float, "time": "ISO string"}
    sorted by time descending. Returns "[]" if none found.
    """
    ...


@interface
def calculate_age(dob: str, reference_date: str) -> int:
    """Calculate age in whole years from DOB to reference date.

    Both dates in ISO format. Returns integer age.
    """
    ...


@interface
def record_vital(fhir_base: str, mrn: str, code: str, value: str, timestamp: str) -> str:
    """Record a vital sign observation for a patient.

    Args:
        fhir_base: FHIR API base URL
        mrn: patient MRN
        code: flowsheet code (e.g. "BP")
        value: measurement string (e.g. "118/77 mmHg")
        timestamp: ISO datetime

    Returns "success" or an error message.
    """
    ...


@interface
def create_order(fhir_base: str, mrn: str, order_type: str, params: str, timestamp: str) -> str:
    """Create a medication order or service request.

    Args:
        fhir_base: FHIR API base URL
        mrn: patient MRN
        order_type: "medication" or "service"
        params: JSON string with order details:
          For medication: {"ndc": "...", "display": "...", "dose": 1.0,
                           "dose_unit": "g", "rate": 1.0, "rate_unit": "h", "route": "IV"}
          For service: {"system": "http://loinc.org", "code": "2823-3",
                        "display": "...", "priority": "stat", "note": "", "occurrence": ""}
        timestamp: ISO datetime for authoredOn

    Returns "success" or an error message.
    """
    ...


# ──────────────────────────────────────────────────────────────────────
# High-level: LLM reasoning (bound to simulate)
# ──────────────────────────────────────────────────────────────────────

@interface
def parse_task(instruction: str, context: str) -> str:
    """Parse the medical task instruction and extract all parameters as JSON.

    Read the instruction and context carefully. Return a JSON string with:
    - task_type: one of "lookup", "age", "record", "get_lab", "conditional_order",
                 "average", "recent_value", "referral", "multi_step", "check_stale"
    - fhir_base: FHIR API base URL (from context)
    - patient_mrn: patient MRN if given (e.g. "S6534835")
    - family: family/last name if given
    - given: given/first name if given
    - dob: date of birth if given (ISO format)
    - current_time: current timestamp from context
    - lab_code: observation code (e.g. "MG", "K", "GLU", "A1C")
    - measurement: value to record (e.g. "118/77 mmHg")
    - flowsheet_id: flowsheet code (e.g. "BP")
    - ndc_code: NDC medication code
    - loinc_code: LOINC code
    - snomed_code: SNOMED code
    - dosing: full dosing instruction text
    - note: free text (referral notes etc.)
    - hours: time window in hours (e.g. 24)

    Use null for any parameter not found.

    Example:
    >>> parse_task("What's the MRN of Peter Stafford DOB 1932-12-29?", "FHIR API base URL: http://localhost:8080/fhir/")
    '{"task_type": "lookup", "family": "Stafford", "given": "Peter", "dob": "1932-12-29", "fhir_base": "http://localhost:8080/fhir/"}'
    """
    ...


@interface
def extract_answer(observations_json: str, question: str) -> str:
    """Extract the specific answer from FHIR observation data.

    Given a JSON string of observations (from get_observations) and
    a description of what to extract, return just the value.

    Examples:
    >>> extract_answer('[{"value": 2.3, "time": "2023-11-12T14:30:00"}]', "most recent value")
    "2.3"
    >>> extract_answer('[{"value": 97}, {"value": 110}, {"value": 87}]', "average of all values")
    "98.0"

    Return "-1" if the data is empty or the value can't be determined.
    """
    ...
