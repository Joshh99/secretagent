"""Tools for the date_understanding benchmark.

The task: given a date understanding multiple-choice question, determine
the correct date by reasoning about calendars, date arithmetic, and
temporal relationships.

Derived from the BIG-Bench Hard date_understanding task.
"""

from typing import Any, List, Tuple

from secretagent.core import interface, implement_via
from secretagent.evaluate import Evaluator


class DateUnderstandingEvaluator(Evaluator):
    def compare_predictions(self, predicted_output, expected_output) -> dict[str, Any]:
        def normalize(s):
            return str(s).strip().strip('()')
        return dict(correct=float(normalize(predicted_output) == normalize(expected_output)))


# ── sub-tools ────────────────────────────────────────────────────────────────

@interface
def find_date_from_sentence(description: str) -> str:
    """Extract a date from a natural-language description and return it
    as a string in MM/DD/YYYY format.

    Examples:
    >>> find_date_from_sentence("Today is Christmas Eve of 1937.")
    '12/24/1937'
    >>> find_date_from_sentence("the first Monday of 2019")
    '01/07/2019'
    """
    ...

@interface
def normalize_date_order(date_str: str) -> str:
    """Take a date string that may use a non-American ordering (e.g.
    DD/MM/YYYY) and return it in standard American MM/DD/YYYY format.

    If the input is already in MM/DD/YYYY format, return it unchanged.

    Examples:
    >>> normalize_date_order('02/01/1987')  # UK format, day-first
    '01/02/1987'
    """
    ...

@interface
def find_time_difference(date1: str, date2: str) -> int:
    """Given two dates as MM/DD/YYYY strings, return the signed difference
    between them in hours. A positive result means date2 is after date1.

    >>> find_time_difference('01/01/2020', '01/02/2020')
    24
    """
    ...

@interface
def identify_time_difference(description: str) -> int:
    """Parse a natural-language description of a time offset and return
    the corresponding number of hours (signed).

    Examples:
    >>> identify_time_difference("one week ago")
    -168
    >>> identify_time_difference("24 hours later")
    24
    >>> identify_time_difference("a month ago")
    -720
    """
    ...

@interface
def adjust_date_by_hours(date_str: str, hours: int) -> str:
    """Apply an offset in hours to a date and return the resulting date
    as a string in MM/DD/YYYY format.

    >>> adjust_date_by_hours('01/01/2020', 48)
    '01/03/2020'
    """
    ...

@interface
def parse_question(question: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Parse a date understanding question into its component parts.

    Returns (scenario, options) where:
      - scenario is the natural-language description including the date
        context and the question being asked
      - options is a list of (letter, date_str) pairs, e.g.
        [('A', '12/24/1936'), ('B', '01/10/1937'), ...]

    Examples:
    >>> parse_question("Today is Christmas Eve of 1937. What is the date one year ago from today in MM/DD/YYYY?\\nOptions:\\n(A) 12/24/1862\\n(B) 01/10/1937\\n(C) 10/24/1936\\n(D) 12/24/1936")
    ('Today is Christmas Eve of 1937. What is the date one year ago from today in MM/DD/YYYY?', [('A', '12/24/1862'), ('B', '01/10/1937'), ('C', '10/24/1936'), ('D', '12/24/1936')])
    """
    ...

@interface
def select_closest_option(date_str: str, options: List[Tuple[str, str]]) -> str:
    """Given a computed date in MM/DD/YYYY format and a list of
    multiple-choice options (each a (letter, date_str) pair), return
    the letter of the option whose date is closest to the input date.

    Returns just the letter, e.g. 'D'.

    >>> select_closest_option('12/24/1936', [('A', '12/24/1862'), ('B', '01/10/1937'), ('C', '10/24/1936'), ('D', '12/24/1936')])
    'D'
    """
    ...

# ── top-level interface ───────────────────────────────────────────────────────

@interface
def answer_date_question(question: str) -> str:
    """Given a date understanding multiple-choice question, return the correct
    option label, e.g. '(A)'.

    The input includes a scenario involving dates, a question about the
    resulting date, and labeled answer options in MM/DD/YYYY format.
    """
    ...

@interface
def answer_date_question_orchestrated(question: str) -> str:
    """Given a date understanding multiple-choice question, return the correct
    option label in parentheses, e.g. '(A)'.
    """
    ...

#
# zeroshot unstructured model is a workflow - first get a string
# answer, then use a second tool to extract the option letter
#

@implement_via('prompt_llm', prompt_template_file='prompt_templates/zeroshot.txt')
def zeroshot_answer_date_question(question: str) -> str:
    ...

@implement_via('simulate')
def extract_option_letter(llm_output: str) -> str:
    """Given raw LLM output, extract and return the multiple-choice letter
    in parentheses, e.g. '(A)'.
    """
    ...

def zeroshot_unstructured_workflow(question: str) -> str:
    """Workflow for using a zero-shot prompt and coercing the answer to a letter.

    To run the zeroshot unstructured model, bind this to the
    implementation of 'answer_date_question'.
    """
    llm_output = zeroshot_answer_date_question(question)
    return extract_option_letter(llm_output)
