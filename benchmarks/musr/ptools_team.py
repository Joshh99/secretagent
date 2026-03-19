"""Interfaces for MUSR team allocation.

Migrated from AgentProject v2 ptools. Constraint-based role assignment
with severity-aware scoring.

Key insight (v2): Many instances have NO perfect assignment — every choice
violates some constraint. The extraction must preserve DEGREE of unfitness
(not just binary disqualified/not) so the evaluation can pick the least
bad option.
"""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def extract_profiles(narrative: str) -> str:
    """Extract roles, person profiles with fit SCORES, and constraints.

    Read the narrative carefully and extract:

    1. roles: each role/position and its requirements

    2. people: for each person, extract:
       - skills: relevant skills, qualifications, experience
       - role_fit: for EACH role, a fit assessment with:
           - score: integer 1-5 where:
               1 = severely unfit (phobia, allergy, total inability, dangerous)
               2 = poor fit (discomfort, lack of skill, but could manage minimally)
               3 = neutral (no strong evidence for or against)
               4 = good fit (relevant skills or experience)
               5 = excellent fit (strong skills, experience, enthusiasm)
           - summary: 1-2 sentences explaining the score with SPECIFIC evidence
             from the narrative. Include both positives AND negatives.
       - work_style: collaboration preferences, interpersonal notes

    3. constraints: list of constraints/restrictions, each with:
       - type: "conflict", "synergy", "requirement", or "preference"
       - people: list of people involved
       - severity: "severe", "moderate", "mild", or "positive" (for synergies)
       - description: what the constraint is

    CRITICAL SCORING RULES:
    - Score EVERY person for EVERY role — do not skip any combination
    - A person can be unfit for ALL roles (score 1-2 for everything)
    - A person's unfitness for one role may be LESS severe than for another —
      capture this difference in scores (e.g. score 2 for Caretaker but score 1
      for Cleaner means Caretaker is the "less bad" option)
    - Physical/health issues (allergies, phobias) that directly conflict with
      a role's core requirements → score 1
    - Discomfort or mild issues that could be managed → score 2
    - Pay attention to COMPARATIVE severity: "uneasy" is less severe than
      "terrified"; "sometimes forgets" is less severe than "completely unable"
    - Look for hidden qualifiers: "despite X, she managed Y" suggests resilience
    - Experience in a role (even imperfect) scores higher than no experience
    """


@interface
def evaluate_allocations(narrative: str, profiles: str, question: str, choices: list) -> str:
    """Evaluate each allocation choice and pick the best one.

    You receive:
    1. The FULL original narrative (re-read it for details extraction may have missed)
    2. Extracted profiles with role_fit SCORES (1-5) for each person×role
    3. The multiple-choice options

    For each choice, compute:
    - total_fit_score: sum of each person's role_fit score for their assigned role
    - score_breakdown: show the math (e.g. "Alex→Caretaker(2) + Mia→Cleaner(5) = 7")
    - critical_issue: the single biggest problem with this allocation
    - strengths: what works well
    - score: "strong", "moderate", or "weak"

    Decision process:
    1. Compute total_fit_score for each choice
    2. Count the number of severe assignments (score 1) in each choice
    3. PREFER choices with FEWER score-1 assignments (avoid putting anyone
       in a role that's physically harmful or impossible)
    4. Among choices with equal severe-assignment counts, prefer higher total score
    5. Consider constraints (conflicts, synergies) as tiebreakers

    KEY INSIGHT: When ALL choices have problems, you must pick the LEAST BAD one.
    Do NOT default to "all choices are equally bad" — use the scores to differentiate.
    A score-2 assignment (uncomfortable but manageable) is meaningfully better than
    a score-1 assignment (dangerous or impossible).
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the narrative and determine the best team allocation.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Solve by extracting profiles, evaluating allocations, then matching."""
    profiles = extract_profiles(narrative)
    text = evaluate_allocations(narrative, profiles, question, choices)
    return extract_index(text, choices)
