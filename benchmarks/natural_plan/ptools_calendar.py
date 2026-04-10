"""Task-specific interfaces for NaturalPlan calendar scheduling.

Decomposition derived from LLM reasoning traces:
1. parse_schedules — extract participants, busy slots, duration, preference
2. find_available_slots — compute free slot intersections
3. select_and_format — pick best slot, format answer
"""

from secretagent.core import interface


@interface
def parse_schedules(prompt: str) -> str:
    """Extract all scheduling information from the calendar problem.

    Return a dict with these keys:
    - participants: list of participant names
    - duration_minutes: meeting duration (30 or 60)
    - working_hours: [start_str, end_str] e.g. ["9:00", "17:00"]
    - days: list of day names e.g. ["Monday", "Tuesday"]
    - preference: "earliest" or "latest" or null
    - schedules: dict mapping participant name to dict mapping day to
      list of busy intervals e.g. {"Alice": {"Monday": [["9:00","10:00"]]}}

    Use 24-hour format. For participants with "wide open" calendars, use empty lists.

    >>> parse_schedules("...schedule a meeting for Alice and Bob for half an hour...")
    {"participants": ["Alice", "Bob"], "duration_minutes": 30, "working_hours": ["9:00", "17:00"], "days": ["Monday"], "preference": "earliest", "schedules": {"Alice": {"Monday": [["9:00", "10:00"]]}, "Bob": {"Monday": []}}}
    """


@interface
def find_available_slots(prompt: str, schedules: str) -> str:
    """Find all valid meeting slots that work for all participants.

    Given the original prompt and extracted schedules, compute free
    intervals for each participant, intersect them, and filter by
    the required meeting duration.

    Return a list of dicts: [{"day": str, "start": str, "end": str}, ...]
    sorted by day order then start time. Use 24-hour "HH:MM" format.

    >>> find_available_slots("...", {"participants": ["Alice","Bob"], "duration_minutes": 30, ...})
    [{"day": "Monday", "start": "11:00", "end": "11:30"}, {"day": "Monday", "start": "13:00", "end": "13:30"}]
    """


@interface
def select_and_format(slots: str, preference: str) -> str:
    """Pick the best slot based on preference and format the answer.

    If preference is "earliest", pick the first slot.
    If preference is "latest", pick the last slot.

    Return exactly: 'Here is the proposed time: {Day}, {HH:MM} - {HH:MM}'

    >>> select_and_format([{"day": "Monday", "start": "11:00", "end": "11:30"}], "earliest")
    'Here is the proposed time: Monday, 11:00 - 11:30'
    """


@interface
def calendar_scheduling(prompt: str) -> str:
    """Solve a calendar scheduling problem.
    Return: 'Here is the proposed time: {Day}, {HH:MM} - {HH:MM}'
    """
    ...


def calendar_workflow(prompt: str) -> str:
    schedules = parse_schedules(prompt)
    slots = find_available_slots(prompt, schedules)
    return select_and_format(slots, "earliest")
