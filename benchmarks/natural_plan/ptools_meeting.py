"""Task-specific interfaces for NaturalPlan meeting planning.

Decomposition derived from LLM reasoning traces:
1. parse_meeting_info — extract locations, friends, travel times, availability
2. plan_visit_order — determine optimal visit order to maximize meetings
3. build_meeting_plan — simulate schedule step-by-step and format answer
"""

from secretagent.core import interface


@interface
def parse_meeting_info(prompt: str) -> str:
    """Extract all meeting planning information from the problem.

    Return a structured summary with these keys:
    - my_location: starting location name
    - my_start_time: start time string e.g. "9:00 AM"
    - friends: list of dicts, each with:
        name, location, available_from, available_to, duration_minutes
    - travel_times: dict mapping "LocationA->LocationB" to minutes (int)

    >>> parse_meeting_info("...starting at Alamo Square at 9:00 AM...")
    '{"my_location": "Alamo Square", "my_start_time": "9:00 AM", "friends": [{"name": "James", "location": "Nob Hill", "available_from": "9:00 AM", "available_to": "5:30 PM", "duration_minutes": 30}], "travel_times": {"Alamo Square->Nob Hill": 11}}'
    """


@interface
def plan_visit_order(prompt: str, info: str) -> str:
    """Determine the optimal order to visit friends to maximize meetings.

    Consider travel times between locations, each friend's availability
    window, and required meeting durations. Use greedy or exhaustive
    search to find the ordering that allows the most meetings.

    Return an ordered list of friend names.

    >>> plan_visit_order("...", '{"friends": [...], ...}')
    '["James", "Nancy", "William", "Margaret", "Laura", "Sandra", "David"]'
    """


@interface
def build_meeting_plan(prompt: str, info: str, order: str) -> str:
    """Build a step-by-step meeting schedule and format the answer.

    Simulate the schedule: for each friend in order, compute travel time,
    wait if needed, meet for the required duration. Skip friends who
    cannot be met within their availability window.

    Output MUST start with 'SOLUTION:' followed by steps like:
    'You start at {location} at {time}.'
    'You travel to {location}. It takes {N} minutes. You arrive at {time}.'
    'You wait until {time}.'
    'You meet {name} for {N} minutes from {time} to {time}.'

    >>> build_meeting_plan("...", '{"my_location": "Alamo Square", ...}', '["James", "Nancy"]')
    'SOLUTION: You start at Alamo Square at 9:00 AM. You travel to Nob Hill. It takes 11 minutes. You arrive at 9:11 AM. You meet James for 30 minutes from 9:11 AM to 9:41 AM. ...'
    """


@interface
def meeting_planning(prompt: str) -> str:
    """Solve a meeting planning problem.
    Return a step-by-step schedule starting with 'SOLUTION:'.
    """
    ...


def meeting_workflow(prompt: str) -> str:
    info = parse_meeting_info(prompt)
    order = plan_visit_order(prompt, info)
    return build_meeting_plan(prompt, info, order)
