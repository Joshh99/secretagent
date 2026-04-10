"""Task-specific interfaces for NaturalPlan trip planning.

Decomposition derived from LLM reasoning traces:
1. parse_trip_constraints — extract cities, durations, flights, time windows
2. find_valid_route — find city ordering respecting all constraints
3. build_trip_plan — assign day ranges and format itinerary
"""

from secretagent.core import interface


@interface
def parse_trip_constraints(prompt: str) -> str:
    """Extract all trip planning constraints from the problem.

    Return a structured summary with these keys:
    - total_days: total trip duration (int)
    - cities: dict mapping city name to required stay duration in days
        e.g. {"Helsinki": 5, "Barcelona": 3, "Florence": 5}
    - flights: adjacency list of direct flights
        e.g. {"Helsinki": ["Barcelona"], "Barcelona": ["Helsinki", "Florence"]}
    - time_windows: list of time window constraints, each with:
        city, earliest_day, latest_day
        e.g. [{"city": "Florence", "earliest_day": 9, "latest_day": 14}]

    >>> parse_trip_constraints("...visit 3 European cities for 13 days...")
    '{"total_days": 13, "cities": {"Helsinki": 5, "Barcelona": 3, "Florence": 5}, "flights": {"Helsinki": ["Barcelona"], "Barcelona": ["Helsinki", "Florence"], "Florence": ["Barcelona"]}, "time_windows": [{"city": "Florence", "earliest_day": 9, "latest_day": 14}]}'
    """


@interface
def find_valid_route(prompt: str, constraints: str) -> str:
    """Find a valid ordering of cities that respects all constraints.

    Check: direct flights exist between consecutive cities,
    total stay durations fit within total_days,
    time window constraints are satisfied.

    Return an ordered list of city names.

    >>> find_valid_route("...", '{"total_days": 13, "cities": {"Helsinki": 5, ...}, ...}')
    '["Helsinki", "Barcelona", "Florence"]'
    """


@interface
def build_trip_plan(prompt: str, constraints: str, route: str) -> str:
    """Build a day-by-day itinerary and format the answer.

    Assign day ranges to each city based on required durations.
    Include flight days between cities.

    Output format:
    'Here is the trip plan for visiting the {N} European cities for {total_days} days:'
    '**Day 1-5:** Visit Helsinki for 5 days.'
    '**Day 5:** Fly from Helsinki to Barcelona.'
    '**Day 5-7:** Visit Barcelona for 3 days.'
    ...

    >>> build_trip_plan("...", '{"total_days": 13, ...}', '["Helsinki", "Barcelona", "Florence"]')
    'Here is the trip plan for visiting the 3 European cities for 13 days:\\n**Day 1-5:** Visit Helsinki...'
    """


@interface
def trip_planning(prompt: str) -> str:
    """Solve a trip planning problem.
    Return a day-by-day itinerary with visit and flight lines.
    """
    ...


def trip_workflow(prompt: str) -> str:
    constraints = parse_trip_constraints(prompt)
    route = find_valid_route(prompt, constraints)
    return build_trip_plan(prompt, constraints, route)
