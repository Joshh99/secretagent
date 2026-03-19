"""Interfaces for MUSR object placement (theory of mind).

Migrated from AgentProject v2 ptools. The task: "Where would [person]
look for [object]?" This is a false-belief task — the answer depends
on what the person SAW, not where the object actually is.

Design principle: every ptool gets the raw narrative to avoid
information bottlenecks from lossy extraction.
"""

from secretagent.core import interface
from ptools_common import raw_answer, extract_index


@interface
def extract_movements(narrative: str) -> str:
    """Extract all people, objects, object movements, and incidental discoveries.

    Read the narrative carefully and extract:

    1. people: list of all people/characters mentioned
    2. objects: each trackable object and its INITIAL location
    3. movements: chronological list of every time an object changes location
    4. discoveries: times someone SEES or ENCOUNTERS an object at a location
       WITHOUT having witnessed the move that put it there

    For each MOVEMENT, extract:
    - step: sequential number (1, 2, 3, ...)
    - object: which object was moved
    - from: location it was moved FROM
    - to: location it was moved TO
    - mover: who moved it
    - description: brief description of what happened
    - present: list of people who WITNESSED this movement
    - absent: list of people who did NOT witness this movement
    - awareness: brief explanation of WHY each person is present or absent

    For each DISCOVERY, extract:
    - person: who discovered/noticed the object
    - object: which object they saw
    - location: where they saw it
    - after_step: which movement step this occurred after
    - description: what happened

    CRITICAL — what counts as a DISCOVERY:
    - Someone goes to a location and SEES/NOTICES/GLIMPSES an object there
    - Someone is TOLD where an object is by another person
    - Someone encounters an object while doing something else (e.g. tidying,
      passing by, reaching for something nearby)
    - These are DIFFERENT from movements — the person didn't move the object,
      they just became aware of its location

    CRITICAL RULES for determining presence/absence during MOVEMENTS:
    - A person who LEFT the area before a move is ABSENT
    - A person who is "engrossed", "distracted", "in another room",
      "in a sound booth", "couldn't see", "view blocked" is ABSENT
    - A person nearby, watching, or with line of sight is PRESENT
    - The MOVER is always PRESENT
    - "sees" or "notices" a move → PRESENT
    - "view blocked" or "out of visual range" → ABSENT
    - When in doubt, look for narrative cues about attention and line of sight

    IMPORTANT: Track ALL objects mentioned in the narrative, not just the ones
    that move the most.
    """


@interface
def infer_belief(narrative: str, movements: str, question: str, choices: list) -> str:
    """Determine where the target person believes the target object is located.

    You receive:
    1. The FULL original narrative (re-read it for details extraction may have missed)
    2. Extracted object movements with presence/absence info
    3. Incidental discoveries (someone saw/noticed an object without witnessing the move)
    4. The question identifying the target person and object, and answer choices

    Your task — determine the person's BELIEF about the object's location:

    Step 1: Start with the object's initial location (everyone knows this)
    Step 2: For each movement of the target_object (in chronological order):
       - If target_person is in "present" → they know the new location
       - If target_person is in "absent" → they still believe the old location
    Step 3: Check discoveries — if target_person discovered the target_object
       at a location, that UPDATES their belief to that location
    Step 4: Re-read the narrative to check for anything the extraction missed:
       - Did someone TELL the target person where the object is?
       - Did the target person GO TO the object's location and see it there?
       - Did the target person interact with something NEAR the object?
    Step 5: Match the believed location to one of the answer choices

    IMPORTANT: The question asks where the person would LOOK, which means
    where they BELIEVE the object is — not necessarily where it actually is.
    """


@interface
def answer_question(narrative: str, question: str, choices: list) -> int:
    """Read the narrative and answer where someone would look for an object.

    This is a theory-of-mind task: the answer is based on what the person
    believes, not the object's actual location.
    Return the 0-based index of the correct choice.
    """
    text = raw_answer(narrative, question, choices)
    return extract_index(text, choices)


@interface
def answer_question_workflow(narrative: str, question: str, choices: list) -> int:
    """Solve by extracting movements, inferring belief, then matching."""
    movements = extract_movements(narrative)
    text = infer_belief(narrative, movements, question, choices)
    return extract_index(text, choices)
