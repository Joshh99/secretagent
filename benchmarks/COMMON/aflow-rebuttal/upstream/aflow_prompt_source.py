"""Repair one formatting error in Python prompts proposed by AFlow's optimizer.

Some optimizer responses put the two characters ``\\n`` between Python
statements instead of an actual newline. Leave escapes inside strings alone.
Only return a repair when the original source fails to parse and the repaired
source parses; every other proposal is returned unchanged.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize


def repair_escaped_linebreaks(source: str) -> tuple[str, int]:
    """Return source with invalid out-of-string ``\\n`` changed to newlines.

    The count is zero when no safe, syntax-valid repair was found. This changes
    only source formatting, never the contents of a Python string or comment.
    """
    try:
        ast.parse(source)
        return source, 0
    except SyntaxError:
        pass

    line_starts = [0]
    for line in source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))

    protected: list[tuple[int, int]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.STRING, tokenize.COMMENT):
                start = line_starts[token.start[0] - 1] + token.start[1]
                end = line_starts[token.end[0] - 1] + token.end[1]
                protected.append((start, end))
    except (tokenize.TokenError, IndentationError, IndexError):
        return source, 0

    positions = [
        match.start()
        for match in re.finditer(r"\\n", source)
        if not any(start <= match.start() < end for start, end in protected)
    ]
    if not positions:
        return source, 0

    parts = []
    cursor = 0
    for position in positions:
        parts.extend((source[cursor:position], "\n"))
        cursor = position + 2
    parts.append(source[cursor:])
    repaired = "".join(parts)

    try:
        ast.parse(repaired)
    except SyntaxError:
        return source, 0
    return repaired, len(positions)
