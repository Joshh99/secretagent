"""Maps DEAP integer vectors to/from secretagent dotlist config strings."""

from dataclasses import dataclass, field
from math import prod
from typing import Any


@dataclass
class SearchDimension:
    key: str
    values: list[Any] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.values)


def encode(dims: list[SearchDimension], config: dict[str, Any]) -> list[int]:
    """Map a {key: value} config dict to an integer vector.

    Each element is the index of the config value in the corresponding
    dimension's values list.

    Raises KeyError if a dimension key is missing from config.
    Raises ValueError if a config value is not in the dimension's values.
    """
    vec = []
    for dim in dims:
        val = config[dim.key]
        try:
            vec.append(dim.values.index(val))
        except ValueError:
            raise ValueError(
                f"{dim.key}={val!r} not in {dim.values}"
            )
    return vec


def decode(dims: list[SearchDimension], vec: list[int]) -> list[str]:
    """Map an integer vector to a list of dotlist strings.

    Returns e.g. ["llm.model=gpt-4o", "ptools.foo.method=simulate"].

    Raises IndexError if any index is out of bounds for its dimension.
    """
    if len(vec) != len(dims):
        raise ValueError(
            f"Vector length {len(vec)} != number of dimensions {len(dims)}"
        )
    parts = []
    for dim, idx in zip(dims, vec):
        if idx < 0 or idx >= dim.size:
            raise IndexError(
                f"{dim.key}: index {idx} out of range [0, {dim.size})"
            )
        parts.append(f"{dim.key}={dim.values[idx]}")
    return parts


def decode_dict(dims: list[SearchDimension], vec: list[int]) -> dict[str, Any]:
    """Map an integer vector to a {key: value} config dict."""
    if len(vec) != len(dims):
        raise ValueError(
            f"Vector length {len(vec)} != number of dimensions {len(dims)}"
        )
    result = {}
    for dim, idx in zip(dims, vec):
        if idx < 0 or idx >= dim.size:
            raise IndexError(
                f"{dim.key}: index {idx} out of range [0, {dim.size})"
            )
        result[dim.key] = dim.values[idx]
    return result


def space_size(dims: list[SearchDimension]) -> int:
    """Total number of configs in the search space (product of all dim sizes)."""
    if not dims:
        return 0
    return prod(dim.size for dim in dims)


def dim_sizes(dims: list[SearchDimension]) -> list[int]:
    """Return the number of valid values per dimension."""
    return [dim.size for dim in dims]
