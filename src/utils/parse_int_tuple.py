"""Utility for parsing integer tuple config values."""

from __future__ import annotations

import ast


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse a config value that may be a scalar or a sequence of integers."""
    parsed_value = ast.literal_eval(value)
    if isinstance(parsed_value, int):
        return (parsed_value,)
    if isinstance(parsed_value, (list, tuple)) and all(
        isinstance(item, int) for item in parsed_value
    ):
        return tuple(parsed_value)
    raise ValueError("Expected an int, list[int], or tuple[int, ...].")
