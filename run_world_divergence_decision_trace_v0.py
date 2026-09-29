#!/usr/bin/env python3
from __future__ import annotations

import copy
from typing import Any


def first_diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    """Return leaf paths whose values differ between two JSON-like values."""
    if type(left) is not type(right):
        return [path or "$"]

    if isinstance(left, dict):
        out: list[str] = []
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                out.append(child)
            else:
                out.extend(first_diff_paths(left[key], right[key], child))
        return out

    if isinstance(left, list):
        out: list[str] = []
        limit = max(len(left), len(right))
        for idx in range(limit):
            child = f"{path}[{idx}]"
            if idx >= len(left) or idx >= len(right):
                out.append(child)
            else:
                out.extend(first_diff_paths(left[idx], right[idx], child))
        return out

    return [] if left == right else [path or "$"]


def candidate_identity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip score/rank fields so candidate structure can be compared directly."""
    ignored = {"rank", "central_terminal_cash", "strict_terminal_cash"}
    return [
        {key: copy.deepcopy(value) for key, value in row.items() if key not in ignored}
        for row in rows
    ]
