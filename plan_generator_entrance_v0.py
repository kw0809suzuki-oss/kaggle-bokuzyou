"""Plan Generator entrance v0.

Purpose:
    Official Kaggriculture observation
    -> schema bind
    -> validate
    -> canonical StateSnapshot
    -> raw-backed accessor
    -> generate_plans(state)

This module deliberately does NOT derive economic meanings such as
productive_tiles, occupied_capacity, empty_capacity, or any other inferred
planning feature. Canonical means deterministic and typed enough to hand off,
not compressed or interpreted.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "official-state-snapshot-v0"
REQUIRED_TOP_LEVEL = ("player", "day", "hour", "farms", "private", "market", "town")


class StateSchemaError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    raise StateSchemaError(f"Unsupported Official State value type: {type(value).__name__}")


def validate_official_state(raw: Mapping[str, Any]) -> None:
    if not isinstance(raw, Mapping):
        raise StateSchemaError("Official State must be mapping-like")

    missing = [k for k in REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise StateSchemaError(f"Missing required Official State field(s): {missing}")

    player = raw["player"]
    if not isinstance(player, int) or isinstance(player, bool):
        raise StateSchemaError("player must be int")

    farms = raw["farms"]
    if not isinstance(farms, Sequence) or isinstance(farms, (str, bytes)):
        raise StateSchemaError("farms must be a sequence")
    if not farms:
        raise StateSchemaError("farms must not be empty")
    if player < 0 or player >= len(farms):
        raise StateSchemaError("player index outside farms")

    for k in ("day", "hour"):
        v = raw[k]
        if not isinstance(v, int) or isinstance(v, bool):
            raise StateSchemaError(f"{k} must be int")

    for k in ("private", "market", "town"):
        if not isinstance(raw[k], Mapping):
            raise StateSchemaError(f"{k} must be mapping-like")


@dataclass(frozen=True)
class StateSnapshot:
    """Canonical, lossless wrapper around one Official observation."""

    schema_version: str
    _raw: dict[str, Any]
    _canonical_json: str
    _canonical_hash: str

    @classmethod
    def bind(cls, official_state: Any) -> "StateSnapshot":
        raw = _plain(official_state)
        validate_official_state(raw)

        # Deep-copy once at the boundary. The snapshot owns this exact
        # lossless plain representation of the Official observation.
        owned = copy.deepcopy(raw)
        canonical_json = json.dumps(
            owned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        canonical_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return cls(
            schema_version=SCHEMA_VERSION,
            _raw=owned,
            _canonical_json=canonical_json,
            _canonical_hash=canonical_hash,
        )

    @property
    def canonical_hash(self) -> str:
        return self._canonical_hash

    def canonical_json(self) -> str:
        return self._canonical_json

    def raw(self) -> dict[str, Any]:
        """Return a defensive copy of the exact bound Official State."""
        return copy.deepcopy(self._raw)

    def get(self, *path: Any) -> Any:
        """Read an exact raw-backed value without adding derived meaning."""
        value: Any = self._raw
        for key in path:
            if isinstance(value, Mapping):
                value = value[key]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = value[key]
            else:
                raise KeyError(path)
        return copy.deepcopy(value)


def bind_official_state(official_state: Any) -> StateSnapshot:
    return StateSnapshot.bind(official_state)


def generate_plans(state: StateSnapshot) -> list[Any]:
    """Plan Generator entrance only.

    v0 intentionally returns no candidates. The connection is executable,
    but plan semantics are not invented at the State boundary.
    """
    if not isinstance(state, StateSnapshot):
        raise TypeError("generate_plans expects StateSnapshot")
    return []
