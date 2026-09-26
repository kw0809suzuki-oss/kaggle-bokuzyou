"""Current-World Relationship Lens v0.

Purpose:
Externalize only the Current World provenance already carried by each
ShortPlanCandidate.source_paths.

This module does NOT:
- add/remove/reorder Candidates
- score/rank/value Candidates
- predict effects or terminal outcomes
- infer OPEN / ADVANCE / CLOSE semantics
- consume RNG
- mutate the StateSnapshot or Candidate

Raw resolved path/value provenance is the authoritative output.
A small derived signature is emitted only from the resolved Current State.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from plan_generator_entrance_v0 import StateSnapshot, ShortPlanCandidate


class LensIntegrityError(RuntimeError):
    pass


def _plain(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    raise TypeError(f"Unsupported value type: {type(v).__name__}")


def _path_to_list(path):
    return [copy.deepcopy(x) for x in path]


def _resolve_path(raw: Any, path) -> Any:
    v = raw
    traversed = []
    for key in path:
        traversed.append(copy.deepcopy(key))
        try:
            if isinstance(v, Mapping):
                v = v[key]
            elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                if not isinstance(key, int):
                    raise TypeError(
                        f"sequence path key must be int, got {type(key).__name__}"
                    )
                v = v[key]
            else:
                raise TypeError(
                    f"cannot traverse {type(v).__name__} at {traversed[:-1]}"
                )
        except Exception as exc:
            raise LensIntegrityError(
                f"source_path failed to resolve at {traversed}: {exc!r}"
            ) from exc
    return copy.deepcopy(v)


def _source_path_type(path) -> str:
    p = list(path)

    if len(p) >= 3 and p[0] == "private" and p[1] == "seeds":
        return "seed"
    if len(p) >= 3 and p[0] == "private" and p[1] == "inventories":
        return "inventory"
    if len(p) >= 3 and p[0] == "private" and p[1] == "shed":
        return "shed"
    if len(p) >= 3 and p[0] == "market" and p[1] == "prices":
        return "market_price"
    if (
        len(p) >= 4
        and p[0] == "farms"
        and p[2] == "tiles"
    ):
        return "tile"
    if (
        len(p) >= 3
        and p[0] == "farms"
        and p[2] == "money"
    ):
        return "cash"
    return "unknown"


def _normalized_fact(source_path_type: str, value: Any) -> str:
    if source_path_type == "seed":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > 0:
                return "seed_positive"
            if value == 0:
                return "seed_zero"
            return "seed_negative"
        return "seed_unknown_value"

    if source_path_type == "tile":
        if value is None:
            return "tile_empty"
        if value == "LOCKED":
            return "tile_locked"
        if isinstance(value, Mapping):
            kind = value.get("kind")
            if kind == "PLANT":
                return "tile_plant"
            if kind == "WEED":
                return "tile_weed"
            return "tile_other_mapping"
        return "tile_unknown_value"

    if source_path_type == "inventory":
        if isinstance(value, Mapping):
            positive = any(
                isinstance(q, (int, float))
                and not isinstance(q, bool)
                and q > 0
                for q in value.values()
            )
            return "inventory_positive" if positive else "inventory_nonpositive"
        return "inventory_unknown_value"

    if source_path_type == "shed":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "shed_positive" if value > 0 else "shed_nonpositive"
        return "shed_unknown_value"

    if source_path_type == "market_price":
        # The path resolved. This token states only present Current World data.
        return "market_price_present"

    if source_path_type == "cash":
        # Do not infer BUY/SELL direction or value. Only path provenance.
        return "cash_path_referenced"

    return "unknown_source_path_type"


def annotate_candidate(
    snapshot: StateSnapshot,
    candidate: ShortPlanCandidate,
) -> dict[str, Any]:
    if not isinstance(snapshot, StateSnapshot):
        raise TypeError("snapshot must be StateSnapshot")
    if not isinstance(candidate, ShortPlanCandidate):
        raise TypeError("candidate must be ShortPlanCandidate")

    raw = snapshot.raw()
    refs = []
    facts = []

    for source_path in candidate.source_paths:
        path = tuple(source_path)
        value = _resolve_path(raw, path)
        path_type = _source_path_type(path)
        fact = _normalized_fact(path_type, value)

        refs.append({
            "source_path_type": path_type,
            "source_path": _path_to_list(path),
            "resolution_status": "RESOLVED",
            "resolved_value": _plain(value),
            "value_presence": "ABSENT_VALUE" if value is None else "PRESENT_VALUE",
            "normalized_fact": fact,
        })
        facts.append(fact)

    # Derived only from Current State provenance above. No effect semantics.
    signature = tuple(sorted(facts))

    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "target": _plain(candidate.target),
        "authoritative_provenance": refs,
        "derived_relationship_signature": list(signature),
        "has_unknown_source_path_type": any(
            r["source_path_type"] == "unknown" for r in refs
        ),
    }


def annotate_candidates(
    snapshot: StateSnapshot,
    candidates: list[ShortPlanCandidate],
) -> list[dict[str, Any]]:
    return [annotate_candidate(snapshot, c) for c in candidates]
