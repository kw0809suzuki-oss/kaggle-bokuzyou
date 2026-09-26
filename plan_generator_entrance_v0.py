"""Plan Generator entrance + minimal ShortPlan candidate generation v0.

Flow:
    Official Kaggriculture observation
    -> schema bind
    -> validate
    -> canonical StateSnapshot
    -> raw-backed accessor
    -> generate_plans(state)

Design boundary:
- candidates are short jobs, not Action rankings or fixed strategies
- every candidate is grounded in raw Official State facts
- no terminal value, score, preference, or selection is assigned
- no ActionBundle is generated yet
- no economic derived fields such as productive_tiles / occupied_capacity /
  empty_capacity are invented here

Canonical means deterministic and raw-backed, not compressed or interpreted.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "official-state-snapshot-v0"
PLAN_SCHEMA_VERSION = "short-plan-candidate-v0"
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
        return copy.deepcopy(self._raw)

    def get(self, *path: Any) -> Any:
        value: Any = self._raw
        for key in path:
            if isinstance(value, Mapping):
                value = value[key]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                value = value[key]
            else:
                raise KeyError(path)
        return copy.deepcopy(value)


@dataclass(frozen=True)
class ShortPlanCandidate:
    """A raw-grounded short job. It is not an ActionBundle or a preference."""

    plan_schema_version: str
    candidate_id: str
    kind: str
    target: dict[str, Any]
    completion: dict[str, Any]
    requirements: tuple[dict[str, Any], ...]
    unknowns: tuple[str, ...]
    source_paths: tuple[tuple[Any, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_schema_version": self.plan_schema_version,
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "target": copy.deepcopy(self.target),
            "completion": copy.deepcopy(self.completion),
            "requirements": [copy.deepcopy(x) for x in self.requirements],
            "unknowns": list(self.unknowns),
            "source_paths": [list(x) for x in self.source_paths],
        }


def bind_official_state(official_state: Any) -> StateSnapshot:
    return StateSnapshot.bind(official_state)


def _candidate_id(state: StateSnapshot, kind: str, target: Mapping[str, Any]) -> str:
    payload = {
        "state": state.canonical_hash,
        "kind": kind,
        "target": _plain(target),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"{kind}:{digest}"


def _first_carried_inventory(raw: Mapping[str, Any]):
    inventories = raw["private"].get("inventories", []) or []
    for unit_index, inv in enumerate(inventories):
        if not isinstance(inv, Mapping):
            continue
        positive = {str(item): qty for item, qty in inv.items() if isinstance(qty, (int, float)) and qty > 0}
        if positive:
            return unit_index, dict(sorted(positive.items()))
    return None


def _first_sellable_shed_item(raw: Mapping[str, Any]):
    shed = raw["private"].get("shed", {}) or {}
    prices = raw["market"].get("prices", {}) or {}
    # Raw market price keys define the market-visible product names in this State.
    for item in sorted(prices):
        qty = shed.get(item, 0)
        if isinstance(qty, (int, float)) and qty > 0:
            return item, qty
    return None


def _first_seed_and_raw_empty_tile(raw: Mapping[str, Any]):
    player = raw["player"]
    tiles = raw["farms"][player].get("tiles", []) or []
    target_tile = None
    for y, row in enumerate(tiles):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        for x, tile in enumerate(row):
            # This is a direct raw predicate, not an empty_capacity derivation.
            if tile is None:
                target_tile = (x, y)
                break
        if target_tile is not None:
            break

    if target_tile is None:
        return None

    seeds = raw["private"].get("seeds", {}) or {}
    for crop in sorted(seeds):
        qty = seeds.get(crop, 0)
        if isinstance(qty, (int, float)) and qty > 0:
            return crop, qty, target_tile
    return None


def generate_plans(state: StateSnapshot) -> list[ShortPlanCandidate]:
    """Generate at most three raw-grounded short jobs.

    Stable enumeration order is only for reproducibility; it is NOT a ranking:
      1. carried inventory -> shed-available state
      2. shed stock -> sale realization
      3. seed + one raw None tile -> planted state

    The generator does not choose a plan, score a plan, or project actions.
    """
    if not isinstance(state, StateSnapshot):
        raise TypeError("generate_plans expects StateSnapshot")

    raw = state.raw()
    player = raw["player"]
    plans: list[ShortPlanCandidate] = []

    carried = _first_carried_inventory(raw)
    if carried is not None:
        unit_index, items = carried
        target = {"unit_index": unit_index, "carried_items": items}
        plans.append(ShortPlanCandidate(
            plan_schema_version=PLAN_SCHEMA_VERSION,
            candidate_id=_candidate_id(state, "deliver_carried_to_shed", target),
            kind="deliver_carried_to_shed",
            target=target,
            completion={
                "observable": "the targeted unit no longer carries the targeted items and Official World shows any realized shed transfer",
            },
            requirements=(
                {"raw_fact": "private.inventories[unit_index] contains positive quantity", "unit_index": unit_index},
                {"official_condition": "the unit must reach a shed-access position before DROP can realize"},
            ),
            unknowns=(
                "Action sequence and unit routing are not generated yet.",
                "Realized transfer remains an Official World result; shed-capacity configuration is not derived here.",
            ),
            source_paths=(("private", "inventories", unit_index),),
        ))

    sellable = _first_sellable_shed_item(raw)
    if sellable is not None:
        item, qty = sellable
        pre_money = raw["farms"][player].get("money", 0)
        target = {"item": item, "available_quantity": qty}
        plans.append(ShortPlanCandidate(
            plan_schema_version=PLAN_SCHEMA_VERSION,
            candidate_id=_candidate_id(state, "realize_shed_stock_sale", target),
            kind="realize_shed_stock_sale",
            target=target,
            completion={
                "observable": "Official World shows the targeted shed quantity decrease and self cash increase",
                "pre_cash": pre_money,
                "pre_shed_quantity": qty,
            },
            requirements=(
                {"raw_fact": "private.shed[item] > 0", "item": item, "quantity": qty},
                {"raw_fact": "market.prices exposes the same item", "item": item},
            ),
            unknowns=(
                "Exact realized sale price is left to the shared Official World.",
                "Opponent market actions are not predicted by this generator.",
            ),
            source_paths=(
                ("private", "shed", item),
                ("market", "prices", item),
                ("farms", player, "money"),
            ),
        ))

    plantable = _first_seed_and_raw_empty_tile(raw)
    if plantable is not None:
        crop, seed_qty, (x, y) = plantable
        target = {"crop": crop, "tile": [x, y]}
        plans.append(ShortPlanCandidate(
            plan_schema_version=PLAN_SCHEMA_VERSION,
            candidate_id=_candidate_id(state, "establish_plant", target),
            kind="establish_plant",
            target=target,
            completion={
                "observable": "Official World shows the target raw tile as a PLANT of the target crop",
            },
            requirements=(
                {"raw_fact": "private.seeds[crop] > 0", "crop": crop, "quantity": seed_qty},
                {"raw_fact": "farms[player].tiles[y][x] is None", "tile": [x, y]},
                {"official_condition": "a farmer/hand must be assigned to reach the target tile and PLANT"},
            ),
            unknowns=(
                "Action sequence and unit assignment are not generated yet.",
                "Plant establishment is only this Plan's completion; economic recovery is not claimed.",
            ),
            source_paths=(
                ("private", "seeds", crop),
                ("farms", player, "tiles", y, x),
            ),
        ))

    return plans[:3]
