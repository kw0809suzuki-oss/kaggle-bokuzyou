"""Explicit ShortPlan -> ActionBundle projector v0.

Only deliver_carried_to_shed is supported.
No ranking, selection, or value judgment lives here.
The caller must explicitly provide the ShortPlanCandidate.
"""

from __future__ import annotations

import copy
from typing import Any

from plan_generator_entrance_v0 import StateSnapshot, ShortPlanCandidate


def _unit_position(raw: dict[str, Any], unit_index: int) -> list[int]:
    p = raw["player"]
    farm = raw["farms"][p]
    if unit_index == 0:
        return list(farm["farmer"])
    return list(farm["hands"][unit_index - 1])


def _shed_access_tiles(board_size: int) -> list[tuple[int, int]]:
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _nearest_access(pos: list[int], board_size: int) -> tuple[int, int]:
    x, y = pos
    access = _shed_access_tiles(board_size)
    # Stable geometry tie-break only; this is not Plan ranking.
    return min(access, key=lambda p: (abs(p[0]-x)+abs(p[1]-y), access.index(p)))


def _move_toward(pos: list[int], target: tuple[int, int]) -> list[str]:
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["DROP"]


def project_short_plan(state: StateSnapshot, plan: ShortPlanCandidate) -> dict[str, Any]:
    """Project one explicit deliver ShortPlan into one-turn whole ActionBundle.

    Every untargeted unit PASSes and market is empty. The plan is re-projected
    from each new StateSnapshot by the caller.
    """
    if not isinstance(state, StateSnapshot):
        raise TypeError("state must be StateSnapshot")
    if not isinstance(plan, ShortPlanCandidate):
        raise TypeError("plan must be ShortPlanCandidate")
    if plan.kind != "deliver_carried_to_shed":
        raise NotImplementedError(plan.kind)

    raw = state.raw()
    p = raw["player"]
    farm = raw["farms"][p]
    unit_index = int(plan.target["unit_index"])
    units = 1 + len(farm.get("hands", []) or [])
    if unit_index < 0 or unit_index >= units:
        raise ValueError("target unit no longer exists")

    farmer = ["PASS"]
    hands = [["PASS"] for _ in (farm.get("hands", []) or [])]

    pos = _unit_position(raw, unit_index)
    board_size = len(farm.get("tiles", []) or [])
    target = _nearest_access(pos, board_size)
    action = _move_toward(pos, target)

    if unit_index == 0:
        farmer = action
    else:
        hands[unit_index - 1] = action

    return {
        "farmer": farmer,
        "hands": hands,
        "market": [],
    }


def baseline_pass_bundle(state: StateSnapshot) -> dict[str, Any]:
    raw = state.raw()
    p = raw["player"]
    farm = raw["farms"][p]
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (farm.get("hands", []) or [])],
        "market": [],
    }


def semantic_delivery_match(plan: ShortPlanCandidate, unit_index: int, carried_items: dict[str, Any]) -> bool:
    return (
        plan.kind == "deliver_carried_to_shed"
        and int(plan.target.get("unit_index", -1)) == int(unit_index)
        and dict(plan.target.get("carried_items", {})) == dict(carried_items)
    )


def completion_from_states(
    plan: ShortPlanCandidate,
    pre: StateSnapshot,
    post: StateSnapshot,
) -> dict[str, Any]:
    """Observe completion from Official PostState, not from submitted Action."""
    unit_index = int(plan.target["unit_index"])
    target_items = dict(plan.target["carried_items"])
    pre_raw = pre.raw()
    post_raw = post.raw()

    pre_inv = pre_raw["private"]["inventories"][unit_index]
    post_inv = post_raw["private"]["inventories"][unit_index]
    pre_shed = pre_raw["private"].get("shed", {}) or {}
    post_shed = post_raw["private"].get("shed", {}) or {}

    item_rows = {}
    all_left_unit = True
    all_arrived_shed = True
    for item, target_qty in target_items.items():
        pi = pre_inv.get(item, 0)
        qi = post_inv.get(item, 0)
        ps = pre_shed.get(item, 0)
        qs = post_shed.get(item, 0)
        left = max(0, pi - qi)
        shed_gain = qs - ps
        item_rows[item] = {
            "target_quantity": target_qty,
            "pre_unit": pi,
            "post_unit": qi,
            "left_unit": left,
            "pre_shed": ps,
            "post_shed": qs,
            "shed_gain": shed_gain,
        }
        all_left_unit = all_left_unit and qi == 0
        all_arrived_shed = all_arrived_shed and shed_gain >= target_qty

    complete = all_left_unit and all_arrived_shed
    status = "complete" if complete else ("in_progress" if not all_left_unit else "left_unit_without_full_shed_transfer")
    return {
        "complete": complete,
        "status": status,
        "items": item_rows,
    }
