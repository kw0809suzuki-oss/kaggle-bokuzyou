"""ShortPlan -> one-turn ActionBundle projector v2.

Supported ShortPlan kinds:
- deliver_carried_to_shed
- realize_shed_stock_sale
- establish_plant
- prepare_for_plant

This module executes an explicitly supplied ShortPlan. It does not rank or select
plans and does not attach economic value to them. Callers must re-run
generate_plans() on every Official PostState and project again from that State.
"""

from __future__ import annotations

import copy
from typing import Any

from plan_generator_entrance_v0 import StateSnapshot, ShortPlanCandidate, generate_plans


def _unit_positions(raw: dict[str, Any]) -> list[list[int]]:
    farm = raw["farms"][raw["player"]]
    return [list(farm["farmer"])] + [list(p) for p in (farm.get("hands", []) or [])]


def _unit_position(raw: dict[str, Any], unit_index: int) -> list[int]:
    return _unit_positions(raw)[unit_index]


def _shed_access_tiles(board_size: int) -> list[tuple[int, int]]:
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _nearest_access(pos: list[int], board_size: int) -> tuple[int, int]:
    x, y = pos
    access = _shed_access_tiles(board_size)
    return min(access, key=lambda p: (abs(p[0] - x) + abs(p[1] - y), access.index(p)))


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
    return ["PASS"]


def _empty_bundle(raw: dict[str, Any]) -> dict[str, Any]:
    farm = raw["farms"][raw["player"]]
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (farm.get("hands", []) or [])],
        "market": [],
    }


def _set_unit_action(bundle: dict[str, Any], unit_index: int, action: list[Any]) -> None:
    if unit_index == 0:
        bundle["farmer"] = action
    else:
        bundle["hands"][unit_index - 1] = action


def _nearest_unit(raw: dict[str, Any], tile: tuple[int, int]) -> int:
    tx, ty = tile
    positions = _unit_positions(raw)
    return min(
        range(len(positions)),
        key=lambda i: (abs(positions[i][0] - tx) + abs(positions[i][1] - ty), i),
    )


def _owned_crop_quantity(raw: dict[str, Any], crop: str) -> int:
    total = int((raw["private"].get("shed", {}) or {}).get(crop, 0) or 0)
    for inv in (raw["private"].get("inventories", []) or []):
        if isinstance(inv, dict):
            total += int(inv.get(crop, 0) or 0)
    return total


def project_short_plan(state: StateSnapshot, plan: ShortPlanCandidate) -> dict[str, Any]:
    if not isinstance(state, StateSnapshot):
        raise TypeError("state must be StateSnapshot")
    if not isinstance(plan, ShortPlanCandidate):
        raise TypeError("plan must be ShortPlanCandidate")

    raw = state.raw()
    bundle = _empty_bundle(raw)

    if plan.kind == "deliver_carried_to_shed":
        unit_index = int(plan.target["unit_index"])
        positions = _unit_positions(raw)
        if unit_index < 0 or unit_index >= len(positions):
            raise ValueError("target unit no longer exists")
        board_size = len(raw["farms"][raw["player"]].get("tiles", []) or [])
        pos = positions[unit_index]
        access = _nearest_access(pos, board_size)
        action = ["DROP"] if tuple(pos) == access else _move_toward(pos, access)
        _set_unit_action(bundle, unit_index, action)
        return bundle

    if plan.kind == "realize_shed_stock_sale":
        item = str(plan.target["item"])
        qty = int(plan.target["available_quantity"])
        if qty <= 0:
            raise ValueError("sale quantity must be positive")
        bundle["market"] = [["SELL", item, qty]]
        return bundle

    if plan.kind == "establish_plant":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        unit_index = _nearest_unit(raw, (x, y))
        pos = _unit_position(raw, unit_index)
        action = ["PLANT", crop] if tuple(pos) == (x, y) else _move_toward(pos, (x, y))
        _set_unit_action(bundle, unit_index, action)
        return bundle

    if plan.kind == "prepare_for_plant":
        crop = str(plan.target["crop"])
        missing = int(plan.target["missing_seed_quantity"])
        if missing <= 0:
            raise ValueError("prepare_for_plant has no missing seed")
        bundle["market"] = [["BUY_SEED", crop, missing]]
        return bundle

    if plan.kind == "maintain_plant_today":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        unit_index = _nearest_unit(raw, (x, y))
        pos = _unit_position(raw, unit_index)
        action = ["WATER"] if tuple(pos) == (x, y) else _move_toward(pos, (x, y))
        _set_unit_action(bundle, unit_index, action)
        return bundle

    if plan.kind == "collect_plant_output":
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        unit_index = _nearest_unit(raw, (x, y))
        pos = _unit_position(raw, unit_index)
        action = ["HARVEST"] if tuple(pos) == (x, y) else _move_toward(pos, (x, y))
        _set_unit_action(bundle, unit_index, action)
        return bundle

    if plan.kind == "prepare_surface_for_plant":
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        unit_index = _nearest_unit(raw, (x, y))
        pos = _unit_position(raw, unit_index)
        action = ["DIG"] if tuple(pos) == (x, y) else _move_toward(pos, (x, y))
        _set_unit_action(bundle, unit_index, action)
        return bundle

    raise NotImplementedError(plan.kind)


def baseline_pass_bundle(state: StateSnapshot) -> dict[str, Any]:
    return _empty_bundle(state.raw())


def semantic_plan_match(
    plan: ShortPlanCandidate,
    *,
    kind: str,
    target: dict[str, Any],
) -> bool:
    if plan.kind != kind:
        return False
    if kind == "deliver_carried_to_shed":
        return (
            int(plan.target.get("unit_index", -1)) == int(target["unit_index"])
            and dict(plan.target.get("carried_items", {})) == dict(target["carried_items"])
        )
    if kind == "realize_shed_stock_sale":
        return (
            str(plan.target.get("item")) == str(target["item"])
            and int(plan.target.get("available_quantity", -1)) == int(target["available_quantity"])
        )
    if kind == "establish_plant":
        return (
            str(plan.target.get("crop")) == str(target["crop"])
            and list(plan.target.get("tile", [])) == list(target["tile"])
        )
    if kind == "prepare_for_plant":
        return (
            str(plan.target.get("crop")) == str(target["crop"])
            and list(plan.target.get("tile", [])) == list(target["tile"])
            and int(plan.target.get("missing_seed_quantity", -1)) == int(target["missing_seed_quantity"])
        )
    if kind == "maintain_plant_today":
        return (
            str(plan.target.get("crop")) == str(target["crop"])
            and list(plan.target.get("tile", [])) == list(target["tile"])
            and int(plan.target.get("planted_day", -1)) == int(target["planted_day"])
        )
    if kind == "collect_plant_output":
        return (
            str(plan.target.get("crop")) == str(target["crop"])
            and list(plan.target.get("tile", [])) == list(target["tile"])
            and int(plan.target.get("planted_day", -1)) == int(target["planted_day"])
        )
    if kind == "prepare_surface_for_plant":
        return (
            str(plan.target.get("crop")) == str(target["crop"])
            and list(plan.target.get("tile", [])) == list(target["tile"])
        )
    return False


def semantic_delivery_match(plan: ShortPlanCandidate, unit_index: int, carried_items: dict[str, Any]) -> bool:
    return semantic_plan_match(
        plan,
        kind="deliver_carried_to_shed",
        target={"unit_index": unit_index, "carried_items": carried_items},
    )


def _establish_present(post: StateSnapshot, crop: str, tile: list[int]) -> bool:
    for p in generate_plans(post):
        if (
            p.kind == "establish_plant"
            and str(p.target.get("crop")) == crop
            and list(p.target.get("tile", [])) == list(tile)
        ):
            return True
    return False


def completion_from_states(
    plan: ShortPlanCandidate,
    pre: StateSnapshot,
    post: StateSnapshot,
) -> dict[str, Any]:
    pre_raw = pre.raw()
    post_raw = post.raw()

    if plan.kind == "deliver_carried_to_shed":
        unit_index = int(plan.target["unit_index"])
        target_items = dict(plan.target["carried_items"])
        pre_inv = pre_raw["private"]["inventories"][unit_index]
        post_inv = post_raw["private"]["inventories"][unit_index]
        pre_shed = pre_raw["private"].get("shed", {}) or {}
        post_shed = post_raw["private"].get("shed", {}) or {}

        rows = {}
        all_left = True
        all_arrived = True
        for item, target_qty in target_items.items():
            pi = pre_inv.get(item, 0)
            qi = post_inv.get(item, 0)
            ps = pre_shed.get(item, 0)
            qs = post_shed.get(item, 0)
            gain = qs - ps
            rows[item] = {
                "target_quantity": target_qty,
                "pre_unit": pi,
                "post_unit": qi,
                "pre_shed": ps,
                "post_shed": qs,
                "shed_gain": gain,
            }
            all_left = all_left and qi == 0
            all_arrived = all_arrived and gain >= target_qty

        complete = all_left and all_arrived
        return {
            "complete": complete,
            "status": "complete" if complete else ("in_progress" if not all_left else "left_unit_without_full_shed_transfer"),
            "items": rows,
        }

    if plan.kind == "realize_shed_stock_sale":
        item = str(plan.target["item"])
        qty = int(plan.target["available_quantity"])
        p = pre_raw["player"]
        pre_shed = pre_raw["private"].get("shed", {}).get(item, 0)
        post_shed = post_raw["private"].get("shed", {}).get(item, 0)
        pre_cash = pre_raw["farms"][p].get("money", 0)
        post_cash = post_raw["farms"][p].get("money", 0)
        sold = pre_shed - post_shed
        cash_gain = post_cash - pre_cash
        complete = sold >= qty and cash_gain > 0
        return {
            "complete": complete,
            "status": "complete" if complete else "incomplete",
            "item": item,
            "target_quantity": qty,
            "pre_shed": pre_shed,
            "post_shed": post_shed,
            "sold_quantity": sold,
            "pre_cash": pre_cash,
            "post_cash": post_cash,
            "cash_gain": cash_gain,
        }

    if plan.kind == "establish_plant":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        p = pre_raw["player"]
        post_tile = post_raw["farms"][p]["tiles"][y][x]
        complete = (
            isinstance(post_tile, dict)
            and post_tile.get("kind") == "PLANT"
            and post_tile.get("crop") == crop
        )
        return {
            "complete": complete,
            "status": "complete" if complete else "in_progress",
            "crop": crop,
            "tile": [x, y],
            "post_tile": copy.deepcopy(post_tile),
            "pre_seed_quantity": pre_raw["private"].get("seeds", {}).get(crop, 0),
            "post_seed_quantity": post_raw["private"].get("seeds", {}).get(crop, 0),
        }

    if plan.kind == "prepare_for_plant":
        crop = str(plan.target["crop"])
        tile = [int(plan.target["tile"][0]), int(plan.target["tile"][1])]
        p = pre_raw["player"]
        pre_seed = pre_raw["private"].get("seeds", {}).get(crop, 0)
        post_seed = post_raw["private"].get("seeds", {}).get(crop, 0)
        pre_cash = pre_raw["farms"][p].get("money", 0)
        post_cash = post_raw["farms"][p].get("money", 0)
        establish_present = _establish_present(post, crop, tile)
        complete = bool(establish_present)
        return {
            "complete": complete,
            "status": "complete" if complete else "incomplete",
            "crop": crop,
            "tile": tile,
            "missing_seed_quantity": int(plan.target["missing_seed_quantity"]),
            "pre_seed_quantity": pre_seed,
            "post_seed_quantity": post_seed,
            "seed_gain": post_seed - pre_seed,
            "pre_cash": pre_cash,
            "post_cash": post_cash,
            "cash_delta": post_cash - pre_cash,
            "target_establish_plan_present": establish_present,
        }

    if plan.kind == "maintain_plant_today":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        p = pre_raw["player"]
        pre_tile = pre_raw["farms"][p]["tiles"][y][x]
        post_tile = post_raw["farms"][p]["tiles"][y][x]
        same_plant = (
            isinstance(post_tile, dict)
            and post_tile.get("kind") == "PLANT"
            and post_tile.get("crop") == crop
            and int(post_tile.get("planted_day", -1)) == int(plan.target["planted_day"])
        )
        watered = bool(post_tile.get("watered_today", False)) if isinstance(post_tile, dict) else False
        complete = same_plant and watered
        return {
            "complete": complete,
            "status": "complete" if complete else "in_progress",
            "crop": crop,
            "tile": [x, y],
            "pre_tile": copy.deepcopy(pre_tile),
            "post_tile": copy.deepcopy(post_tile),
            "post_watered_today": watered,
            "post_consecutive_unwatered": (
                post_tile.get("consecutive_unwatered") if isinstance(post_tile, dict) else None
            ),
        }

    if plan.kind == "collect_plant_output":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        p = pre_raw["player"]
        pre_tile = pre_raw["farms"][p]["tiles"][y][x]
        post_tile = post_raw["farms"][p]["tiles"][y][x]
        pre_yield = int(pre_tile.get("yield_units", 0) or 0) if isinstance(pre_tile, dict) else 0
        post_yield = int(post_tile.get("yield_units", 0) or 0) if isinstance(post_tile, dict) else 0
        pre_owned = _owned_crop_quantity(pre_raw, crop)
        post_owned = _owned_crop_quantity(post_raw, crop)
        output_gain = post_owned - pre_owned
        yield_removed = max(0, pre_yield - post_yield)
        complete = pre_yield > 0 and yield_removed > 0 and output_gain >= yield_removed
        return {
            "complete": complete,
            "status": "complete" if complete else "in_progress",
            "crop": crop,
            "tile": [x, y],
            "pre_tile": copy.deepcopy(pre_tile),
            "post_tile": copy.deepcopy(post_tile),
            "pre_yield_units": pre_yield,
            "post_yield_units": post_yield,
            "yield_removed": yield_removed,
            "pre_owned_quantity": pre_owned,
            "post_owned_quantity": post_owned,
            "owned_output_gain": output_gain,
        }


    if plan.kind == "prepare_surface_for_plant":
        crop = str(plan.target["crop"])
        x, y = (int(plan.target["tile"][0]), int(plan.target["tile"][1]))
        p = pre_raw["player"]
        pre_tile = pre_raw["farms"][p]["tiles"][y][x]
        post_tile = post_raw["farms"][p]["tiles"][y][x]
        exact_establish_present = any(
            q.kind == "establish_plant"
            and str(q.target.get("crop")) == crop
            and list(q.target.get("tile", [])) == [x, y]
            for q in generate_plans(post)
        )
        complete = post_tile is None and exact_establish_present
        return {
            "complete": complete,
            "status": "complete" if complete else "in_progress",
            "crop": crop,
            "tile": [x, y],
            "pre_tile": copy.deepcopy(pre_tile),
            "post_tile": copy.deepcopy(post_tile),
            "target_establish_plan_present": exact_establish_present,
        }

    raise NotImplementedError(plan.kind)
