"""Strong Model v0 — state-driven whole-farm operator.

Implementation target: Strong_Model_v0.md.
This lives in its own folder/branch and does not modify existing agents.

The agent replans from the Official observation every turn.  It:
- keeps urgent maintenance and cash-return work alive,
- assigns different workers to compatible jobs in the same turn,
- avoids double-allocating workers/targets/seeds,
- uses official rule tables for costs, maturity and market pricing,
- stops new commitments when they cannot reasonably return before terminal,
- emits only one turn of actions and observes again next turn.

The implementation is deliberately v0: estimates are conservative heuristics,
not claims about the hidden opponent policy or future market state.
"""
from __future__ import annotations

import math
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

PRODUCTS = tuple(rules.PRODUCTS)
CROPS = tuple(rules.CROPS)
ANIMALS = tuple(rules.ANIMALS)

_LAST_STEP: dict[int, int] = {}


def reset_agent() -> None:
    _LAST_STEP.clear()


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _cfg(configuration: Any, key: str, default: int) -> int:
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return int(configuration.get(key, default))
    return int(getattr(configuration, key, default))


def _distance(a: list[int] | tuple[int, int], b: list[int] | tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _shed_access(board_size: int) -> list[tuple[int, int]]:
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _nearest_shed(pos: list[int], board_size: int) -> tuple[int, int]:
    access = _shed_access(board_size)
    return min(access, key=lambda p: (_distance(pos, p), access.index(p)))


def _move_toward(pos: list[int], target: tuple[int, int]) -> list[str]:
    x, y = int(pos[0]), int(pos[1])
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


def _all_units(farm: dict[str, Any], private: dict[str, Any]) -> list[dict[str, Any]]:
    positions = [list(farm["farmer"])] + [list(p) for p in (farm.get("hands", []) or [])]
    invs = list(private.get("inventories", []) or [])
    while len(invs) < len(positions):
        invs.append({})
    return [
        {"index": i, "pos": positions[i], "inv": dict(invs[i] or {})}
        for i in range(len(positions))
    ]


def _set_unit(bundle: dict[str, Any], unit_index: int, action: list[Any]) -> None:
    if unit_index == 0:
        bundle["farmer"] = action
    else:
        bundle["hands"][unit_index - 1] = action


def _tile_rows(farm: dict[str, Any]):
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row or []):
            yield x, y, tile


def _market_base(item: str, raw: dict[str, Any]) -> int:
    params = (raw.get("market", {}) or {}).get("params")
    table = params or rules.MARKET_PARAMS
    return int(table[item]["base"])


def _remaining(raw: dict[str, Any], configuration: Any) -> tuple[int, float]:
    turns_per_day = _cfg(configuration, "turnsPerDay", 24)
    episode_steps = _cfg(configuration, "episodeSteps", 720)
    step = int(raw.get("step", int(raw["day"]) * turns_per_day + int(raw["hour"])))
    steps = max(0, episode_steps - step)
    return steps, steps / max(1, turns_per_day)


def _crop_terminal_value(crop: str, raw: dict[str, Any], remaining_days: float) -> float:
    spec = rules.CROPS[crop]
    first = int(spec["first_yield_day"])
    if remaining_days < first + 0.5:
        return -1e18
    price = float(raw["market"]["prices"][crop])
    if spec["ongoing"]:
        interval = max(1, int(spec["interval"]))
        cycles = 1 + int(max(0.0, remaining_days - first) // interval)
        units = min(int(spec["max_yield"]), cycles)
    else:
        units = int(spec["max_yield"])
    conservative_units = max(1.0, units * 0.78)
    gross = conservative_units * price
    cost = float(spec["seed"])
    # Earlier cash-return gets a modest preference without replacing terminal value.
    speed = 1.0 + 0.12 / max(1.0, first)
    return (gross - cost) * speed


def _animal_terminal_value(animal: str, raw: dict[str, Any], remaining_days: float) -> float:
    spec = rules.ANIMALS[animal]
    first = int(spec["first_yield_day"])
    if remaining_days < first + 1.0:
        return -1e18
    interval = max(1, int(spec["interval"]))
    cycles = 1 + int(max(0.0, remaining_days - first) // interval)
    product = str(spec["product"])
    product_price = float(raw["market"]["prices"][product])
    wheat_price = float(raw["market"]["prices"]["WHEAT"])
    # Feed overhead is real work/cash opportunity cost; use a conservative proxy.
    feed_cost = min(remaining_days, first + cycles * interval) * wheat_price * 0.55
    gross = cycles * product_price * 0.82
    return gross - float(spec["cost"]) - feed_cost


def _best_crop(raw: dict[str, Any], remaining_days: float, counts: dict[str, int]) -> str | None:
    ranked = []
    for crop in CROPS:
        v = _crop_terminal_value(crop, raw, remaining_days)
        if v <= 0:
            continue
        # Diversity emerges from diminishing value of already-heavy lanes.
        adjusted = v / (1.0 + 0.16 * counts.get(crop, 0))
        ranked.append((adjusted, v, crop))
    return max(ranked)[2] if ranked else None


def _best_animal(raw: dict[str, Any], remaining_days: float, counts: dict[str, int]) -> str | None:
    ranked = []
    for animal in ANIMALS:
        v = _animal_terminal_value(animal, raw, remaining_days)
        if v <= 0:
            continue
        adjusted = v / (1.0 + 0.20 * counts.get(animal, 0))
        ranked.append((adjusted, v, animal))
    return max(ranked)[2] if ranked else None


def _farm_summary(raw: dict[str, Any]) -> dict[str, Any]:
    p = int(raw["player"])
    farm = raw["farms"][p]
    private = raw["private"]
    plants: dict[str, int] = {c: 0 for c in CROPS}
    animals: dict[str, int] = {a: 0 for a in ANIMALS}
    empty: list[tuple[int, int]] = []
    weeds: list[tuple[int, int]] = []
    empty_structures: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}
    harvest: list[tuple[int, int, float]] = []
    water: list[tuple[int, int, int]] = []
    feed: list[tuple[int, int, int]] = []
    care: list[tuple[int, int]] = []

    day = int(raw["day"])
    prices = raw["market"]["prices"]

    for x, y, tile in _tile_rows(farm):
        if tile is None:
            empty.append((x, y))
            continue
        if tile == "LOCKED":
            continue
        if not isinstance(tile, dict):
            continue
        kind = tile.get("kind")
        if kind == "WEED":
            weeds.append((x, y))
            continue
        if kind == "PLANT":
            crop = str(tile["crop"])
            plants[crop] = plants.get(crop, 0) + 1
            streak = int(tile.get("consecutive_unwatered", 0) or 0)
            if not bool(tile.get("watered_today", False)) and streak >= 1:
                water.append((x, y, streak))
            spec = rules.CROPS[crop]
            if int(tile.get("yield_units", 0) or 0) > 0 and day - int(tile["planted_day"]) >= int(spec["first_yield_day"]):
                value = float(tile["yield_units"]) * float(prices[crop])
                harvest.append((x, y, value))
            continue
        if kind in ("COOP", "PASTURE"):
            animal = tile.get("animal")
            if not animal:
                empty_structures[kind].append((x, y))
                continue
            animal = str(animal)
            animals[animal] = animals.get(animal, 0) + 1
            streak = int(tile.get("consecutive_unfed", 0) or 0)
            if not bool(tile.get("fed_today", False)):
                feed.append((x, y, streak))
            if bool(tile.get("fed_today", False)) and not bool(tile.get("cared_today", False)):
                care.append((x, y))
            yield_units = int(tile.get("yield_units", 0) or 0)
            if yield_units > 0:
                product = rules.ANIMALS[animal]["product"]
                harvest.append((x, y, yield_units * float(prices[product])))

    return {
        "farm": farm,
        "private": private,
        "plants": plants,
        "animals": animals,
        "empty": empty,
        "weeds": weeds,
        "empty_structures": empty_structures,
        "harvest": harvest,
        "water": water,
        "feed": feed,
        "care": care,
    }


def _task_action(
    unit: dict[str, Any],
    task: dict[str, Any],
    raw: dict[str, Any],
    summary: dict[str, Any],
    board_size: int,
) -> list[Any]:
    pos = unit["pos"]
    inv = unit["inv"]
    kind = task["kind"]

    if kind == "deliver":
        access = _nearest_shed(pos, board_size)
        return ["DROP"] if tuple(pos) == access else _move_toward(pos, access)

    if kind == "pickup":
        access = _nearest_shed(pos, board_size)
        if tuple(pos) != access:
            return _move_toward(pos, access)
        return ["PICKUP", task["item"], int(task.get("qty", 1))]

    target = tuple(task["target"])
    if tuple(pos) != target:
        return _move_toward(pos, target)

    if kind == "water":
        return ["WATER"]
    if kind == "harvest":
        return ["HARVEST"]
    if kind == "dig":
        return ["DIG"]
    if kind == "care":
        return ["CARE"]
    if kind == "feed":
        return ["FEED"]
    if kind == "plant":
        return ["PLANT", task["crop"]]
    if kind == "build":
        return ["BUILD_COOP"] if task["structure"] == "COOP" else ["BUILD_PASTURE"]
    if kind == "place_animal":
        return ["PLACE", task["animal"], 1]
    return ["PASS"]


def _build_unit_bundle(
    raw: dict[str, Any],
    summary: dict[str, Any],
    remaining_days: float,
    configuration: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    farm = summary["farm"]
    private = summary["private"]
    board_size = _cfg(configuration, "boardSize", len(farm["tiles"]))
    units = _all_units(farm, private)
    bundle = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (farm.get("hands", []) or [])],
        "market": [],
    }
    used_units: set[int] = set()
    used_targets: set[tuple[int, int]] = set()
    virtual_seeds = {c: int(private.get("seeds", {}).get(c, 0) or 0) for c in CROPS}
    virtual_shed_wheat = int(private.get("shed", {}).get("WHEAT", 0) or 0)
    expected_drop = {p: 0 for p in PRODUCTS}

    # A worker already carrying value owns that return path first.
    for unit in units:
        inv = unit["inv"]
        carried_products = {k: int(v) for k, v in inv.items() if k in PRODUCTS and int(v) > 0}
        carried_animals = {k: int(v) for k, v in inv.items() if k in ANIMALS and int(v) > 0}
        if carried_animals:
            animal = max(carried_animals, key=carried_animals.get)
            structure = rules.ANIMALS[animal]["structure"]
            targets = summary["empty_structures"].get(structure, [])
            if targets:
                target = min(targets, key=lambda t: (_distance(unit["pos"], t), t))
                if target not in used_targets:
                    action = _task_action(unit, {"kind": "place_animal", "target": target, "animal": animal}, raw, summary, board_size)
                    _set_unit(bundle, unit["index"], action)
                    used_units.add(unit["index"])
                    used_targets.add(target)
                    continue
        if carried_products:
            action = _task_action(unit, {"kind": "deliver"}, raw, summary, board_size)
            _set_unit(bundle, unit["index"], action)
            used_units.add(unit["index"])
            if action == ["DROP"]:
                for item, qty in carried_products.items():
                    expected_drop[item] += qty

    def assign_spatial(tasks: list[dict[str, Any]]) -> None:
        nonlocal virtual_shed_wheat
        for task in tasks:
            target = tuple(task["target"])
            if target in used_targets:
                continue
            candidates = [u for u in units if u["index"] not in used_units]
            if not candidates:
                return

            if task["kind"] == "plant":
                crop = task["crop"]
                if virtual_seeds.get(crop, 0) <= 0:
                    continue
            if task["kind"] == "feed":
                wheat_carriers = [u for u in candidates if int(u["inv"].get("WHEAT", 0) or 0) > 0]
                if wheat_carriers:
                    candidates = wheat_carriers
                elif virtual_shed_wheat <= 0:
                    continue
                else:
                    # First move a free worker through the shed to obtain one feed unit.
                    unit = min(candidates, key=lambda u: (_distance(u["pos"], _nearest_shed(u["pos"], board_size)), u["index"]))
                    action = _task_action(unit, {"kind": "pickup", "item": "WHEAT", "qty": 1}, raw, summary, board_size)
                    _set_unit(bundle, unit["index"], action)
                    used_units.add(unit["index"])
                    virtual_shed_wheat -= 1
                    continue

            unit = min(candidates, key=lambda u: (_distance(u["pos"], target), u["index"]))
            action = _task_action(unit, task, raw, summary, board_size)
            _set_unit(bundle, unit["index"], action)
            used_units.add(unit["index"])
            used_targets.add(target)
            if task["kind"] == "plant":
                virtual_seeds[task["crop"]] -= 1

    # Survival first only when the state itself says the asset is at risk.
    water_tasks = [
        {"kind": "water", "target": (x, y), "priority": 100 + 20 * streak}
        for x, y, streak in summary["water"]
    ]
    feed_tasks = [
        {"kind": "feed", "target": (x, y), "priority": 110 + 25 * streak}
        for x, y, streak in summary["feed"]
    ]
    assign_spatial(sorted(feed_tasks + water_tasks, key=lambda t: -t["priority"]))

    # Realized output has already paid its production cost: get it moving toward cash.
    harvest_tasks = [
        {"kind": "harvest", "target": (x, y), "priority": 80 + min(50, value / 100.0)}
        for x, y, value in summary["harvest"]
    ]
    assign_spatial(sorted(harvest_tasks, key=lambda t: -t["priority"]))

    # If an animal is waiting in the shed and a compatible structure exists, start placement.
    shed = private.get("shed", {}) or {}
    for animal in ANIMALS:
        if int(shed.get(animal, 0) or 0) <= 0:
            continue
        structure = rules.ANIMALS[animal]["structure"]
        targets = summary["empty_structures"].get(structure, [])
        for target in targets:
            candidates = [u for u in units if u["index"] not in used_units]
            if not candidates:
                break
            unit = min(candidates, key=lambda u: (_distance(u["pos"], _nearest_shed(u["pos"], board_size)), u["index"]))
            if int(unit["inv"].get(animal, 0) or 0) > 0:
                task = {"kind": "place_animal", "target": target, "animal": animal}
            else:
                task = {"kind": "pickup", "item": animal, "qty": 1, "target": target}
            action = _task_action(unit, task, raw, summary, board_size)
            _set_unit(bundle, unit["index"], action)
            used_units.add(unit["index"])
            if task["kind"] == "place_animal":
                used_targets.add(target)
            break

    # Profitable future lanes can reserve one new structure at a time.
    best_animal = _best_animal(raw, remaining_days, summary["animals"])
    if best_animal is not None:
        structure = rules.ANIMALS[best_animal]["structure"]
        waiting = sum(int(shed.get(a, 0) or 0) for a in ANIMALS)
        if waiting == 0 and not summary["empty_structures"].get(structure):
            empties = [t for t in summary["empty"] if t not in used_targets]
            if empties:
                target = min(empties, key=lambda t: (_distance(t, (board_size // 2 - 1, board_size // 2 - 1)), t))
                assign_spatial([{"kind": "build", "target": target, "structure": structure, "priority": 35}])

    # Use seeds already owned before buying more. Choose crop from current terminal opportunity.
    empties = [t for t in summary["empty"] if t not in used_targets]
    crop_counts = dict(summary["plants"])
    plant_tasks: list[dict[str, Any]] = []
    for target in sorted(empties, key=lambda t: (_distance(t, (board_size // 2 - 1, board_size // 2 - 1)), t)):
        available = [c for c in CROPS if virtual_seeds.get(c, 0) > 0 and _crop_terminal_value(c, raw, remaining_days) > 0]
        if not available:
            break
        crop = max(
            available,
            key=lambda c: _crop_terminal_value(c, raw, remaining_days) / (1.0 + 0.16 * crop_counts.get(c, 0)),
        )
        plant_tasks.append({"kind": "plant", "target": target, "crop": crop, "priority": 30})
        crop_counts[crop] = crop_counts.get(crop, 0) + 1
        virtual_seeds[crop] -= 1
        if len(plant_tasks) >= len(units):
            break
    # restore because assign_spatial owns the reservation accounting
    virtual_seeds = {c: int(private.get("seeds", {}).get(c, 0) or 0) for c in CROPS}
    assign_spatial(plant_tasks)

    # Care is useful after basic feed/harvest/plant work, without becoming a global emergency.
    assign_spatial([{"kind": "care", "target": (x, y), "priority": 20} for x, y in summary["care"]])

    # Clear weeds only with spare workers.
    assign_spatial([{"kind": "dig", "target": t, "priority": 10} for t in summary["weeds"]])

    return bundle, expected_drop


def _planned_sell_orders(
    raw: dict[str, Any],
    summary: dict[str, Any],
    expected_drop: dict[str, int],
    remaining_steps: int,
    remaining_days: float,
) -> list[list[Any]]:
    private = summary["private"]
    shed = private.get("shed", {}) or {}
    prices = raw["market"]["prices"]
    animals_total = sum(summary["animals"].values())
    orders: list[list[Any]] = []

    for item in PRODUCTS:
        qty = int(shed.get(item, 0) or 0) + int(expected_drop.get(item, 0) or 0)
        if qty <= 0:
            continue
        if item == "WHEAT":
            feed_reserve = min(qty, animals_total * 2)
            qty -= feed_reserve
        if qty <= 0:
            continue

        base = _market_base(item, raw)
        current = int(prices[item])
        late = remaining_days <= 2.0 or remaining_steps <= 48
        cash_tight = float(summary["farm"].get("money", 0) or 0) < 500
        if late or cash_tight or current >= int(base * 0.78):
            sell_qty = qty
        elif current >= int(base * 0.55):
            sell_qty = max(1, qty // 2)
        else:
            sell_qty = 0
        if sell_qty > 0:
            orders.append(["SELL", item, sell_qty])
    return orders


def _investment_orders(
    raw: dict[str, Any],
    summary: dict[str, Any],
    remaining_days: float,
    configuration: Any,
    already_orders: list[list[Any]],
) -> list[list[Any]]:
    farm = summary["farm"]
    private = summary["private"]
    cash = float(farm.get("money", 0) or 0)
    max_orders = _cfg(configuration, "maxMarketOrdersPerTurn", 10)
    if len(already_orders) >= max_orders:
        return []

    # Conservative liquidity reserve: near-term feeding plus a small execution buffer.
    animal_count = sum(summary["animals"].values())
    wheat_price = float(raw["market"]["prices"]["WHEAT"])
    reserve = 100.0 + animal_count * wheat_price * 1.5
    spendable = max(0.0, cash - reserve)
    orders: list[list[Any]] = []

    # Ensure near-term feed can actually become available in the shed.
    shed_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0) or 0)
    carried_wheat = sum(int((inv or {}).get("WHEAT", 0) or 0) for inv in (private.get("inventories", []) or []))
    feed_need = max(0, animal_count * 2 - shed_wheat - carried_wheat)
    if feed_need > 0 and spendable > wheat_price:
        qty = min(feed_need, 4)
        orders.append(["BUY_PRODUCT", "WHEAT", qty])
        spendable -= qty * wheat_price

    # Workforce follows current farm workload, not a fixed daily cap.
    productive = sum(summary["plants"].values()) + animal_count
    urgent = len(summary["harvest"]) + len(summary["water"]) + len(summary["feed"])
    desired_units = min(12, max(5 if int(raw["day"]) <= 1 else 3, int(math.ceil((productive + urgent) / 5.0)) + 3))
    current_units = 1 + len(farm.get("hands", []) or [])
    hires_today = int(farm.get("hires_today", 0) or 0)
    while current_units < desired_units and len(already_orders) + len(orders) < max_orders:
        cost = float(rules._hire_cost(hires_today, _cfg(configuration, "farmHandCostMult", 1)))
        if spendable < cost or cost > max(25.0, cash * 0.08):
            break
        orders.append(["HIRE"])
        spendable -= cost
        current_units += 1
        hires_today += 1

    # Expand only when owned capacity is genuinely occupied and enough season remains.
    unlocked = list(farm.get("unlocked_quadrants", []) or [])
    owned_tiles = sum(1 for _, _, t in _tile_rows(farm) if t != "LOCKED")
    occupied_tiles = sum(1 for _, _, t in _tile_rows(farm) if t not in (None, "LOCKED"))
    occupancy = occupied_tiles / max(1, owned_tiles)
    if len(unlocked) < 4 and remaining_days > 5.0 and occupancy >= 0.78:
        land_idx = len(unlocked) - 1
        land_cost = float(rules.LAND_PRICES[land_idx])
        if spendable >= land_cost + 250 and len(already_orders) + len(orders) < max_orders:
            orders.append(["BUY_LAND"])
            spendable -= land_cost

    # One animal commitment at a time; do not stack unplaced animals in the shed.
    waiting_animals = sum(int((private.get("shed", {}) or {}).get(a, 0) or 0) for a in ANIMALS)
    best_animal = _best_animal(raw, remaining_days, summary["animals"])
    if best_animal is not None and waiting_animals == 0:
        spec = rules.ANIMALS[best_animal]
        has_structure = bool(summary["empty_structures"].get(spec["structure"]))
        can_build = bool(summary["empty"])
        if (has_structure or can_build) and spendable >= float(spec["cost"]) + 100 and len(already_orders) + len(orders) < max_orders:
            orders.append(["BUY_ANIMAL", best_animal, 1])
            spendable -= float(spec["cost"])

    # Buy only seeds that can still return before terminal.  Quantity is bounded
    # by visible free land and near-term worker throughput, not a fixed crop quota.
    seeds = private.get("seeds", {}) or {}
    current_seed_total = sum(int(seeds.get(c, 0) or 0) for c in CROPS)
    free_tiles = len(summary["empty"])
    seed_target = min(free_tiles, max(0, current_units * 2))
    buy_count = max(0, seed_target - current_seed_total)
    buy_count = min(buy_count, 8)
    planned_counts = dict(summary["plants"])
    for _ in range(buy_count):
        if len(already_orders) + len(orders) >= max_orders:
            break
        crop = _best_crop(raw, remaining_days, planned_counts)
        if crop is None:
            break
        cost = float(rules.CROPS[crop]["seed"])
        if spendable < cost:
            break
        orders.append(["BUY_SEED", crop, 1])
        spendable -= cost
        planned_counts[crop] = planned_counts.get(crop, 0) + 1

    return orders


def agent(obs: Any, configuration: Any = None) -> dict[str, Any]:
    raw = _plain(obs)
    p = int(raw["player"])
    step = int(raw.get("step", 0) or 0)
    _LAST_STEP[p] = step

    summary = _farm_summary(raw)
    remaining_steps, remaining_days = _remaining(raw, configuration)
    bundle, expected_drop = _build_unit_bundle(raw, summary, remaining_days, configuration)

    sell_orders = _planned_sell_orders(raw, summary, expected_drop, remaining_steps, remaining_days)
    invest_orders = _investment_orders(raw, summary, remaining_days, configuration, sell_orders)

    max_orders = _cfg(configuration, "maxMarketOrdersPerTurn", 10)
    bundle["market"] = (sell_orders + invest_orders)[:max_orders]
    return bundle
