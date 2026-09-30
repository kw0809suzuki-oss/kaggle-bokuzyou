"""World Lens Agent v0.

Independent, rule-grounded Kaggriculture prototype.

Design:
World -> Value Flow -> Terminal-Reachable Future -> Action

No replay program and no opponent-policy model are imported.
The opponent is observed only through the shared World state it leaves behind.

World Lens:
- Cash
- productive plants / yield
- carried + shed pipeline
- unlocked / free capacity
- current market prices
- remaining time

The agent keeps value circulating while a new crop can still reach cash before
terminal, then stops expansion and closes the existing pipeline.
"""
from __future__ import annotations

import math
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

EPISODE_STEPS = 720
TURNS_PER_DAY = 24
MAX_ORDERS = 10


def _cfg(configuration, key, default):
    if configuration is None:
        return default
    try:
        return configuration[key]
    except Exception:
        return getattr(configuration, key, default)


def _positions(farm):
    return [list(farm["farmer"])] + [list(x) for x in (farm.get("hands", []) or [])]


def _move(pos, target):
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


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shed_access(board_size):
    h = board_size // 2
    return [(h - 1, h - 1), (h, h - 1), (h - 1, h), (h, h)]


def _crop_full_days(crop):
    c = rules.CROPS[crop]
    if c["ongoing"]:
        return int(c["first_yield_day"] + (c["max_yield"] - 1) * c["interval"])
    return int(c["max_yield_day"])


def _expected_units(crop, remaining_steps):
    c = rules.CROPS[crop]
    days = remaining_steps / TURNS_PER_DAY
    first = int(c["first_yield_day"])
    if days < first + 1:
        return 0
    if not c["ongoing"]:
        return int(c["max_yield"] if days >= int(c["max_yield_day"]) + 1 else 1)
    count = 1 + max(0, int((days - first - 1) // max(1, int(c["interval"]))))
    return min(int(c["max_yield"]), count)


def _crop_metrics(obs, remaining_steps):
    prices = obs["market"]["prices"]
    rows = {}
    for crop, c in rules.CROPS.items():
        units = _expected_units(crop, remaining_steps)
        if units <= 0:
            continue
        seed_cost = float(c["seed"])
        price = float(prices[crop])
        gross = units * price
        margin = gross - seed_cost
        if margin <= 0:
            continue
        first_cash_days = float(c["first_yield_day"]) + 1.0
        full_days = float(_crop_full_days(crop)) + 1.0
        effective_days = min(full_days, max(first_cash_days, remaining_steps / TURNS_PER_DAY))
        rows[crop] = {
            "units": units,
            "seed_cost": seed_cost,
            "price": price,
            "margin": margin,
            "capacity_rate": margin / max(1.0, effective_days),
            "cash_velocity": (margin / max(1.0, seed_cost)) / max(1.0, first_cash_days),
            "first_cash_days": first_cash_days,
        }
    return rows


def _plant_tiles(obs):
    p = int(obs["player"])
    farm = obs["farms"][p]
    out = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                out.append((x, y, tile))
    return out


def _harvestable(tile, day):
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return False
    crop = str(tile.get("crop"))
    if crop not in rules.CROPS:
        return False
    if int(tile.get("yield_units", 0) or 0) <= 0:
        return False
    return int(day) - int(tile.get("planted_day", day)) >= int(rules.CROPS[crop]["first_yield_day"])


def _pipeline(obs):
    private = obs["private"]
    shed = sum(int(v or 0) for v in (private.get("shed", {}) or {}).values())
    carry = 0
    for inv in (private.get("inventories", []) or []):
        if isinstance(inv, dict):
            carry += sum(int(v or 0) for v in inv.values())
    return shed, carry


def lens_snapshot(obs, configuration=None):
    step = int(obs.get("step", 0) or 0)
    p = int(obs["player"])
    farm = obs["farms"][p]
    remaining = max(0, int(_cfg(configuration, "episodeSteps", EPISODE_STEPS)) - 1 - step)
    plants = _plant_tiles(obs)
    mature = sum(int(t.get("yield_units", 0) or 0) for _, _, t in plants if _harvestable(t, obs["day"]))
    shed, carry = _pipeline(obs)
    unlocked = sum(1 for row in farm["tiles"] for tile in row if tile != "LOCKED")
    free = sum(1 for row in farm["tiles"] for tile in row if tile is None)
    metrics = _crop_metrics(obs, remaining)
    cap_crop = max(metrics, key=lambda c: (metrics[c]["capacity_rate"], c)) if metrics else None
    velocity_crop = max(metrics, key=lambda c: (metrics[c]["cash_velocity"], c)) if metrics else None
    mode = "CIRCULATE" if metrics else "LIQUIDATE"
    return {
        "step": step,
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "mode": mode,
        "remaining_steps": remaining,
        "cash": float(farm["money"]),
        "plants": len(plants),
        "mature_yield_units": mature,
        "shed_units": shed,
        "carry_units": carry,
        "unlocked_capacity": unlocked,
        "free_capacity": free,
        "capacity_crop": cap_crop,
        "velocity_crop": velocity_crop,
    }


def _desired_total_units(obs, lens):
    p = int(obs["player"])
    farm = obs["farms"][p]
    if lens["mode"] == "LIQUIDATE":
        outstanding = lens["mature_yield_units"] + lens["carry_units"] + lens["shed_units"]
        return min(10, max(1, 1 + math.ceil(outstanding / 8)))
    seed_total = sum(int(v or 0) for v in (obs["private"].get("seeds", {}) or {}).values())
    projected_surface = min(lens["unlocked_capacity"], lens["plants"] + seed_total + 12)
    desired = max(6, math.ceil((projected_surface + 4) / 5))
    if lens["plants"] > 35:
        desired = max(desired, 10)
    return min(12, desired)


def _pair_assign(actions, positions, free_units, tasks, action_at_target):
    tasks = list(tasks)
    while free_units and tasks:
        best = None
        for u in free_units:
            for i, task in enumerate(tasks):
                pos = positions[u]
                target = (int(task[0]), int(task[1]))
                key = (_dist(pos, target), u, target[1], target[0], i)
                if best is None or key < best[0]:
                    best = (key, u, i, task)
        _, u, i, task = best
        target = (int(task[0]), int(task[1]))
        actions[u] = action_at_target(task) if tuple(positions[u]) == target else _move(positions[u], target)
        free_units.remove(u)
        tasks.pop(i)


def _worker_actions(obs, lens):
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]
    positions = _positions(farm)
    invs = list(private.get("inventories", []) or [])
    while len(invs) < len(positions):
        invs.append({})
    actions = [["PASS"] for _ in positions]
    free_units = set(range(len(positions)))
    board_size = len(farm["tiles"])
    access = _shed_access(board_size)

    # Close carried value before opening new work.
    for u in list(sorted(free_units)):
        inv = invs[u] if isinstance(invs[u], dict) else {}
        if not any(int(v or 0) > 0 for v in inv.values()):
            continue
        target = min(access, key=lambda t: (_dist(positions[u], t), t))
        actions[u] = ["DROP"] if tuple(positions[u]) == target else _move(positions[u], target)
        free_units.remove(u)

    plants = _plant_tiles(obs)
    harvest = [(x, y, t) for x, y, t in plants if _harvestable(t, obs["day"])]
    _pair_assign(actions, positions, free_units, harvest, lambda task: ["HARVEST"])

    critical = [
        (x, y, t) for x, y, t in plants
        if not bool(t.get("watered_today", False)) and int(t.get("consecutive_unwatered", 0) or 0) >= 1
        and not _harvestable(t, obs["day"])
    ]
    _pair_assign(actions, positions, free_units, critical, lambda task: ["WATER"])

    if lens["mode"] == "CIRCULATE":
        regular = [
            (x, y, t) for x, y, t in plants
            if not bool(t.get("watered_today", False))
            and (x, y) not in {(a[0], a[1]) for a in critical}
            and not _harvestable(t, obs["day"])
        ]
        _pair_assign(actions, positions, free_units, regular, lambda task: ["WATER"])

        seeds = {str(k): int(v or 0) for k, v in (private.get("seeds", {}) or {}).items()}
        metrics = _crop_metrics(obs, lens["remaining_steps"])
        viable_seed_crops = [c for c, n in seeds.items() if n > 0 and c in metrics]
        empties = [(x, y) for y, row in enumerate(farm["tiles"]) for x, tile in enumerate(row) if tile is None]
        cap = lens["capacity_crop"]
        vel = lens["velocity_crop"]

        def crop_for_tile(x, y):
            preferred = cap if (x + y + int(obs["day"])) % 2 == 0 else vel
            if preferred in viable_seed_crops:
                return preferred
            if not viable_seed_crops:
                return None
            return max(
                viable_seed_crops,
                key=lambda c: (metrics[c]["capacity_rate"] + 25.0 * metrics[c]["cash_velocity"], c),
            )

        plant_tasks = [(x, y, crop_for_tile(x, y)) for x, y in empties]
        plant_tasks = [t for t in plant_tasks if t[2] is not None]

        # Only actual PLANT actions consume the virtual seed reservation.
        while free_units and plant_tasks:
            best = None
            for u in free_units:
                for i, task in enumerate(plant_tasks):
                    x, y, crop = task
                    key = (_dist(positions[u], (x, y)), u, y, x, i)
                    if best is None or key < best[0]:
                        best = (key, u, i, task)
            _, u, i, task = best
            x, y, crop = task
            if tuple(positions[u]) == (x, y) and seeds.get(crop, 0) > 0:
                actions[u] = ["PLANT", crop]
                seeds[crop] -= 1
            else:
                actions[u] = _move(positions[u], (x, y))
            free_units.remove(u)
            plant_tasks.pop(i)

        # Reopen blocked productive surface only when seed input already exists.
        if free_units and any(v > 0 for v in seeds.values()):
            weeds = [
                (x, y) for y, row in enumerate(farm["tiles"]) for x, tile in enumerate(row)
                if isinstance(tile, dict) and tile.get("kind") == "WEED"
            ]
            _pair_assign(actions, positions, free_units, weeds, lambda task: ["DIG"])

    return actions


def _market_actions(obs, lens, configuration=None):
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]
    prices = obs["market"]["prices"]
    remaining = lens["remaining_steps"]
    max_orders = int(_cfg(configuration, "maxMarketOrdersPerTurn", MAX_ORDERS))
    orders = []

    shed = private.get("shed", {}) or {}
    sell_rows = [
        (float(prices.get(item, 0)) * int(qty or 0), str(item), int(qty or 0))
        for item, qty in shed.items() if int(qty or 0) > 0 and item in prices
    ]
    sell_rows.sort(reverse=True)
    sell_limit = max_orders if lens["mode"] == "LIQUIDATE" else min(4, max_orders)
    for _, item, qty in sell_rows[:sell_limit]:
        orders.append(["SELL", item, qty])

    if lens["mode"] == "LIQUIDATE" or len(orders) >= max_orders:
        return orders[:max_orders]

    # Conservative current-world cash estimate after visible shed realization.
    budget = float(farm["money"])
    for _, item, qty in sell_rows[:sell_limit]:
        budget += 0.75 * float(prices[item]) * qty

    desired_total = _desired_total_units(obs, lens)
    current_total = 1 + len(farm.get("hands", []) or [])
    hire_need = max(0, desired_total - current_total)
    hires_today = int(farm.get("hires_today", 0) or 0)
    hire_mult = int(_cfg(configuration, "farmHandCostMult", 1))
    for j in range(min(4, hire_need)):
        if len(orders) >= max_orders:
            break
        cost = float(rules._hire_cost(hires_today + j, hire_mult))
        if budget < cost:
            break
        orders.append(["HIRE"])
        budget -= cost

    # Expand only when current productive pressure is already consuming the
    # available surface and enough time remains for a fresh crop to return.
    unlocked_quads = len(farm.get("unlocked_quadrants", []) or [])
    extra_unlocked = unlocked_quads - 1
    if extra_unlocked < len(rules.LAND_ORDER) and len(orders) < max_orders:
        next_land_cost = float(rules.LAND_PRICES[extra_unlocked])
        seeds_now = sum(int(v or 0) for v in (private.get("seeds", {}) or {}).values())
        pressure = (lens["plants"] + seeds_now) / max(1, lens["unlocked_capacity"])
        metrics = _crop_metrics(obs, remaining)
        min_first = min((m["first_cash_days"] for m in metrics.values()), default=999)
        if (
            pressure >= 0.55
            and remaining >= int((min_first + 2.0) * TURNS_PER_DAY)
            and budget >= next_land_cost + 200
        ):
            orders.append(["BUY_LAND"])
            budget -= next_land_cost

    if len(orders) >= max_orders:
        return orders[:max_orders]

    metrics = _crop_metrics(obs, remaining)
    if not metrics:
        return orders[:max_orders]

    seed_total = sum(int(v or 0) for v in (private.get("seeds", {}) or {}).values())
    target_buffer = min(
        max(0, lens["free_capacity"] + 8),
        max(6, _desired_total_units(obs, lens) * 2),
    )
    need = max(0, min(12, target_buffer - seed_total))
    if need <= 0:
        return orders[:max_orders]

    cap = lens["capacity_crop"]
    vel = lens["velocity_crop"]
    allocations = []
    if cap == vel or vel is None:
        allocations = [(cap, need)]
    else:
        cap_n = int(math.ceil(need * 0.6))
        allocations = [(cap, cap_n), (vel, need - cap_n)]

    for crop, qty in allocations:
        if crop is None or qty <= 0 or len(orders) >= max_orders:
            continue
        cost = int(rules.CROPS[crop]["seed"])
        affordable = min(qty, int(budget // max(1, cost)))
        if affordable <= 0:
            continue
        orders.append(["BUY_SEED", crop, affordable])
        budget -= affordable * cost

    return orders[:max_orders]


def agent(obs, configuration=None):
    lens = lens_snapshot(obs, configuration)
    unit_actions = _worker_actions(obs, lens)
    return {
        "farmer": unit_actions[0] if unit_actions else ["PASS"],
        "hands": unit_actions[1:],
        "market": _market_actions(obs, lens, configuration),
    }


def reset_agent():
    return None
