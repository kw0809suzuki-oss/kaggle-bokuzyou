"""Adaptive Replay Open Slack v0.

Goal
----
Preserve Adaptive Replay Contract Runtime as the standing proposal, while
leaving an *open* Current-World candidate slot beside it.

The slack is intentionally NOT constrained to:
- the replay's operation set,
- the replay's quantities,
- market-order reordering,
- a fixed action family,
- or similarity to the replay action.

Any generator may emit a complete Kaggriculture ActionBundle.  v0 wires one
independent generator, World Lens v0, because that model was built as the
exercise for producing a world-grounded alternative.

Decision boundary
-----------------
Generation is free. Adoption is not.

Each complete candidate is projected one transition through the Official
Kaggriculture actor + market rules using only the Current World already
observable to self (opponent future action is not guessed).  The projected
post-state is mapped to a terminal-reachable-value ledger:

    cash
  + liquid product value
  + carried product value
  + reachable plant output
  + reachable animal output
  + usable seed option value
  + bounded execution/capacity option value

The replay remains selected unless another candidate has strictly larger
projected terminal-reachable value.  The evaluator is a warrant surface, not a
claim of exact terminal value.

Architecture
------------
Current Official World
    -> generate freely:
         [Adaptive Replay, World Lens, ...future generators]
    -> project each candidate with Official rules
    -> compare terminal-reachable-value ledger
    -> issue one complete ActionBundle
    -> next turn: re-enter from the new Official World

This is an experiment model, not a promoted submission.
"""
from __future__ import annotations

import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "adaptive_replay_contract_runtime_v0.py"
WORLD_LENS_PATH = ROOT / "world_lens_agent_v0.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load(BASE_PATH, "_open_slack_adaptive_replay")
world_lens = _load(WORLD_LENS_PATH, "_open_slack_world_lens")

decision_count = 0
alternative_count = 0
last_decision = None
decision_log = []


def _plain(v: Any):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return [_plain(x) for x in v]
    return v


def _cfg(configuration, key: str, default):
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(key, default)
    return getattr(configuration, key, default)


def _canonical(action) -> str:
    return json.dumps(_plain(action), sort_keys=True, separators=(",", ":"))


def _normalize_bundle(obs, action):
    p = int(obs["player"])
    hands = len(obs["farms"][p].get("hands", []) or [])
    a = _plain(action) if isinstance(action, dict) else {}
    farmer = a.get("farmer", ["PASS"])
    if not isinstance(farmer, list) or not farmer:
        farmer = ["PASS"]
    hs = list(a.get("hands", []) or [])
    hs = (hs + [["PASS"]] * hands)[:hands]
    market = list(a.get("market", []) or [])
    return {"farmer": farmer, "hands": hs, "market": market}


def reset_agent():
    global decision_count, alternative_count, last_decision, decision_log
    decision_count = 0
    alternative_count = 0
    last_decision = None
    decision_log = []
    if hasattr(base, "reset_agent"):
        base.reset_agent()
    if hasattr(world_lens, "reset_agent"):
        world_lens.reset_agent()


def _atomic_plant_blocked(actions, private):
    requests = {}
    for a in actions:
        if isinstance(a, (list, tuple)) and a and a[0] == "PLANT" and len(a) > 1:
            crop = str(a[1])
            requests[crop] = requests.get(crop, 0) + 1
    return {
        crop for crop, n in requests.items()
        if n > int((private.get("seeds", {}) or {}).get(crop, 0) or 0)
    }


def _project_units(obs, action, configuration):
    p = int(obs["player"])
    farm = deepcopy(obs["farms"][p])
    private = deepcopy(obs["private"])
    board_size = int(_cfg(configuration, "boardSize", 10))
    turns_per_day = int(_cfg(configuration, "turnsPerDay", 24))
    shed_capacity = int(_cfg(configuration, "shedCapacity", 100))

    actions = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]
    blocked = _atomic_plant_blocked(actions, private)

    for unit_index, a in enumerate(actions):
        use = a
        if (
            isinstance(a, (list, tuple))
            and a
            and a[0] == "PLANT"
            and len(a) > 1
            and str(a[1]) in blocked
        ):
            use = ["PASS"]
        rules._apply_unit_action(
            farm,
            private,
            unit_index,
            use,
            board_size,
            int(obs["day"]),
            turns_per_day,
            shed_capacity,
        )
    return farm, private


def _project_market(obs, action, farm, private, configuration):
    """Official self-only market projection.

    The current shared market is used exactly as observed.  The other seat emits
    no market order in the projection; this is deliberate non-prediction, not an
    assumption about the real opponent.
    """
    p = int(obs["player"])
    market = deepcopy(obs["market"])
    farms = deepcopy(obs["farms"])
    farms[p] = deepcopy(farm)

    privates = [rules._new_private(), rules._new_private()]
    privates[p] = deepcopy(private)

    states = []
    for seat in (0, 1):
        seat_action = {"farmer": ["PASS"], "hands": [], "market": []}
        if seat == p:
            seat_action = {"market": deepcopy(action.get("market", []) or [])}
        states.append(
            SimpleNamespace(
                observation=SimpleNamespace(
                    market=market,
                    farms=farms,
                    private=privates[seat],
                ),
                action=seat_action,
            )
        )

    cfg = {
        "boardSize": int(_cfg(configuration, "boardSize", 10)),
        "maxMarketOrdersPerTurn": int(_cfg(configuration, "maxMarketOrdersPerTurn", 10)),
        "farmHandCostMult": int(_cfg(configuration, "farmHandCostMult", 1)),
        "shedCapacity": int(_cfg(configuration, "shedCapacity", 100)),
    }
    rules._process_market(states, SimpleNamespace(configuration=cfg))
    return farms[p], privates[p], market


def _project(obs, action, configuration):
    farm, private = _project_units(obs, action, configuration)
    farm, private, market = _project_market(
        obs, action, farm, private, configuration
    )
    return {
        "farm": farm,
        "private": private,
        "market": market,
    }


def _liquidation_value(item: str, qty: int, market: dict) -> float:
    if item not in rules.PRODUCTS or qty <= 0:
        return 0.0
    inv = int(market["inventory"][item])
    params = market.get("params")
    total = 0.0
    for _ in range(int(qty)):
        price = float(rules.market_price(item, inv, params))
        total += price
        if price > rules.PRICE_FLOOR:
            inv += 1
    return total


def _remaining_days(obs, configuration) -> float:
    episode_steps = int(_cfg(configuration, "episodeSteps", 720))
    turns_per_day = int(_cfg(configuration, "turnsPerDay", 24))
    remaining_steps = max(0, episode_steps - 1 - int(obs.get("step", 0) or 0))
    return remaining_steps / max(1, turns_per_day)


def _future_crop_units(tile: dict, current_day: int, remaining_days: float) -> int:
    crop = str(tile.get("crop"))
    if crop not in rules.CROPS:
        return 0
    c = rules.CROPS[crop]
    already = int(tile.get("yield_units", 0) or 0)
    age = int(current_day) - int(tile.get("planted_day", current_day))
    horizon_age = age + int(math.floor(remaining_days))

    if not c["ongoing"]:
        if horizon_age < int(c["first_yield_day"]):
            return already
        # Non-ongoing crops can reach max_yield only through watering during
        # their official growth window. Treat max_yield as reachable capacity,
        # not guaranteed production.
        return max(already, int(c["max_yield"]))

    if horizon_age < int(c["first_yield_day"]):
        return already
    interval = max(1, int(c["interval"]))
    n = 1 + max(0, (horizon_age - int(c["first_yield_day"])) // interval)
    return max(already, min(int(c["max_yield"]), n))


def _future_animal_units(tile: dict, current_day: int, remaining_days: float) -> int:
    animal = str(tile.get("animal"))
    if animal not in rules.ANIMALS:
        return 0
    a = rules.ANIMALS[animal]
    already = int(tile.get("yield_units", 0) or 0)
    age = int(current_day) - int(tile.get("placed_day", current_day))
    horizon_age = age + int(math.floor(remaining_days))
    if horizon_age < int(a["first_yield_day"]):
        return already
    interval = max(1, int(a["interval"]))
    future_pulses = 1 + max(0, (horizon_age - int(a["first_yield_day"])) // interval)
    # max_held is storage-at-once, not lifetime output. Bound the valuation so
    # this remains an option-value estimate rather than an infinite producer.
    return already + min(future_pulses, max(1, int(a["max_held"]) * 2))


def _best_crop_option_value(market: dict, remaining_days: float) -> float:
    best = 0.0
    for crop, c in rules.CROPS.items():
        if remaining_days < float(c["first_yield_day"]) + 1.0:
            continue
        price = float(market["prices"][crop])
        gross = price * float(c["max_yield"])
        best = max(best, gross - float(c["seed"]))
    return max(0.0, best)


def _animal_option_value(animal: str, market: dict, remaining_days: float) -> float:
    if animal not in rules.ANIMALS:
        return 0.0
    a = rules.ANIMALS[animal]
    if remaining_days < float(a["first_yield_day"]) + 1.0:
        return 0.0
    product = str(a["product"])
    price = float(market["prices"][product])
    pulses = 1 + max(
        0,
        int((remaining_days - float(a["first_yield_day"]) - 1.0) // max(1, int(a["interval"]))),
    )
    # 0.70 follows the official pricing note's animal production capacity
    # discount for wheat-feed overhead. This is still only a valuation surface.
    gross = 0.70 * price * pulses
    return max(0.0, gross)


def _reachable_value(obs, projected, configuration):
    farm = projected["farm"]
    private = projected["private"]
    market = projected["market"]
    current_day = int(obs["day"])
    remaining_days = _remaining_days(obs, configuration)

    cash = float(farm.get("money", 0) or 0)

    shed_value = 0.0
    shed = private.get("shed", {}) or {}
    for item in rules.PRODUCTS:
        shed_value += _liquidation_value(
            item, int(shed.get(item, 0) or 0), market
        )

    carry_value = 0.0
    for inv in (private.get("inventories", []) or []):
        if not isinstance(inv, dict):
            continue
        for item in rules.PRODUCTS:
            qty = int(inv.get(item, 0) or 0)
            # Carried goods still need transport to the shed.
            carry_value += 0.90 * _liquidation_value(item, qty, market)

    plant_value = 0.0
    animal_value = 0.0
    empty_structures = 0
    free_tiles = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile is None:
                free_tiles += 1
                continue
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = str(tile.get("crop"))
                units = _future_crop_units(tile, current_day, remaining_days)
                plant_value += 0.72 * _liquidation_value(crop, units, market)
            if "animal" in tile:
                animal = str(tile["animal"])
                product = str(rules.ANIMALS[animal]["product"])
                units = _future_animal_units(tile, current_day, remaining_days)
                animal_value += 0.62 * _liquidation_value(product, units, market)
            elif tile.get("kind") in ("COOP", "PASTURE"):
                empty_structures += 1

    # Seeds are not cash, but some are realizable options while enough time and
    # surface remain. Bound by currently usable surface and execution capacity.
    hands = len(farm.get("hands", []) or [])
    execution_units = 1 + hands
    seed_option = 0.0
    usable_plant_slots = min(
        free_tiles,
        max(0, int(execution_units * min(3.0, max(0.0, remaining_days / 4.0)))),
    )
    seed_rows = []
    for crop, qty in (private.get("seeds", {}) or {}).items():
        q = int(qty or 0)
        if q <= 0 or crop not in rules.CROPS:
            continue
        c = rules.CROPS[crop]
        if remaining_days < float(c["first_yield_day"]) + 1.0:
            continue
        unit = 0.55 * float(market["prices"][crop]) * float(c["max_yield"])
        seed_rows.extend([unit] * q)
    seed_rows.sort(reverse=True)
    seed_option = sum(seed_rows[:usable_plant_slots])

    # Unplaced animals can still become productive if a matching structure can
    # be used/built in time. Treat as bounded option value.
    unplaced_animal_value = 0.0
    animal_units = 0
    for source in [shed, *(private.get("inventories", []) or [])]:
        if not isinstance(source, dict):
            continue
        for animal in rules.ANIMALS:
            animal_units += int(source.get(animal, 0) or 0)
            unplaced_animal_value += int(source.get(animal, 0) or 0) * _animal_option_value(
                animal, market, remaining_days
            )
    unplaced_animal_value *= 0.45 if empty_structures + free_tiles > 0 else 0.15

    # Labor and land are enabling assets, not terminal cash. Give them only
    # bounded option value proportional to remaining usable work, preventing
    # the evaluator from treating raw capacity as equivalent to realized value.
    work_pressure = min(
        40,
        free_tiles
        + len(seed_rows)
        + int(sum(int(v or 0) for v in shed.values()) > 0) * 4
        + animal_units * 3,
    )
    labor_option = min(1200.0, remaining_days * 3.0 * min(execution_units, max(1, work_pressure)))
    best_crop_margin = _best_crop_option_value(market, remaining_days)
    usable_capacity = min(free_tiles, max(0, execution_units * 2))
    capacity_option = 0.12 * usable_capacity * best_crop_margin

    components = {
        "cash": cash,
        "shed_value": shed_value,
        "carry_value": carry_value,
        "plant_value": plant_value,
        "animal_value": animal_value,
        "seed_option": seed_option,
        "unplaced_animal_value": unplaced_animal_value,
        "labor_option": labor_option,
        "capacity_option": capacity_option,
    }
    total = sum(components.values())
    return total, components


def _generate_candidates(obs, configuration):
    """Open candidate surface.

    Generators are allowed to emit an arbitrary complete ActionBundle.  No
    replay-similarity rule is applied here.
    """
    replay_action = _normalize_bundle(obs, base.agent(obs, configuration))
    generated = [("replay", replay_action)]

    # First free generator: the independent World Lens agent.
    try:
        wl = _normalize_bundle(obs, world_lens.agent(obs, configuration))
        generated.append(("world_lens", wl))
    except Exception:
        pass

    # Dedupe identical complete bundles while preserving provenance order.
    out = []
    seen = set()
    for source, action in generated:
        key = _canonical(action)
        if key in seen:
            continue
        seen.add(key)
        out.append((source, action))
    return out


def agent(obs, configuration=None):
    global decision_count, alternative_count, last_decision, decision_log

    candidates = _generate_candidates(obs, configuration)
    evaluated = []
    for source, action in candidates:
        projected = _project(obs, action, configuration)
        score, components = _reachable_value(obs, projected, configuration)
        evaluated.append({
            "source": source,
            "action": action,
            "score": float(score),
            "components": components,
        })

    replay_row = next(r for r in evaluated if r["source"] == "replay")
    best = max(
        evaluated,
        key=lambda r: (
            r["score"],
            1 if r["source"] == "replay" else 0,
        ),
    )

    # Strict improvement is the only adoption gate. Ties stay on replay.
    chosen = best if best["score"] > replay_row["score"] + 1e-9 else replay_row

    decision_count += 1
    if chosen["source"] != "replay":
        alternative_count += 1

    last_decision = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "chosen": chosen["source"],
        "replay_score": replay_row["score"],
        "chosen_score": chosen["score"],
        "delta_score": chosen["score"] - replay_row["score"],
        "replay_action": replay_row["action"],
        "chosen_action": chosen["action"],
        "replay_components": replay_row["components"],
        "chosen_components": chosen["components"],
    }
    if chosen["source"] != "replay":
        decision_log.append(deepcopy(last_decision))

    return deepcopy(chosen["action"])
