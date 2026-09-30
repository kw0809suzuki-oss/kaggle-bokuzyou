#!/usr/bin/env python3
"""Circulation Effect Trace v0.

Question:
    Read the DECEM 157,026 Source Replay as L/P/R/W transition effects, then
    run current Adaptive Circulation Runtime v0 in one fixed current World and
    find the first divergence per effect surface.

This is observation only. The candidate model is behavior-identical to
Adaptive Replay Contract Runtime v0. No repair is applied.

Surfaces:
    L = Liquid cash effect.
    P = Productive-body effect: hands, unlocked capacity, plants, animals.
    R = Return-pipeline effect: yield-on-asset, carried products, shed products.
    W = Work-allocation effect: issued self ActionBundle + actor-position change.
    X = Inputs: seeds.
    C = World context: market inventory/price and unlocked shops.

Absolute values are retained for context, but divergence is judged on
transition effects (deltas / added-removed sets), not on a scalar strength score.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
import sys
from pathlib import Path

import requests
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

EPISODE_ID = 115303987
SOURCE_SEED = 1305581493
SOURCE_URL = f"https://www.kaggle.com/competitions/episodes/{EPISODE_ID}/replay.json"

CURRENT_SEED = 92802001


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def parse_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def obs_from(step_row, seat):
    return plain(parse_obj(step_row[seat].get("observation")))


def action_from(step_row, seat):
    return plain(parse_obj(step_row[seat].get("action")))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def nz_map(d, keys=None):
    d = d or {}
    if keys is None:
        keys = sorted(d)
    out = {}
    for k in keys:
        v = d.get(k, 0)
        if isinstance(v, (int, float)) and v:
            out[str(k)] = float(v)
    return out


def add_counter(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, (int, float)) and v:
            dst[str(k)] += float(v)


def state_surface(obs):
    obs = plain(obs)
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]

    plants = Counter()
    animals = Counter()
    plant_yield = Counter()
    animal_yield = Counter()
    empty_tiles = 0
    locked_tiles = 0
    weeds = 0

    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile is None:
                empty_tiles += 1
                continue
            if tile == "LOCKED":
                locked_tiles += 1
                continue
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind"))
            if kind == "WEED":
                weeds += 1
            elif kind == "PLANT":
                crop = str(tile.get("crop"))
                plants[crop] += 1
                q = float(tile.get("yield_units", 0) or 0)
                if q:
                    plant_yield[crop] += q
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                animal = str(tile.get("animal"))
                animals[animal] += 1
                product = str(rules.ANIMALS[animal]["product"])
                q = float(tile.get("yield_units", 0) or 0)
                if q:
                    animal_yield[product] += q

    carried = Counter()
    for inv in private.get("inventories", []) or []:
        if isinstance(inv, dict):
            add_counter(carried, inv)

    shed = Counter()
    add_counter(shed, private.get("shed", {}) or {})

    seeds = Counter()
    add_counter(seeds, private.get("seeds", {}) or {})

    positions = [plain(farm.get("farmer"))] + [plain(x) for x in (farm.get("hands", []) or [])]

    return {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "L": {
            "cash": float(farm.get("money", 0) or 0),
        },
        "P": {
            "hands": len(farm.get("hands", []) or []),
            "unlocked_quadrants": len(farm.get("unlocked_quadrants", []) or []),
            "unlocked_tiles": sum(1 for row in farm.get("tiles", []) or [] for tile in row if tile != "LOCKED"),
            "empty_tiles": empty_tiles,
            "weeds": weeds,
            "plants": dict(sorted(plants.items())),
            "animals": dict(sorted(animals.items())),
        },
        "R": {
            "plant_yield": dict(sorted(plant_yield.items())),
            "animal_yield": dict(sorted(animal_yield.items())),
            "carried": dict(sorted(carried.items())),
            "shed": dict(sorted(shed.items())),
        },
        "X": {
            "seeds": dict(sorted(seeds.items())),
        },
        "W": {
            "positions": positions,
        },
        "C": {
            "market_inventory": nz_map(obs.get("market", {}).get("inventory", {}), rules.PRODUCTS),
            "market_prices": nz_map(obs.get("market", {}).get("prices", {}), rules.PRODUCTS),
            "unlocked_shops": list(obs.get("town", {}).get("unlocked_shops", []) or []),
        },
    }


def numeric_delta_map(a, b):
    keys = sorted(set((a or {}).keys()) | set((b or {}).keys()))
    out = {}
    for k in keys:
        av = float((a or {}).get(k, 0) or 0)
        bv = float((b or {}).get(k, 0) or 0)
        d = bv - av
        if d:
            out[str(k)] = d
    return out


def scalar_delta(a, b, key):
    return float(b.get(key, 0) or 0) - float(a.get(key, 0) or 0)


def transition_effect(pre_obs, post_obs, action):
    pre = state_surface(pre_obs)
    post = state_surface(post_obs)

    pre_shops = list(pre["C"]["unlocked_shops"])
    post_shops = list(post["C"]["unlocked_shops"])
    added_shops = post_shops[len(pre_shops):] if post_shops[:len(pre_shops)] == pre_shops else [
        x for x in post_shops if x not in pre_shops
    ]

    return {
        "issued_step": pre["step"],
        "observed_step": post["step"],
        "L": {
            "cash_delta": scalar_delta(pre["L"], post["L"], "cash"),
        },
        "P": {
            "hands_delta": scalar_delta(pre["P"], post["P"], "hands"),
            "unlocked_quadrants_delta": scalar_delta(pre["P"], post["P"], "unlocked_quadrants"),
            "unlocked_tiles_delta": scalar_delta(pre["P"], post["P"], "unlocked_tiles"),
            "empty_tiles_delta": scalar_delta(pre["P"], post["P"], "empty_tiles"),
            "weeds_delta": scalar_delta(pre["P"], post["P"], "weeds"),
            "plants_delta": numeric_delta_map(pre["P"]["plants"], post["P"]["plants"]),
            "animals_delta": numeric_delta_map(pre["P"]["animals"], post["P"]["animals"]),
        },
        "R": {
            "plant_yield_delta": numeric_delta_map(pre["R"]["plant_yield"], post["R"]["plant_yield"]),
            "animal_yield_delta": numeric_delta_map(pre["R"]["animal_yield"], post["R"]["animal_yield"]),
            "carried_delta": numeric_delta_map(pre["R"]["carried"], post["R"]["carried"]),
            "shed_delta": numeric_delta_map(pre["R"]["shed"], post["R"]["shed"]),
        },
        "X": {
            "seeds_delta": numeric_delta_map(pre["X"]["seeds"], post["X"]["seeds"]),
        },
        "W": {
            "action": plain(action),
            "pre_positions": pre["W"]["positions"],
            "post_positions": post["W"]["positions"],
        },
        "C": {
            "market_inventory_delta": numeric_delta_map(pre["C"]["market_inventory"], post["C"]["market_inventory"]),
            "market_prices_delta": numeric_delta_map(pre["C"]["market_prices"], post["C"]["market_prices"]),
            "shops_added": added_shops,
        },
        "pre_surface": pre,
        "post_surface": post,
    }


def load_source_replay():
    r = requests.get(SOURCE_URL, timeout=30)
    r.raise_for_status()
    replay = r.json()
    if isinstance(replay, dict) and isinstance(replay.get("replay"), str):
        replay = json.loads(replay["replay"])
    return replay


def source_trace(replay):
    steps = replay["steps"]
    out = []
    for i in range(len(steps) - 1):
        pre = obs_from(steps[i], 0)
        post = obs_from(steps[i + 1], 0)
        action = action_from(steps[i + 1], 0)
        out.append(transition_effect(pre, post, action))
    return out


def current_trace():
    model = load(MODEL, "adaptive_circulation_runtime_trace")
    opponent = load(OPPONENT, "adaptive_circulation_runtime_opponent")

    env = make("kaggriculture", configuration={"seed": CURRENT_SEED}, debug=False)
    env.reset(num_agents=2)
    out = []

    while not env.done:
        pre0 = shared(env, 0)
        pre1 = shared(env, 1)
        a0 = plain(model.agent(deepcopy(pre0), env.configuration))
        a1 = plain(opponent.agent(deepcopy(pre1)))
        env.step([a0, a1])
        post0 = plain(env.state[0].observation if env.done else env._Environment__get_shared_state(0)["observation"])
        out.append(transition_effect(pre0, post0, a0))

    terminal = float(env.state[0].observation["farms"][0]["money"])
    return out, terminal


def stripped(effect, channel):
    return effect[channel]


def first_divergence(source, current, channel):
    n = min(len(source), len(current))
    for i in range(n):
        a = stripped(source[i], channel)
        b = stripped(current[i], channel)
        if a != b:
            return {
                "transition_index": i,
                "source_issued_step": source[i]["issued_step"],
                "current_issued_step": current[i]["issued_step"],
                "source": a,
                "current": b,
                "source_pre_surface": source[i]["pre_surface"],
                "current_pre_surface": current[i]["pre_surface"],
            }
    if len(source) != len(current):
        return {
            "transition_index": n,
            "reason": "trace_length_mismatch",
            "source_length": len(source),
            "current_length": len(current),
        }
    return None


def action_only(effect):
    return effect["W"]["action"]


def first_action_divergence(source, current):
    n = min(len(source), len(current))
    for i in range(n):
        if action_only(source[i]) != action_only(current[i]):
            return {
                "transition_index": i,
                "source_issued_step": source[i]["issued_step"],
                "current_issued_step": current[i]["issued_step"],
                "source_action": action_only(source[i]),
                "current_action": action_only(current[i]),
                "source_effect_L": source[i]["L"],
                "current_effect_L": current[i]["L"],
                "source_effect_P": source[i]["P"],
                "current_effect_P": current[i]["P"],
                "source_effect_R": source[i]["R"],
                "current_effect_R": current[i]["R"],
            }
    return None



def action_operations(action):
    action = action or {}
    unit_actions = [action.get("farmer", ["PASS"])] + list(action.get("hands", []) or [])
    market_actions = list(action.get("market", []) or [])
    ops = []
    for a in unit_actions:
        if isinstance(a, (list, tuple)) and a:
            ops.append(str(a[0]))
    for a in market_actions:
        if isinstance(a, (list, tuple)) and a:
            ops.append(str(a[0]))
    return ops


def flow_roles(action):
    ops = set(action_operations(action))
    roles = set()

    if ops & {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND", "HIRE"}:
        roles.add("EXPAND_OR_INVEST")
    if "BUY_PRODUCT" in ops:
        roles.add("ACQUIRE_MATERIAL")
    if ops & {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}:
        roles.add("ACTIVATE_PRODUCTIVE_BODY")
    if "PLACE" in ops:
        roles.add("PLACE_OR_APPLY_MATERIAL")
    if ops & {"WATER", "FEED", "CARE", "FERTILIZE"}:
        roles.add("MAINTAIN_PRODUCTIVE_BODY")
    if "HARVEST" in ops:
        roles.add("EXTRACT_OUTPUT")
        roles.add("RETURN_PATH")
    if "DROP" in ops:
        roles.add("TRANSFER_TO_SHED")
        roles.add("RETURN_PATH")
    if "SELL" in ops:
        roles.add("REALIZE_CASH")
        roles.add("RETURN_PATH")
    if "PICKUP" in ops:
        roles.add("TRANSFER_FROM_SHED")
    if "COLLECT_FERTILIZER" in ops:
        roles.add("COLLECT_BYPRODUCT")
    if ops & {"NORTH", "SOUTH", "EAST", "WEST"}:
        roles.add("MOVE")
    if not roles:
        roles.add("OTHER")
    return sorted(roles)


def first_effect_divergence_by_role(source, current, role):
    n = min(len(source), len(current))
    for i in range(n):
        sr = flow_roles(source[i]["W"]["action"])
        cr = flow_roles(current[i]["W"]["action"])
        if role not in sr and role not in cr:
            continue
        differing = [
            ch for ch in ("L", "P", "R", "X")
            if source[i][ch] != current[i][ch]
        ]
        if differing:
            return {
                "transition_index": i,
                "source_roles": sr,
                "current_roles": cr,
                "source_action": source[i]["W"]["action"],
                "current_action": current[i]["W"]["action"],
                "differing_effect_channels": differing,
                "source_effect": {ch: source[i][ch] for ch in differing},
                "current_effect": {ch: current[i][ch] for ch in differing},
            }
    return None

def compact_effect(e):
    return {k: e[k] for k in ("issued_step", "observed_step", "L", "P", "R", "X", "W", "C")}


def main():
    replay = load_source_replay()
    source = source_trace(replay)
    current, current_terminal = current_trace()

    channels = ["L", "P", "R", "X", "W", "C"]
    first_by_channel = {ch: first_divergence(source, current, ch) for ch in channels}
    first_action = first_action_divergence(source, current)
    role_names = [
        "EXPAND_OR_INVEST",
        "ACQUIRE_MATERIAL",
        "ACTIVATE_PRODUCTIVE_BODY",
        "MAINTAIN_PRODUCTIVE_BODY",
        "RETURN_PATH",
        "EXTRACT_OUTPUT",
        "TRANSFER_TO_SHED",
        "REALIZE_CASH",
    ]
    first_by_role = {
        role: first_effect_divergence_by_role(source, current, role)
        for role in role_names
    }

    # Preserve a narrow window around the earliest self-circulation divergence.
    self_channels = ["L", "P", "R", "X"]
    self_indices = [
        first_by_channel[ch]["transition_index"]
        for ch in self_channels
        if first_by_channel[ch] is not None
    ]
    first_self_idx = min(self_indices) if self_indices else None
    window = []
    if first_self_idx is not None:
        lo = max(0, first_self_idx - 2)
        hi = min(len(source), len(current), first_self_idx + 3)
        for i in range(lo, hi):
            window.append({
                "transition_index": i,
                "source": compact_effect(source[i]),
                "current": compact_effect(current[i]),
            })

    source_terminal = float(obs_from(replay["steps"][-1], 0)["farms"][0]["money"])

    out = {
        "schema": "adaptive-circulation-effect-trace-v0",
        "meaning": "Observer-only source/current circulation transition comparison; no policy change.",
        "source": {
            "episode_id": EPISODE_ID,
            "seed": SOURCE_SEED,
            "terminal_self": source_terminal,
            "transition_count": len(source),
        },
        "current": {
            "seed": CURRENT_SEED,
            "opponent": "Seyamalam pinned v21",
            "terminal_self": current_terminal,
            "transition_count": len(current),
            "model": "Adaptive Circulation Runtime v0 (passthrough)",
        },
        "surface_definition": {
            "L": "liquid cash transition",
            "P": "productive-body transition",
            "R": "material-stage transition (yield / carry / shed); role-neutral until action provenance is checked",
            "X": "input inventory transition",
            "W": "issued work + actor-position transition",
            "C": "shared World context transition",
        },
        "first_action_divergence": first_action,
        "first_divergence_by_channel": first_by_channel,
        "first_effect_divergence_by_flow_role": first_by_role,
        "first_self_circulation_divergence_index": first_self_idx,
        "window_around_first_self_divergence": window,
        "boundary": [
            "No scalar circulation score is created.",
            "A divergence identifies a changed transition effect, not a cause.",
            "No repair is promoted from this trace alone.",
            "Current v0 emits exactly the existing Adaptive Replay Contract Runtime behavior.",
        ],
    }

    Path("adaptive_circulation_effect_trace_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_terminal_self": source_terminal,
        "current_terminal_self": current_terminal,
        "first_action_divergence": None if first_action is None else first_action["transition_index"],
        "first_L": None if first_by_channel["L"] is None else first_by_channel["L"]["transition_index"],
        "first_P": None if first_by_channel["P"] is None else first_by_channel["P"]["transition_index"],
        "first_R": None if first_by_channel["R"] is None else first_by_channel["R"]["transition_index"],
        "first_X": None if first_by_channel["X"] is None else first_by_channel["X"]["transition_index"],
        "first_W": None if first_by_channel["W"] is None else first_by_channel["W"]["transition_index"],
        "first_C": None if first_by_channel["C"] is None else first_by_channel["C"]["transition_index"],
        "first_self_circulation": first_self_idx,
        "first_return_path_effect": None if first_by_role["RETURN_PATH"] is None else first_by_role["RETURN_PATH"]["transition_index"],
        "first_acquire_material_effect": None if first_by_role["ACQUIRE_MATERIAL"] is None else first_by_role["ACQUIRE_MATERIAL"]["transition_index"],
        "first_expand_effect": None if first_by_role["EXPAND_OR_INVEST"] is None else first_by_role["EXPAND_OR_INVEST"]["transition_index"],
    }
    print("SUMMARY " + json.dumps(summary, separators=(",", ":")))
    for ch in ["L", "P", "R", "X", "W", "C"]:
        row = first_by_channel[ch]
        if row is not None:
            print("FIRST_" + ch + " " + json.dumps({
                "transition_index": row.get("transition_index"),
                "source": row.get("source"),
                "current": row.get("current"),
            }, ensure_ascii=False, separators=(",", ":")))
    print("FIRST_ACTION " + json.dumps(first_action, ensure_ascii=False, separators=(",", ":")))
    for role in role_names:
        row = first_by_role[role]
        if row is not None:
            print("FIRST_ROLE_" + role + " " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
