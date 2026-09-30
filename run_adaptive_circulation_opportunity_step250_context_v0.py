#!/usr/bin/env python3
"""Observe the shared step250 decision context across fixed10.

No policy change. No selector fitting.

Goal:
  Compare the exact Current-World context at the one-shot opportunity point
  where WATER->HARVEST produced outcomes from -9494 to +21995.

Record only observable state and the baseline issued ActionBundle.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import statistics
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, ANIMALS

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]
TARGET_STEP = 250

KNOWN_DELTAS = {
    92802001: -7624,
    92802002: 169,
    92802003: -9494,
    92802004: 21,
    92802005: 16,
    92802006: 686,
    92802007: 173,
    92802008: 17,
    92802009: 20183,
    92802010: 21995,
}


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


def positions(farm):
    return [plain(farm.get("farmer"))] + [plain(x) for x in (farm.get("hands", []) or [])]


def action_ops(bundle):
    unit = [bundle.get("farmer", ["PASS"])] + list(bundle.get("hands", []) or [])
    ops = Counter()
    for a in unit:
        if isinstance(a, (list, tuple)) and a:
            ops[str(a[0])] += 1
    for a in bundle.get("market", []) or []:
        if isinstance(a, (list, tuple)) and a:
            ops[str(a[0])] += 1
    return dict(sorted(ops.items()))


def actor_inventory(private, idx):
    invs = private.get("inventories", []) or []
    if idx < len(invs) and isinstance(invs[idx], dict):
        return {k: float(v) for k, v in invs[idx].items() if isinstance(v, (int, float)) and v}
    return {}


def tile_at(farm, pos):
    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles", []) or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return plain(tiles[y][x])
    return None


def farm_surface(obs):
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]
    day = int(obs["day"])

    plants = Counter()
    mature_plants = Counter()
    mature_units = Counter()
    animals = Counter()
    animal_output_units = Counter()
    weeds = 0
    empty = 0
    harvestable_actor_indices = []

    pos = positions(farm)
    for idx, actor_pos in enumerate(pos):
        tile = tile_at(farm, actor_pos)
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = str(tile.get("crop"))
            cd = CROPS.get(crop)
            if cd and day - int(tile.get("planted_day", day)) >= int(cd["first_yield_day"]) and float(tile.get("yield_units",0) or 0) > 0:
                harvestable_actor_indices.append(idx)
        elif isinstance(tile, dict) and tile.get("animal") and float(tile.get("yield_units",0) or 0) > 0:
            harvestable_actor_indices.append(idx)

    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile is None:
                empty += 1
            elif tile == "LOCKED":
                continue
            elif isinstance(tile, dict):
                if tile.get("kind") == "WEED":
                    weeds += 1
                elif tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop"))
                    plants[crop] += 1
                    cd = CROPS.get(crop)
                    yu = float(tile.get("yield_units", 0) or 0)
                    if cd and day - int(tile.get("planted_day", day)) >= int(cd["first_yield_day"]) and yu > 0:
                        mature_plants[crop] += 1
                        mature_units[crop] += yu
                elif tile.get("animal"):
                    animal = str(tile.get("animal"))
                    animals[animal] += 1
                    yu = float(tile.get("yield_units", 0) or 0)
                    if yu:
                        animal_output_units[str(ANIMALS[animal]["product"])] += yu

    carried = Counter()
    for inv in private.get("inventories", []) or []:
        if isinstance(inv, dict):
            for k, v in inv.items():
                if isinstance(v, (int, float)) and v:
                    carried[str(k)] += float(v)

    return {
        "cash": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "unlocked_quadrants": list(farm.get("unlocked_quadrants", []) or []),
        "plants": dict(sorted(plants.items())),
        "mature_plants": dict(sorted(mature_plants.items())),
        "mature_units": dict(sorted(mature_units.items())),
        "animals": dict(sorted(animals.items())),
        "animal_output_units": dict(sorted(animal_output_units.items())),
        "weeds": weeds,
        "empty_tiles": empty,
        "seeds": {k: float(v) for k,v in (private.get("seeds",{}) or {}).items() if isinstance(v,(int,float)) and v},
        "shed": {k: float(v) for k,v in (private.get("shed",{}) or {}).items() if isinstance(v,(int,float)) and v},
        "carried": dict(sorted(carried.items())),
        "actor_positions": pos,
        "harvestable_actor_indices": harvestable_actor_indices,
        "actor3_inventory": actor_inventory(private, 3),
        "actor3_tile": tile_at(farm, pos[3]) if len(pos) > 3 else None,
        "actor6_inventory": actor_inventory(private, 6),
        "actor6_tile": tile_at(farm, pos[6]) if len(pos) > 6 else None,
    }


def run_to_step(seed):
    model = load(MODEL, f"ctx_model_{seed}")
    opp = load(OPPONENT, f"ctx_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        step = int(o0["step"])
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))

        if step == TARGET_STEP:
            surface = farm_surface(o0)
            opp_farm = o0["farms"][1]
            return {
                "seed": seed,
                "known_terminal_delta_from_one_shot": KNOWN_DELTAS[seed],
                "day": int(o0["day"]),
                "hour": int(o0["hour"]),
                "self": surface,
                "opponent_cash": float(opp_farm.get("money",0) or 0),
                "market_prices": plain(o0.get("market",{}).get("prices",{})),
                "market_inventory": plain(o0.get("market",{}).get("inventory",{})),
                "unlocked_shops": plain(o0.get("town",{}).get("unlocked_shops",[]) or []),
                "baseline_action": a0,
                "baseline_action_ops": action_ops(a0),
            }

        env.step([a0, a1])

    raise RuntimeError(f"step {TARGET_STEP} not reached for seed {seed}")


def pick_fields(row):
    s = row["self"]
    return {
        "seed": row["seed"],
        "delta": row["known_terminal_delta_from_one_shot"],
        "cash": s["cash"],
        "opp_cash": row["opponent_cash"],
        "hands": s["hands"],
        "mature_plants": s["mature_plants"],
        "mature_units": s["mature_units"],
        "seeds": s["seeds"],
        "shed": s["shed"],
        "carried": s["carried"],
        "harvestable_actor_indices": s["harvestable_actor_indices"],
        "actor3_inventory": s["actor3_inventory"],
        "actor3_tile": s["actor3_tile"],
        "actor6_inventory": s["actor6_inventory"],
        "actor6_tile": s["actor6_tile"],
        "WHEAT_price": row["market_prices"].get("WHEAT"),
        "STRAWBERRY_price": row["market_prices"].get("STRAWBERRY"),
        "MELON_price": row["market_prices"].get("MELON"),
        "MILK_price": row["market_prices"].get("MILK"),
        "WOOL_price": row["market_prices"].get("WOOL"),
        "action_ops": row["baseline_action_ops"],
    }


def main():
    rows = [run_to_step(seed) for seed in SEEDS]
    compact = [pick_fields(r) for r in rows]

    negatives = [r for r in compact if r["delta"] < 0]
    large_positive = [r for r in compact if r["delta"] >= 20000]

    out = {
        "schema": "adaptive-circulation-opportunity-step250-context-v0",
        "meaning": "Observation only. Same decision point across fixed10; no selector fitted.",
        "target_step": TARGET_STEP,
        "all_cases": compact,
        "negative_cases": negatives,
        "large_positive_cases": large_positive,
        "boundary": [
            "Differences observed here are candidate context features, not causes.",
            "No threshold or selector is created from these 10 outcomes.",
            "Any future condition must be tested on held-out worlds before promotion.",
        ],
    }

    Path("adaptive_circulation_opportunity_step250_context_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for r in compact:
        print("CTX " + json.dumps(r, ensure_ascii=False, separators=(",", ":")))
    print("NEGATIVE " + json.dumps(negatives, ensure_ascii=False, separators=(",", ":")))
    print("LARGE_POSITIVE " + json.dumps(large_positive, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
