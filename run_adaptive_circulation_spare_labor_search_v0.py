#!/usr/bin/env python3
"""Find strict spare-labor return opportunities in fixed10.

Observation only.

Strict candidate:
  - an actor stands on a mature harvestable crop,
  - that actor's baseline action is PASS,
  - another actor in the same turn performs productive expansion
    (PLANT / BUILD_COOP / BUILD_PASTURE / animal PLACE),
  - no HARVEST is already issued in the baseline bundle.

If no strict PASS opportunity exists, also record MOVE-on-mature candidates
as a separate, weaker observation class. No action is changed.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]

MOVE_OPS = {"NORTH", "SOUTH", "EAST", "WEST"}
EXPANSION_OPS = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}


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


def op(action):
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return "PASS"


def positions(farm):
    return [plain(farm.get("farmer"))] + [plain(x) for x in (farm.get("hands", []) or [])]


def unit_actions(bundle, actor_count):
    out = [plain(bundle.get("farmer", ["PASS"]))]
    out.extend(plain(bundle.get("hands", []) or []))
    while len(out) < actor_count:
        out.append(["PASS"])
    return out[:actor_count]


def is_expansion(action):
    o = op(action)
    if o in EXPANSION_OPS:
        return True
    return (
        o == "PLACE"
        and isinstance(action, (list, tuple))
        and len(action) >= 2
        and str(action[1]) in {"GOOSE", "COW", "SHEEP"}
    )


def tile_at(farm, pos):
    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles", []) or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return plain(tiles[y][x])
    return None


def mature_crop(tile, day):
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return False
    crop = str(tile.get("crop"))
    if crop not in CROPS:
        return False
    if float(tile.get("yield_units", 0) or 0) <= 0:
        return False
    age = int(day) - int(tile.get("planted_day", day) or day)
    return age >= int(CROPS[crop]["first_yield_day"])


def inventory(private, idx):
    invs = private.get("inventories", []) or []
    if idx < len(invs) and isinstance(invs[idx], dict):
        return {k: float(v) for k, v in invs[idx].items() if isinstance(v, (int, float)) and v}
    return {}


def market_snapshot(obs):
    return {
        "prices": plain(obs.get("market", {}).get("prices", {})),
        "inventory": plain(obs.get("market", {}).get("inventory", {})),
    }


def snapshot_candidate(obs, action, actor_idx, klass, expansion):
    p = int(obs["player"])
    farm = obs["farms"][p]
    private = obs["private"]
    pos = positions(farm)
    tile = tile_at(farm, pos[actor_idx])
    return {
        "class": klass,
        "step": int(obs["step"]),
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "actor_index": actor_idx,
        "position": pos[actor_idx],
        "tile": tile,
        "actor_inventory": inventory(private, actor_idx),
        "original_action": unit_actions(action, len(pos))[actor_idx],
        "expansion_actions": expansion,
        "cash": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "shed": {k: float(v) for k, v in (private.get("shed", {}) or {}).items() if isinstance(v,(int,float)) and v},
        "seeds": {k: float(v) for k, v in (private.get("seeds", {}) or {}).items() if isinstance(v,(int,float)) and v},
        "market": market_snapshot(obs),
        "action_bundle": action,
    }


def scan_seed(seed):
    model = load(MODEL, f"spare_model_{seed}")
    opp = load(OPPONENT, f"spare_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    strict = []
    move = []

    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))

        p = int(o0["player"])
        farm = o0["farms"][p]
        pos = positions(farm)
        actions = unit_actions(a0, len(pos))
        expansion = [
            {"actor_index": i, "action": a}
            for i, a in enumerate(actions)
            if is_expansion(a)
        ]
        has_harvest = any(op(a) == "HARVEST" for a in actions)

        if expansion and not has_harvest:
            day = int(o0["day"])
            for i, (actor_pos, actor_action) in enumerate(zip(pos, actions)):
                tile = tile_at(farm, actor_pos)
                if not mature_crop(tile, day):
                    continue
                o = op(actor_action)
                if o == "PASS":
                    strict.append(snapshot_candidate(o0, a0, i, "PASS_ON_MATURE_WITH_EXPANSION", expansion))
                elif o in MOVE_OPS:
                    move.append(snapshot_candidate(o0, a0, i, "MOVE_ON_MATURE_WITH_EXPANSION", expansion))

        env.step([a0, a1])

    return {
        "seed": seed,
        "strict_count": len(strict),
        "move_count": len(move),
        "first_strict": strict[0] if strict else None,
        "first_move": move[0] if move else None,
        "strict_examples": strict[:5],
        "move_examples": move[:5],
    }


def main():
    rows = [scan_seed(seed) for seed in SEEDS]
    strict_seeds = [r["seed"] for r in rows if r["strict_count"] > 0]
    move_seeds = [r["seed"] for r in rows if r["move_count"] > 0]

    summary = {
        "n": len(rows),
        "strict_PASS_candidate_seeds": strict_seeds,
        "strict_PASS_seed_count": len(strict_seeds),
        "move_candidate_seeds": move_seeds,
        "move_seed_count": len(move_seeds),
        "strict_total": sum(r["strict_count"] for r in rows),
        "move_total": sum(r["move_count"] for r in rows),
    }

    out = {
        "schema": "adaptive-circulation-spare-labor-search-v0",
        "meaning": "Observation only; no action changed.",
        "summary": summary,
        "cases": rows,
        "boundary": [
            "PASS-on-mature is the strict candidate class.",
            "MOVE-on-mature is recorded separately and is not assumed spare.",
            "No HARVEST candidate is promoted by frequency alone.",
        ],
    }

    Path("adaptive_circulation_spare_labor_search_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps(summary, separators=(",", ":")))
    for r in rows:
        print("CASE " + json.dumps({
            "seed": r["seed"],
            "strict_count": r["strict_count"],
            "move_count": r["move_count"],
            "first_strict": r["first_strict"],
            "first_move": r["first_move"],
        }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
