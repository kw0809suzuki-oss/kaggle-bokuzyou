#!/usr/bin/env python3
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "adaptive_replay_contract_runtime_v0.py"
CANDIDATE = ROOT / "world_lens_agent_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [92802001, 92802002, 92802003, 92802004, 92802005,
         92802006, 92802007, 92802008, 92802009, 92802010]


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
    return env._Environment__get_shared_state(seat)["observation"]


def invoke(mod, obs, cfg):
    try:
        return mod.agent(obs, cfg)
    except TypeError:
        return mod.agent(obs)


def snapshot(obs):
    p = int(obs["player"])
    farm = obs["farms"][p]
    tiles = farm["tiles"]
    plants = 0
    animals = 0
    yld = 0
    weeds = 0
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    plants += 1
                    yld += int(tile.get("yield_units", 0) or 0)
                elif tile.get("kind") == "WEED":
                    weeds += 1
                if "animal" in tile:
                    animals += 1
    shed = sum(int(v or 0) for v in (obs["private"].get("shed", {}) or {}).values())
    carry = sum(
        sum(int(v or 0) for v in inv.values())
        for inv in (obs["private"].get("inventories", []) or [])
        if isinstance(inv, dict)
    )
    return {
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "cash": float(farm["money"]),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "hands": len(farm.get("hands", []) or []),
        "plants": plants,
        "animals": animals,
        "yield": yld,
        "weeds": weeds,
        "shed": shed,
        "carry": carry,
        "seed_units": sum(int(v or 0) for v in (obs["private"].get("seeds", {}) or {}).values()),
    }


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    daily = []

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        if int(obs0.get("hour", 0) or 0) == 0:
            row = snapshot(obs0)
            if hasattr(model, "lens_snapshot"):
                row["lens"] = plain(model.lens_snapshot(obs0, env.configuration))
            daily.append(row)
        a0 = plain(invoke(model, obs0, env.configuration))
        a1 = plain(invoke(opp, obs1, env.configuration))
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "daily": daily,
        "terminal_state": snapshot({**final, "player": 0, "private": plain(env.state[0].observation.private)}),
    }


def summarize(rows):
    deltas = [r["delta_terminal_self"] for r in rows]
    base = [r["baseline"]["terminal_self"] for r in rows]
    cand = [r["candidate"]["terminal_self"] for r in rows]
    margins = [r["delta_margin"] for r in rows]
    return {
        "n": len(rows),
        "improved_vs_adaptive": sum(d > 0 for d in deltas),
        "worsened_vs_adaptive": sum(d < 0 for d in deltas),
        "same_vs_adaptive": sum(d == 0 for d in deltas),
        "adaptive_mean_terminal_self": statistics.mean(base),
        "world_lens_mean_terminal_self": statistics.mean(cand),
        "world_lens_median_terminal_self": statistics.median(cand),
        "world_lens_min_terminal_self": min(cand),
        "world_lens_max_terminal_self": max(cand),
        "mean_delta_vs_adaptive": statistics.mean(deltas),
        "mean_delta_margin_vs_adaptive": statistics.mean(margins),
        "world_lens_wins": sum(r["candidate"]["margin"] > 0 for r in rows),
    }


def main():
    rows = []
    for seed in SEEDS:
        baseline = run_one(seed, BASELINE, "adaptive")
        candidate = run_one(seed, CANDIDATE, "world_lens")
        row = {
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "delta_terminal_self": candidate["terminal_self"] - baseline["terminal_self"],
            "delta_margin": candidate["margin"] - baseline["margin"],
        }
        rows.append(row)
        print("WORLD_LENS_CASE " + json.dumps({
            "seed": seed,
            "adaptive": baseline["terminal_self"],
            "world_lens": candidate["terminal_self"],
            "delta": row["delta_terminal_self"],
            "world_lens_margin": candidate["margin"],
            "world_lens_terminal_state": candidate["terminal_state"],
        }, separators=(",", ":")))

    out = {
        "schema": "world-lens-agent-v0-fixed10",
        "objective": "terminal_self",
        "seat": 0,
        "opponent": "seyamalam_v21",
        "baseline": "adaptive_replay_contract_runtime_v0",
        "candidate": "world_lens_agent_v0",
        "seeds": SEEDS,
        "summary": summarize(rows),
        "cases": rows,
        "boundary": [
            "World Lens v0 is an independent rule-grounded prototype; it imports no replay program and no opponent policy.",
            "This is a first implementation test of the design philosophy, not a promotion test.",
            "A weak terminal result does not falsify World Lens as a design family; it evaluates only this v0 operationalization.",
        ],
    }
    Path("world_lens_agent_v0_fixed10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("WORLD_LENS_SUMMARY " + json.dumps(out["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
