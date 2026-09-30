#!/usr/bin/env python3
"""Fixed10 WATER / PASS / HARVEST comparison for the first opportunity."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
WATER = ROOT / "adaptive_circulation_runtime_v0.py"
PASS = ROOT / "adaptive_circulation_step250_pass_v0.py"
HARVEST = ROOT / "adaptive_circulation_opportunity_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]


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


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "event": plain(getattr(model, "last_event", None)),
    }


def main():
    rows = []
    for seed in SEEDS:
        water = run_one(seed, WATER, "water")
        passed = run_one(seed, PASS, "pass")
        harvest = run_one(seed, HARVEST, "harvest")

        row = {
            "seed": seed,
            "water": water,
            "pass": passed,
            "harvest": harvest,
            "delta_pass_minus_water": passed["terminal_self"] - water["terminal_self"],
            "delta_harvest_minus_water": harvest["terminal_self"] - water["terminal_self"],
            "delta_harvest_minus_pass": harvest["terminal_self"] - passed["terminal_self"],
        }
        rows.append(row)
        print("CASE " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    dpw = [r["delta_pass_minus_water"] for r in rows]
    dhw = [r["delta_harvest_minus_water"] for r in rows]
    dhp = [r["delta_harvest_minus_pass"] for r in rows]

    summary = {
        "n": len(rows),
        "triggered_pass_cases": sum(r["pass"]["trigger_count"] > 0 for r in rows),
        "triggered_harvest_cases": sum(r["harvest"]["trigger_count"] > 0 for r in rows),
        "pass_vs_water": {
            "improved": sum(x > 0 for x in dpw),
            "worsened": sum(x < 0 for x in dpw),
            "same": sum(x == 0 for x in dpw),
            "mean": statistics.mean(dpw),
            "median": statistics.median(dpw),
            "min": min(dpw),
            "max": max(dpw),
        },
        "harvest_vs_water": {
            "improved": sum(x > 0 for x in dhw),
            "worsened": sum(x < 0 for x in dhw),
            "same": sum(x == 0 for x in dhw),
            "mean": statistics.mean(dhw),
            "median": statistics.median(dhw),
            "min": min(dhw),
            "max": max(dhw),
        },
        "harvest_vs_pass": {
            "improved": sum(x > 0 for x in dhp),
            "worsened": sum(x < 0 for x in dhp),
            "same": sum(x == 0 for x in dhp),
            "mean": statistics.mean(dhp),
            "median": statistics.median(dhp),
            "min": min(dhp),
            "max": max(dhp),
        },
    }

    out = {
        "schema": "adaptive-circulation-step250-pass-v0",
        "objective": "terminal_self",
        "comparison": {
            "water": "existing runtime action",
            "pass": "same selected actor changed once to PASS",
            "harvest": "same selected actor changed once to HARVEST",
        },
        "summary": summary,
        "cases": rows,
        "boundary": [
            "PASS is also a world-changing intervention.",
            "HARVEST-PASS is not pure crop value; downstream route/world effects remain included.",
            "This probe separates stopping WATER from using the turn for HARVEST at this one observed opportunity.",
            "No state rule or contract is promoted by this run alone.",
        ],
    }
    Path("adaptive_circulation_step250_pass_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
