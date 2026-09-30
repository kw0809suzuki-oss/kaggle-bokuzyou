#!/usr/bin/env python3
"""World Gate + fixed10 terminal A/B for Intent Detour v0.

Gate purpose:
    Validate implementation only.
    It does NOT judge whether the candidate is promising.

If every fixed10 case satisfies the intended comparison:
    HARVEST WHEAT3 -> resume actor11 intent -> Official STRAWBERRY at [5,7]
    while other workers/market are untouched by the override layer,
then run terminal A/B unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "adaptive_circulation_runtime_v0.py"
CANDIDATE = ROOT / "adaptive_circulation_intent_detour_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]
ACTOR_INDEX = 11
TARGET = [5, 7]


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


def inv(obs, actor):
    xs = obs.get("private", {}).get("inventories", []) or []
    if actor < len(xs) and isinstance(xs[actor], dict):
        return {k: float(v) for k, v in xs[actor].items() if isinstance(v, (int, float)) and v}
    return {}


def tile_at(obs, pos):
    farm = obs["farms"][0]
    x, y = pos
    return plain(farm["tiles"][y][x])


def run_gate(seed):
    model = load(CANDIDATE, f"gate_candidate_{seed}")
    opp = load(OPPONENT, f"gate_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    pre252_wheat = None
    post252_wheat = None
    target_seen = None
    trace = []

    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        step = int(o0["step"])
        p = positions(o0["farms"][0])
        actor_pos = p[ACTOR_INDEX] if len(p) > ACTOR_INDEX else None

        a0 = plain(model.agent(deepcopy(o0), env.configuration))
        a1 = plain(opp.agent(deepcopy(o1)))

        if 252 <= step <= 258:
            trace.append({
                "step": step,
                "actor11_position": actor_pos,
                "actor11_inventory": inv(o0, ACTOR_INDEX),
                "actor11_action": (
                    [a0.get("farmer", ["PASS"])] + list(a0.get("hands", []) or [])
                )[ACTOR_INDEX],
                "target_tile": tile_at(o0, TARGET),
            })

        if step == 252:
            pre252_wheat = float(inv(o0, ACTOR_INDEX).get("WHEAT", 0))
            env.step([a0, a1])
            post = shared(env, 0)
            post252_wheat = float(inv(post, ACTOR_INDEX).get("WHEAT", 0))
            continue

        env.step([a0, a1])

        if step >= 258:
            post = shared(env, 0) if not env.done else plain(env.state[0].observation)
            target_seen = tile_at(post, TARGET)
            break

    events = plain(getattr(model, "event_log", []))
    other_ok = all(
        e.get("other_workers_and_market_unchanged", True)
        for e in events
        if e.get("kind") == "override"
    )
    target_ok = (
        isinstance(target_seen, dict)
        and target_seen.get("kind") == "PLANT"
        and target_seen.get("crop") == "STRAWBERRY"
    )
    harvest_delta = None
    if pre252_wheat is not None and post252_wheat is not None:
        harvest_delta = post252_wheat - pre252_wheat

    valid = all([
        int(getattr(model, "trigger_count", 0)) == 1,
        int(getattr(model, "completed_count", 0)) == 1,
        getattr(model, "failure_reason", None) is None,
        harvest_delta == 3.0,
        other_ok,
        target_ok,
    ])

    return {
        "seed": seed,
        "valid": valid,
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "completed_count": int(getattr(model, "completed_count", 0)),
        "failure_reason": plain(getattr(model, "failure_reason", None)),
        "harvest_wheat_delta": harvest_delta,
        "other_workers_and_market_unchanged": other_ok,
        "target_strawberry_observed": target_ok,
        "target_tile": target_seen,
        "events": events,
        "trace": trace,
    }


def run_terminal(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        a0 = plain(model.agent(deepcopy(o0), env.configuration))
        a1 = plain(opp.agent(deepcopy(o1)))
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "completed_count": int(getattr(model, "completed_count", 0)),
        "failure_reason": plain(getattr(model, "failure_reason", None)),
        "events": plain(getattr(model, "event_log", [])),
    }


def main():
    gates = []
    for seed in SEEDS:
        row = run_gate(seed)
        gates.append(row)
        print("GATE " + json.dumps({
            "seed": seed,
            "valid": row["valid"],
            "harvest_wheat_delta": row["harvest_wheat_delta"],
            "completed_count": row["completed_count"],
            "failure_reason": row["failure_reason"],
            "other_workers_and_market_unchanged": row["other_workers_and_market_unchanged"],
            "target_strawberry_observed": row["target_strawberry_observed"],
        }, ensure_ascii=False, separators=(",", ":")))

    gate_pass = all(r["valid"] for r in gates)
    out = {
        "schema": "adaptive-circulation-intent-detour-v0",
        "gate_role": "implementation validity only",
        "gate_pass": gate_pass,
        "gates": gates,
        "terminal": None,
        "boundary": [
            "The retained intent is only actor11 establishing STRAWBERRY at [5,7].",
            "Post-plant WATER/care is not retained.",
            "Detour side effects such as carry changes and later timing shifts are part of the candidate cost/benefit.",
            "Adoption is judged by terminal_self only after a valid gate.",
        ],
    }

    if not gate_pass:
        Path("adaptive_circulation_intent_detour_v0_result.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("GATE_SUMMARY " + json.dumps({"pass": False}, separators=(",", ":")))
        raise SystemExit(2)

    rows = []
    for seed in SEEDS:
        baseline = run_terminal(seed, BASELINE, "baseline")
        candidate = run_terminal(seed, CANDIDATE, "candidate")
        row = {
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "delta_terminal_self": candidate["terminal_self"] - baseline["terminal_self"],
            "delta_margin": candidate["margin"] - baseline["margin"],
        }
        rows.append(row)
        print("CASE " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    deltas = [r["delta_terminal_self"] for r in rows]
    summary = {
        "n": len(rows),
        "improved": sum(d > 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "same": sum(d == 0 for d in deltas),
        "baseline_mean_terminal_self": statistics.mean(r["baseline"]["terminal_self"] for r in rows),
        "candidate_mean_terminal_self": statistics.mean(r["candidate"]["terminal_self"] for r in rows),
        "mean_delta_terminal_self": statistics.mean(deltas),
        "median_delta_terminal_self": statistics.median(deltas),
        "min_delta_terminal_self": min(deltas),
        "max_delta_terminal_self": max(deltas),
        "mean_delta_margin": statistics.mean(r["delta_margin"] for r in rows),
    }
    out["terminal"] = {"summary": summary, "cases": rows}
    Path("adaptive_circulation_intent_detour_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
