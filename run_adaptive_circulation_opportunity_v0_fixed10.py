#!/usr/bin/env python3
"""Fixed10 terminal A/B for Adaptive Circulation Opportunity v0."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "adaptive_circulation_runtime_v0.py"
CANDIDATE = ROOT / "adaptive_circulation_opportunity_v0.py"
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
    return env._Environment__get_shared_state(seat)["observation"]


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0 = plain(shared(env, 0))
        o1 = plain(shared(env, 1))
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
        baseline = run_one(seed, BASELINE, "base")
        candidate = run_one(seed, CANDIDATE, "cand")
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
    active = [r for r in rows if r["candidate"]["trigger_count"] > 0]
    summary = {
        "n": len(rows),
        "triggered_cases": len(active),
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
        "trigger_steps": [r["candidate"]["event"]["step"] for r in active if r["candidate"]["event"]],
    }

    out = {
        "schema": "adaptive-circulation-opportunity-v0-fixed10",
        "objective": "terminal_self",
        "baseline": "adaptive_circulation_runtime_v0",
        "candidate": "adaptive_circulation_opportunity_v0",
        "candidate_rule": "one-shot mature-crop harvest only when concurrent expansion labor exists and no HARVEST is already issued",
        "summary": summary,
        "cases": rows,
        "boundary": [
            "A positive result supports only this condition/transform pair.",
            "A negative result does not reject harvest prioritization in other contexts.",
            "No contract is promoted automatically by this run.",
        ],
    }
    Path("adaptive_circulation_opportunity_v0_fixed10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
