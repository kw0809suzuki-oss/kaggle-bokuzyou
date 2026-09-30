#!/usr/bin/env python3
"""Adaptive Replay terminal-self distribution gate v0.

Purpose:
    Observe broad terminal self distribution before any new internal analysis.

Scope:
    Contract Runtime v0.1 behavior only.
    Opponent: Seyamalam v21.
    Fresh 10 seeds x subject seat0/seat1 = 20 worlds.
    No baseline comparison, no internal divergence tracing.

Report:
    mean, median, min, p10, p25, lower five,
    count below 50k and below 25k (transparent operational counters only).
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_replay_contract_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    93901001, 93901002, 93901003, 93901004, 93901005,
    93901006, 93901007, 93901008, 93901009, 93901010,
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


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def percentile_nearest(values, q):
    xs = sorted(values)
    if not xs:
        return None
    idx = round((len(xs) - 1) * q)
    return xs[idx]


def run_one(seed: int, subject_seat: int):
    model = load(MODEL, f"terminal_dist_model_{seed}_{subject_seat}_{os.getpid()}")
    opp = load(OPPONENT, f"terminal_dist_opp_{seed}_{subject_seat}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        if subject_seat == 0:
            a0 = plain(model.agent(obs0))
            a1 = plain(opp.agent(obs1))
        else:
            a0 = plain(opp.agent(obs0))
            a1 = plain(model.agent(obs1))
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_money = float(final["farms"][subject_seat]["money"])
    opp_money = float(final["farms"][1-subject_seat]["money"])
    return {
        "seed": seed,
        "subject_seat": subject_seat,
        "terminal_self": self_money,
        "terminal_opponent": opp_money,
        "margin": self_money - opp_money,
        "guard_trigger_count": int(getattr(model, "trigger_count", 0)),
    }


def summarize(cases):
    vals = [c["terminal_self"] for c in cases]
    ordered = sorted(cases, key=lambda c: c["terminal_self"])
    return {
        "n": len(vals),
        "mean_terminal_self": statistics.mean(vals),
        "median_terminal_self": statistics.median(vals),
        "min_terminal_self": min(vals),
        "max_terminal_self": max(vals),
        "p10_terminal_self_nearest": percentile_nearest(vals, 0.10),
        "p25_terminal_self_nearest": percentile_nearest(vals, 0.25),
        "below_50000_count": sum(v < 50000 for v in vals),
        "below_25000_count": sum(v < 25000 for v in vals),
        "lower_five": [
            {
                "seed": c["seed"],
                "seat": c["subject_seat"],
                "terminal_self": c["terminal_self"],
                "terminal_opponent": c["terminal_opponent"],
                "margin": c["margin"],
                "guard_trigger_count": c["guard_trigger_count"],
            }
            for c in ordered[:5]
        ],
    }


def main():
    cases = []
    for seed in SEEDS:
        for seat in (0, 1):
            case = run_one(seed, seat)
            cases.append(case)
            print("TERMINAL_DISTRIBUTION_CASE " + json.dumps(case, separators=(",", ":")))

    overall = summarize(cases)
    by_seat = {
        "seat0": summarize([c for c in cases if c["subject_seat"] == 0]),
        "seat1": summarize([c for c in cases if c["subject_seat"] == 1]),
    }

    out = {
        "schema": "adaptive-replay-terminal-distribution-v0",
        "model": "adaptive_replay_contract_runtime_v0.py",
        "opponent": "Seyamalam v21",
        "seeds": SEEDS,
        "world_count": len(cases),
        "summary": overall,
        "by_seat": by_seat,
        "cases": cases,
        "boundary": {
            "terminal_self_only_gate": True,
            "no_baseline_comparison": True,
            "no_internal_divergence_analysis": True,
            "threshold_note": (
                "below_50000_count and below_25000_count are transparent "
                "operational counters, not promoted success criteria."
            ),
            "next_rule": (
                "Only if clearly low worlds appear should a later probe enter "
                "those specific worlds and search for first Effect divergence."
            ),
        },
    }

    Path("adaptive_replay_terminal_distribution_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("TERMINAL_DISTRIBUTION_SUMMARY " + json.dumps({
        "overall": overall,
        "by_seat": by_seat,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
