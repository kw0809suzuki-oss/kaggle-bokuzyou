#!/usr/bin/env python3
"""Verify built single-file submission against the proven standing runtime."""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "adaptive_replay_contract_runtime_v0.py"
SUBMISSION = ROOT / "dist" / "free_slack_submission" / "main.py"
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


def invoke(mod, obs, cfg):
    try:
        return mod.agent(obs, cfg)
    except TypeError:
        return mod.agent(obs)


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    actions = []
    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        a0 = plain(invoke(model, obs0, env.configuration))
        a1 = plain(invoke(opp, obs1, env.configuration))
        actions.append(a0)
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "actions": actions,
        "surface_count": int(getattr(model, "surface_count", 0)),
        "distinct_alternative_count": int(
            getattr(model, "distinct_alternative_count", 0)
        ),
    }


def main():
    rows = []
    for seed in SEEDS:
        baseline = run_one(seed, BASELINE, "baseline")
        built = run_one(seed, SUBMISSION, "built")
        row = {
            "seed": seed,
            "action_trace_equal": baseline["actions"] == built["actions"],
            "baseline_terminal_self": baseline["terminal_self"],
            "built_terminal_self": built["terminal_self"],
            "delta_terminal_self": built["terminal_self"] - baseline["terminal_self"],
            "surface_count": built["surface_count"],
            "distinct_alternative_count": built["distinct_alternative_count"],
        }
        rows.append(row)
        print("BUILT_SUBMISSION_CASE " + json.dumps(row, separators=(",", ":")))

    summary = {
        "n": len(rows),
        "action_trace_equal_cases": sum(r["action_trace_equal"] for r in rows),
        "terminal_equal_cases": sum(r["delta_terminal_self"] == 0 for r in rows),
        "baseline_mean_terminal_self": statistics.mean(
            r["baseline_terminal_self"] for r in rows
        ),
        "built_mean_terminal_self": statistics.mean(
            r["built_terminal_self"] for r in rows
        ),
        "surface_count_total": sum(r["surface_count"] for r in rows),
        "distinct_alternative_count_total": sum(
            r["distinct_alternative_count"] for r in rows
        ),
    }

    assert summary["action_trace_equal_cases"] == len(rows), summary
    assert summary["terminal_equal_cases"] == len(rows), summary
    assert summary["surface_count_total"] == 719 * len(rows), summary
    assert summary["distinct_alternative_count_total"] > 0, summary

    out = {"schema": "built-free-slack-submission-fixed10", "summary": summary, "cases": rows}
    Path("dist/free_slack_submission/verification.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("BUILT_SUBMISSION_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
