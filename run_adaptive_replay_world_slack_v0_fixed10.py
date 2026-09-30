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
CANDIDATE = ROOT / "adaptive_replay_world_slack_v0.py"
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


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    events = []
    prev_triggers = 0
    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        a0 = plain(invoke(model, obs0, env.configuration))
        a1 = plain(invoke(opp, obs1, env.configuration))
        now = int(getattr(model, "slack_trigger_count", 0))
        if now > prev_triggers:
            events.append(plain(getattr(model, "last_slack_event", None)))
            prev_triggers = now
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "slack_trigger_count": int(getattr(model, "slack_trigger_count", 0)),
        "slack_events": events,
    }


def summarize(rows):
    deltas = [r["delta_terminal_self"] for r in rows]
    return {
        "n": len(rows),
        "improved": sum(d > 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "same": sum(d == 0 for d in deltas),
        "baseline_mean_terminal_self": statistics.mean(
            r["baseline"]["terminal_self"] for r in rows
        ),
        "candidate_mean_terminal_self": statistics.mean(
            r["candidate"]["terminal_self"] for r in rows
        ),
        "mean_delta_terminal_self": statistics.mean(deltas),
        "median_delta_terminal_self": statistics.median(deltas),
        "min_delta_terminal_self": min(deltas),
        "max_delta_terminal_self": max(deltas),
        "triggered_cases": sum(r["candidate"]["slack_trigger_count"] > 0 for r in rows),
        "total_triggers": sum(r["candidate"]["slack_trigger_count"] for r in rows),
    }


def main():
    rows = []
    for seed in SEEDS:
        baseline = run_one(seed, BASELINE, "baseline")
        candidate = run_one(seed, CANDIDATE, "candidate")
        row = {
            "seed": seed,
            "baseline": baseline,
            "candidate": candidate,
            "delta_terminal_self": candidate["terminal_self"] - baseline["terminal_self"],
            "delta_margin": candidate["margin"] - baseline["margin"],
        }
        rows.append(row)
        print("WORLD_SLACK_CASE " + json.dumps(row, separators=(",", ":")))

    out = {
        "schema": "adaptive-replay-world-slack-v0-fixed10",
        "objective": "terminal_self",
        "seat": 0,
        "opponent": "seyamalam_v21",
        "baseline": "adaptive_replay_contract_runtime_v0",
        "candidate": "adaptive_replay_world_slack_v0",
        "summary": summarize(rows),
        "cases": rows,
        "boundary": [
            "Replay action multiset is preserved.",
            "Only one existing SELL may move earlier when Current-World projection realizes more of replay's own non-SELL market intent.",
            "This is an experiment gate, not a promotion."
        ],
    }
    Path("adaptive_replay_world_slack_v0_fixed10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("WORLD_SLACK_SUMMARY " + json.dumps(out["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
