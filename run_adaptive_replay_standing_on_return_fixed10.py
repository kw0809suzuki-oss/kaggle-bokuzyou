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
CANDIDATE = ROOT / "adaptive_replay_standing_on_return_v0.py"
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
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        a0 = plain(model.agent(obs0))
        a1 = plain(opp.agent(obs1))
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "step24_contract_trigger_count": int(
            getattr(model, "trigger_count",
                    getattr(getattr(model, "base", None), "trigger_count", 0))
        ),
        "standing_return_trigger_count": int(
            getattr(model, "standing_return_trigger_count", 0)
        ),
        "last_standing_return_event": plain(
            getattr(model, "last_standing_return_event", None)
        ),
    }


def summarize(rows):
    deltas = [r["delta_terminal_self"] for r in rows]
    base = [r["baseline"]["terminal_self"] for r in rows]
    cand = [r["candidate"]["terminal_self"] for r in rows]
    margin_deltas = [r["delta_margin"] for r in rows]
    return {
        "n": len(rows),
        "improved": sum(d > 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "same": sum(d == 0 for d in deltas),
        "activated_cases": sum(
            r["candidate"]["standing_return_trigger_count"] > 0 for r in rows
        ),
        "standing_return_triggers_total": sum(
            r["candidate"]["standing_return_trigger_count"] for r in rows
        ),
        "baseline_mean_terminal_self": statistics.mean(base),
        "candidate_mean_terminal_self": statistics.mean(cand),
        "mean_delta_terminal_self": statistics.mean(deltas),
        "median_delta_terminal_self": statistics.median(deltas),
        "min_delta_terminal_self": min(deltas),
        "max_delta_terminal_self": max(deltas),
        "mean_delta_margin": statistics.mean(margin_deltas),
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
            "delta_terminal_self":
                candidate["terminal_self"] - baseline["terminal_self"],
            "delta_margin":
                candidate["margin"] - baseline["margin"],
        }
        rows.append(row)
        print("STANDING_RETURN_CASE " + json.dumps(row, separators=(",", ":")))

    out = {
        "schema": "adaptive-replay-standing-on-return-v0-fixed10",
        "objective": "terminal_self",
        "seat": 0,
        "opponent": "seyamalam_v21",
        "baseline": "adaptive_replay_contract_runtime_v0",
        "candidate": "adaptive_replay_standing_on_return_v0",
        "seeds": SEEDS,
        "summary": summarize(rows),
        "cases": rows,
        "boundary": {
            "candidate_change":
                "only workers already standing on crop tiles with yield_units > 0 are changed to HARVEST",
            "market_orders_unchanged": True,
            "other_worker_actions_unchanged": True,
            "baseline_runtime_modified": False,
        },
    }
    Path("adaptive_replay_standing_on_return_v0_fixed10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("STANDING_RETURN_SUMMARY " +
          json.dumps(out["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
