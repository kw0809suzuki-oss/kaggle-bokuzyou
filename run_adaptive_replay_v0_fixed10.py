#!/usr/bin/env python3
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "decem_replay_distilled_157026_v0.py"
CANDIDATE = ROOT / "adaptive_replay_v0.py"
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


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    step24 = {"pre": None, "issued_action": None, "post": None}

    while not env.done:
        obs0 = plain(shared(env, 0))
        step = int(obs0["step"])
        a0 = plain(model.agent(obs0))
        a1 = plain(opp.agent(shared(env, 1)))

        if step == 24:
            step24["pre"] = {
                "money": float(obs0["farms"][0]["money"]),
                "hands": len(obs0["farms"][0]["hands"]),
                "hires_today": int(obs0["farms"][0]["hires_today"]),
            }
            step24["issued_action"] = a0

        env.step([a0, a1])

        if step == 24 and not env.done:
            post = plain(shared(env, 0))
            step24["post"] = {
                "money": float(post["farms"][0]["money"]),
                "hands": len(post["farms"][0]["hands"]),
                "hires_today": int(post["farms"][0]["hires_today"]),
            }

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "guard_trigger_count": int(getattr(model, "trigger_count", 0)),
        "guard_event": plain(getattr(model, "last_guard_event", None)),
        "step24": step24,
    }


def summarize(rows):
    deltas = [r["delta_terminal_self"] for r in rows]
    base = [r["baseline"]["terminal_self"] for r in rows]
    cand = [r["candidate"]["terminal_self"] for r in rows]
    margins = [r["candidate"]["margin"] - r["baseline"]["margin"] for r in rows]
    return {
        "n": len(rows),
        "improved": sum(d > 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "same": sum(d == 0 for d in deltas),
        "guard_triggered_cases": sum(r["candidate"]["guard_trigger_count"] > 0 for r in rows),
        "baseline_mean_terminal_self": statistics.mean(base),
        "candidate_mean_terminal_self": statistics.mean(cand),
        "mean_delta_terminal_self": statistics.mean(deltas),
        "median_delta_terminal_self": statistics.median(deltas),
        "min_delta_terminal_self": min(deltas),
        "max_delta_terminal_self": max(deltas),
        "mean_delta_margin": statistics.mean(margins),
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
        print("ADAPTIVE_REPLAY_CASE " + json.dumps(row, separators=(",", ":")))

    out = {
        "schema": "adaptive-replay-v0-fixed10-gate",
        "objective": "terminal_self",
        "seat": 0,
        "opponent": "seyamalam_v21",
        "baseline": "decem_replay_distilled_157026_v0",
        "candidate": "adaptive_replay_v0",
        "seeds": SEEDS,
        "summary": summarize(rows),
        "cases": rows,
        "boundary": {
            "active_guard": "step24 HIRE realized-effect guard only",
            "closed_not_included": "step85 STRAWBERRY restoration (delta terminal 0/0)",
            "observer": "diagnostic only; no automatic recovery",
        },
    }
    Path("adaptive_replay_v0_fixed10_gate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("ADAPTIVE_REPLAY_SUMMARY " + json.dumps(out["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
