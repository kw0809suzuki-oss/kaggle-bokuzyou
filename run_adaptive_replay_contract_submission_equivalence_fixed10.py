#!/usr/bin/env python3
"""Exact fixed10 equivalence: source Contract Runtime v0 vs generated main.py."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "adaptive_replay_contract_runtime_v0.py"
SUBMISSION = ROOT / "main.py"
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


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}_{os.getpid()}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    trace = []
    while not env.done:
        a0 = plain(model.agent(plain(shared(env, 0))))
        a1 = plain(opp.agent(plain(shared(env, 1))))
        trace.append(a0)
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "last_guard_event": plain(getattr(model, "last_guard_event", None)),
        "action_count": len(trace),
        "action_sha256": digest(trace),
        "_trace": trace,
    }


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"step": i, "source": x, "submission": y}
    if len(a) != len(b):
        return {"step": min(len(a), len(b)), "source_len": len(a), "submission_len": len(b)}
    return None


def main():
    rows = []
    for seed in SEEDS:
        src = run_one(seed, SOURCE, "source")
        sub = run_one(seed, SUBMISSION, "submission")
        mismatch = first_mismatch(src["_trace"], sub["_trace"])
        row = {
            "seed": seed,
            "action_trace_equal": mismatch is None,
            "terminal_self_equal": src["terminal_self"] == sub["terminal_self"],
            "terminal_opponent_equal": src["terminal_opponent"] == sub["terminal_opponent"],
            "trigger_count_equal": src["trigger_count"] == sub["trigger_count"],
            "guard_event_equal": src["last_guard_event"] == sub["last_guard_event"],
            "first_action_mismatch": mismatch,
            "source": {k:v for k,v in src.items() if not k.startswith("_")},
            "submission": {k:v for k,v in sub.items() if not k.startswith("_")},
        }
        row["equivalent"] = all([
            row["action_trace_equal"],
            row["terminal_self_equal"],
            row["terminal_opponent_equal"],
            row["trigger_count_equal"],
            row["guard_event_equal"],
        ])
        rows.append(row)
        print("SUBMISSION_EQ_CASE " + json.dumps(row, separators=(",", ":")))

    summary = {
        "n": len(rows),
        "equivalent": sum(r["equivalent"] for r in rows),
        "not_equivalent": sum(not r["equivalent"] for r in rows),
        "action_trace_equal": sum(r["action_trace_equal"] for r in rows),
        "terminal_self_equal": sum(r["terminal_self_equal"] for r in rows),
        "guard_event_equal": sum(r["guard_event_equal"] for r in rows),
    }
    Path("adaptive_replay_contract_submission_equivalence_fixed10.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUBMISSION_EQ_SUMMARY " + json.dumps(summary, separators=(",", ":")))
    if summary["not_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
