#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "adaptive_replay_v0.py"
CONTRACT = ROOT / "adaptive_replay_contract_runtime_v0.py"
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

    self_trace = []
    opponent_trace = []

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        a0 = plain(model.agent(obs0))
        a1 = plain(opp.agent(obs1))
        self_trace.append(a0)
        opponent_trace.append(a1)
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "margin": float(final["farms"][0]["money"] - final["farms"][1]["money"]),
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "last_guard_event": plain(getattr(model, "last_guard_event", None)),
        "self_action_count": len(self_trace),
        "self_action_trace_sha256": digest(self_trace),
        "opponent_action_trace_sha256": digest(opponent_trace),
        "_self_trace": self_trace,
        "_opponent_trace": opponent_trace,
    }


def first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "reference": x, "contract": y}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "reference_length": len(a), "contract_length": len(b)}
    return None


def strip_private_trace(result):
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main():
    cases = []
    for seed in SEEDS:
        ref = run_one(seed, REFERENCE, "reference")
        con = run_one(seed, CONTRACT, "contract")

        self_mismatch = first_mismatch(ref["_self_trace"], con["_self_trace"])
        opp_mismatch = first_mismatch(ref["_opponent_trace"], con["_opponent_trace"])

        case = {
            "seed": seed,
            "self_action_trace_equal": self_mismatch is None,
            "opponent_action_trace_equal": opp_mismatch is None,
            "terminal_self_equal": ref["terminal_self"] == con["terminal_self"],
            "terminal_opponent_equal": ref["terminal_opponent"] == con["terminal_opponent"],
            "trigger_count_equal": ref["trigger_count"] == con["trigger_count"],
            "guard_event_equal": ref["last_guard_event"] == con["last_guard_event"],
            "first_self_action_mismatch": self_mismatch,
            "first_opponent_action_mismatch": opp_mismatch,
            "reference": strip_private_trace(ref),
            "contract": strip_private_trace(con),
        }
        case["equivalent"] = all([
            case["self_action_trace_equal"],
            case["opponent_action_trace_equal"],
            case["terminal_self_equal"],
            case["terminal_opponent_equal"],
            case["trigger_count_equal"],
            case["guard_event_equal"],
        ])
        cases.append(case)
        print("CONTRACT_EQUIVALENCE_CASE " + json.dumps(case, separators=(",", ":")))

    summary = {
        "n": len(cases),
        "equivalent": sum(c["equivalent"] for c in cases),
        "not_equivalent": sum(not c["equivalent"] for c in cases),
        "self_action_trace_equal": sum(c["self_action_trace_equal"] for c in cases),
        "opponent_action_trace_equal": sum(c["opponent_action_trace_equal"] for c in cases),
        "terminal_self_equal": sum(c["terminal_self_equal"] for c in cases),
        "guard_event_equal": sum(c["guard_event_equal"] for c in cases),
    }

    out = {
        "schema": "adaptive-replay-contract-runtime-v0-equivalence-fixed10",
        "reference": "adaptive_replay_v0.py",
        "candidate": "adaptive_replay_contract_runtime_v0.py",
        "seeds": SEEDS,
        "summary": summary,
        "cases": cases,
        "acceptance": "10/10 exact equivalence; otherwise fail abstraction",
    }

    Path("adaptive_replay_contract_runtime_v0_equivalence_fixed10.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("CONTRACT_EQUIVALENCE_SUMMARY " + json.dumps(summary, separators=(",", ":")))

    if summary["not_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
