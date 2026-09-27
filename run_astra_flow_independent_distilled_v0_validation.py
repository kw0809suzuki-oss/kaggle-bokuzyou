#!/usr/bin/env python3
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent

MODELS = {
    "independent": ROOT / "astra_flow_independent_distilled_v0.py",
    "v0": ROOT / "astra_flow_terminal_model_v0.py",
    "frozen": ROOT / "relationship_surface_body_v0.py",
}
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def run_model(seed, label):
    model = load_module(MODELS[label], f"model_{label}_{seed}")
    opp = load_module(OPPONENT, f"opp_{label}_{seed}")
    if hasattr(model, "reset_agent"):
        model.reset_agent()
    if hasattr(opp, "reset_agent"):
        opp.reset_agent()

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    actions = []
    turns = []

    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)
        a0 = plain(model.agent(obs0))
        a1 = plain(opp.agent(obs1))
        actions.append(a0)
        turns.append({
            "step": int(obs0["step"]),
            "self_cash": float(obs0["farms"][0]["money"]),
            "opponent_cash": float(obs0["farms"][1]["money"]),
            "self_action": a0,
        })
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    return {
        "label": label,
        "turn_count": len(actions),
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "margin": float(final["farms"][0]["money"]) - float(final["farms"][1]["money"]),
        "actions": actions,
        "turns": turns,
    }


def main():
    seed = int(os.environ["SEED"])
    results = {label: run_model(seed, label) for label in ("independent", "v0", "frozen")}

    first_mismatch = None
    mismatches = 0
    for i, (a, b) in enumerate(zip(results["independent"]["actions"], results["v0"]["actions"])):
        if a != b:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "step": i,
                    "independent_action": a,
                    "v0_action": b,
                }

    summary = {
        "schema": "astra-flow-independent-distilled-v0-validation",
        "seed": seed,
        "results": [
            {
                "label": label,
                "turn_count": results[label]["turn_count"],
                "terminal_self": results[label]["terminal_self"],
                "terminal_opponent": results[label]["terminal_opponent"],
                "margin": results[label]["margin"],
            }
            for label in ("independent", "v0", "frozen")
        ],
        "independent_vs_v0": {
            "action_mismatch_count": mismatches,
            "first_action_mismatch": first_mismatch,
            "terminal_self_delta": results["independent"]["terminal_self"] - results["v0"]["terminal_self"],
        },
    }

    full = {
        "schema": "astra-flow-independent-distilled-v0-validation-full",
        "seed": seed,
        "summary": summary,
        "runs": {
            label: {
                "turns": results[label]["turns"],
                "terminal_self": results[label]["terminal_self"],
                "terminal_opponent": results[label]["terminal_opponent"],
                "margin": results[label]["margin"],
            }
            for label in ("independent", "v0", "frozen")
        },
    }

    Path(f"astra_flow_independent_seed{seed}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with gzip.open(f"astra_flow_independent_seed{seed}.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(full, fh, ensure_ascii=False, separators=(",", ":"))

    print("INDEPENDENT " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
