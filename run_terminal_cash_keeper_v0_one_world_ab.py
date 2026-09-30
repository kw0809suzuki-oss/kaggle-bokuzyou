#!/usr/bin/env python3
"""One-world terminal-cash A/B.

Current Question:
    On the exact smoke World (seed7391, seat0, Seyamalam v21), what terminal
    cash does Terminal Cash Keeper v0 leave compared with the current
    Adaptive Replay Contract Runtime v0?

Boundary:
    One seed only. No strength generalization and no cause analysis.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
KEEPER = ROOT / "terminal_cash_keeper_v0.py"
ADAPTIVE = ROOT / "adaptive_replay_contract_runtime_v0.py"
OPPONENT = ROOT / "opponents" / "seyamalam_v21.py"

SEED = 7391


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


def run_one(model_path, tag):
    model = load(model_path, f"{tag}_model")
    opp = load(OPPONENT, f"{tag}_opponent")
    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    turns = 0
    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        try:
            a0 = plain(model.agent(obs0, env.configuration))
        except TypeError:
            a0 = plain(model.agent(obs0))
        a1 = plain(opp.agent(obs1))
        env.step([a0, a1])
        turns += 1

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][0]["money"])
    opp_cash = float(final["farms"][1]["money"])
    return {
        "turns": turns,
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
    }


def main():
    keeper = run_one(KEEPER, "keeper")
    adaptive = run_one(ADAPTIVE, "adaptive")

    result = {
        "schema": "terminal-cash-keeper-v0-one-world-ab",
        "seed": SEED,
        "seat": 0,
        "opponent": "Seyamalam pinned v21",
        "current_question": "terminal cash on the same smoke World",
        "keeper": keeper,
        "adaptive_replay_contract_runtime_v0": adaptive,
        "delta_keeper_minus_adaptive": {
            "terminal_self": keeper["terminal_self"] - adaptive["terminal_self"],
            "margin": keeper["margin"] - adaptive["margin"],
        },
        "boundary": "One seed only; no general strength conclusion and no cause analysis.",
    }

    Path("terminal_cash_keeper_v0_one_world_ab_seed7391.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
