#!/usr/bin/env python3
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
SOURCE_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    raise TypeError(type(x).__name__)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def market_totals(action):
    totals = defaultdict(int)
    for order in action.get("market", []) or []:
        if isinstance(order, list) and len(order) >= 3:
            op, item, qty = str(order[0]), str(order[1]), int(order[2])
            totals[(op, item)] += qty
    return {f"{op}:{item}": qty for (op, item), qty in sorted(totals.items())}


def sell_totals(action):
    out = {}
    for key, qty in market_totals(action).items():
        if key.startswith("SELL:"):
            out[key.split(":", 1)[1]] = qty
    return out


def dict_delta(a, b):
    keys = sorted(set(a) | set(b))
    return {k: b.get(k, 0) - a.get(k, 0) for k in keys if b.get(k, 0) != a.get(k, 0)}


def snapshot(obs):
    f0 = obs["farms"][0]
    f1 = obs["farms"][1]
    market = obs.get("market", {}) or {}
    private = obs.get("private", {}) or {}
    return {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "self_cash": float(f0.get("money", 0) or 0),
        "opponent_cash": float(f1.get("money", 0) or 0),
        "opponent_minus_self_cash": float(f1.get("money", 0) or 0) - float(f0.get("money", 0) or 0),
        "market_prices": plain(market.get("prices", {}) or {}),
        "market_inventory": plain(market.get("inventory", {}) or {}),
        "self_shed": plain(private.get("shed", {}) or {}),
    }


def run_pair(seed):
    v7 = load_module(ROOT / "teacher_candidates" / "candidate_v7_public_v18.py", f"v7_{seed}")
    v19 = load_module(ROOT / "teacher_candidates" / "candidate_v19_wool_floor.py", f"v19_{seed}")
    opp7 = load_module(ROOT / "teacher_candidates" / "main_v21_a.py", f"opp7_{seed}")
    opp19 = load_module(ROOT / "teacher_candidates" / "main_v21_b.py", f"opp19_{seed}")

    env7 = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env19 = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env7.reset(num_agents=2)
    env19.reset(num_agents=2)

    first = None

    while not env7.done and not env19.done:
        obs70 = shared_obs(env7, 0)
        obs71 = shared_obs(env7, 1)
        obs190 = shared_obs(env19, 0)
        obs191 = shared_obs(env19, 1)

        a7 = plain(v7.agent(obs70))
        a19 = plain(v19.agent(obs190))
        o7 = plain(opp7.agent(obs71))
        o19 = plain(opp19.agent(obs191))

        if first is None and a7 != a19:
            s7 = snapshot(obs70)
            s19 = snapshot(obs190)
            sell7 = sell_totals(a7)
            sell19 = sell_totals(a19)
            first = {
                "state_equal_before_action": s7 == s19,
                "state_v7": s7,
                "state_v19": s19,
                "v7_action": a7,
                "v19_action": a19,
                "v7_sell_totals": sell7,
                "v19_sell_totals": sell19,
                "sell_delta_v19_minus_v7": dict_delta(sell7, sell19),
                "market_delta_v19_minus_v7": dict_delta(market_totals(a7), market_totals(a19)),
            }

        env7.step([a7, o7])
        env19.step([a19, o19])

    f7 = env7.state[0].observation
    f19 = env19.state[0].observation

    return {
        "v7_terminal": float(f7["farms"][0]["money"]),
        "v19_terminal": float(f19["farms"][0]["money"]),
        "delta_v19_minus_v7": float(f19["farms"][0]["money"]) - float(f7["farms"][0]["money"]),
        "first_divergence": first,
    }


def main():
    seed = int(os.environ["SEED"])
    result = {
        "schema": "v7-v19-first-divergence-diagnostic-v0",
        "seed": seed,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "source_commit": SOURCE_COMMIT,
            "opponent": "main.py V21",
            "seat": "self seat 0",
        },
        **run_pair(seed),
    }

    out = Path(f"v7_v19_first_divergence_seed{seed}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FIRSTDIV " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
