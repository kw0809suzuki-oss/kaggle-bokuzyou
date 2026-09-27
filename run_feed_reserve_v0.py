#!/usr/bin/env python3
import gzip
import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEEDS = [92802801, 92802802, 92802803]

OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
INDEPENDENT_COMMIT = "eff2cf1ae6eb03809d215248f9b05b79a2b4dfe9"
SELLESTA_COMMIT = "0a5ce7ef83211d6df5824a4e7e17f3d4a40c22e7"

BASELINE_TERMINAL = {
    (92802801, 0): 100919,
    (92802801, 1): 86181,
    (92802802, 0): 81930,
    (92802802, 1): 75802,
    (92802803, 0): 45995,
    (92802803, 1): 45995,
}

CANDIDATE_PATH = ROOT / "independent_feed_reserve_v0.py"
SELLESTA_PATH = ROOT / "opponents" / "sellesta_main_pinned.py"


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
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def call_agent(mod, obs):
    try:
        return plain(mod.agent(obs))
    except TypeError:
        return plain(mod.agent(obs, None))


def shared_obs(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def market_counts(action):
    out = {
        "buy_wheat_product": 0,
        "sell_wheat": 0,
    }
    for order in (action or {}).get("market", []) or []:
        if not isinstance(order, list) or len(order) < 3:
            continue
        if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT":
            out["buy_wheat_product"] += int(order[2])
        if order[0] == "SELL" and order[1] == "WHEAT":
            out["sell_wheat"] += int(order[2])
    return out


def run_one(seed, self_seat):
    tag = f"{seed}_{self_seat}"
    candidate = load_module(CANDIDATE_PATH, f"candidate_{tag}")
    sellesta = load_module(SELLESTA_PATH, f"sellesta_{tag}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    mods = [None, None]
    labels = [None, None]
    mods[self_seat] = candidate
    labels[self_seat] = "feed_reserve_v0"
    mods[1 - self_seat] = sellesta
    labels[1 - self_seat] = "sellesta"

    turns = []
    buy_wheat = 0
    sell_wheat = 0

    while not env.done:
        obs = [shared_obs(env, 0), shared_obs(env, 1)]
        actions = [call_agent(mods[0], obs[0]), call_agent(mods[1], obs[1])]
        c = market_counts(actions[self_seat])
        buy_wheat += c["buy_wheat_product"]
        sell_wheat += c["sell_wheat"]

        turns.append({
            "step": int(obs[self_seat].get("step", len(turns))),
            "day": int(obs[self_seat].get("day", 0)),
            "hour": int(obs[self_seat].get("hour", 0)),
            "pre_observation": obs[self_seat],
            "candidate_action": actions[self_seat],
            "opponent_action": actions[1 - self_seat],
        })
        env.step(actions)

    final = shared_obs(env, self_seat)
    self_cash = float(final["farms"][self_seat]["money"])
    opp_cash = float(final["farms"][1 - self_seat]["money"])
    baseline = BASELINE_TERMINAL[(seed, self_seat)]

    summary = {
        "seed": seed,
        "self_seat": self_seat,
        "terminal_self": self_cash,
        "baseline_terminal_self": baseline,
        "delta_self": self_cash - baseline,
        "terminal_opponent": opp_cash,
        "terminal_margin": self_cash - opp_cash,
        "requested_buy_product_wheat": buy_wheat,
        "requested_sell_wheat": sell_wheat,
        "turn_count": len(turns),
    }

    with gzip.open(ROOT / f"feed_reserve_v0_{seed}_seat{self_seat}.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump({
            "schema": "feed-reserve-v0-battle",
            "provenance": {
                "official_commit": OFFICIAL_COMMIT,
                "independent_commit": INDEPENDENT_COMMIT,
                "sellesta_commit": SELLESTA_COMMIT,
            },
            "summary": summary,
            "turns": turns,
        }, fh, ensure_ascii=False, separators=(",", ":"))

    return summary


def main():
    results = []
    for seed in SEEDS:
        for seat in (0, 1):
            results.append(run_one(seed, seat))

    deltas = [r["delta_self"] for r in results]
    out = {
        "schema": "feed-reserve-v0-summary",
        "results": results,
        "mean_delta_self": sum(deltas) / len(deltas),
        "improved": sum(1 for x in deltas if x > 0),
        "worse": sum(1 for x in deltas if x < 0),
        "same": sum(1 for x in deltas if x == 0),
    }
    Path("feed_reserve_v0_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("FEEDRESERVE " + json.dumps(out, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
