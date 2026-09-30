#!/usr/bin/env python3
"""Adaptive Replay step96->97 Cash Emergence Probe v0.

Question:
    At the first observed candidate self-Cash divergence, does the same Replay
    SELL realize a different Cash return because current market quotes differ?

Scope:
    Contract Runtime v0 only
    Seyamalam v21, subject seat0
    positive seed 93803004
    negative seed 93803005
    one transition only: step96 pre-state -> step97 post-state

Observe:
    - exact subject/opponent actions
    - pre/post self Cash
    - pre shed and market prices/inventory
    - every realized market unit commit with quoted price
    - HIRE / BUY_LAND atomic Cash effects
    - successful SELL units and realized SELL revenue

No new Guard, Recovery, policy, or downstream explanation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_replay_contract_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = {
    "positive": 93803004,
    "negative": 93803005,
}
TARGET_STEP = 96


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


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def run_transition(seed: int, tag: str):
    model = load(MODEL, f"cash_probe_model_{tag}_{seed}_{os.getpid()}")
    opponent = load(OPPONENT, f"cash_probe_opp_{tag}_{seed}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        step = int(obs0.get("step", 0) or 0)

        a0 = plain(model.agent(obs0))
        a1 = plain(opponent.agent(obs1))

        if step != TARGET_STEP:
            env.step([a0, a1])
            continue

        live_farms = env.state[0].observation.farms
        farm_to_player = {id(live_farms[i]): i for i in range(len(live_farms))}
        commit_log = []
        atomic_log = []

        original_commit = kg._commit_unit
        original_hire = kg._do_hire
        original_land = kg._do_buy_land

        def wrapped_commit(op, item, price, farm, private, market, shed_capacity=100):
            player_id = farm_to_player.get(id(farm))
            before = {
                "cash": float(farm["money"]),
                "shed_item": int(private["shed"].get(item, 0)),
                "market_inventory": int(market["inventory"].get(item, 0)) if item in market["inventory"] else None,
            }
            ok = original_commit(op, item, price, farm, private, market, shed_capacity)
            after = {
                "cash": float(farm["money"]),
                "shed_item": int(private["shed"].get(item, 0)),
                "market_inventory": int(market["inventory"].get(item, 0)) if item in market["inventory"] else None,
            }
            commit_log.append({
                "player": player_id,
                "op": op,
                "item": item,
                "quote_price": float(price),
                "success": bool(ok),
                "before": before,
                "after": after,
                "cash_effect": after["cash"] - before["cash"],
            })
            return ok

        def wrapped_hire(farm, private, board_size, mult=kg.FARM_HAND_COST_MULT):
            player_id = farm_to_player.get(id(farm))
            before_cash = float(farm["money"])
            before_hands = len(farm["hands"])
            before_hires_today = int(farm["hires_today"])
            result = original_hire(farm, private, board_size, mult)
            atomic_log.append({
                "player": player_id,
                "op": "HIRE",
                "before_cash": before_cash,
                "after_cash": float(farm["money"]),
                "cash_effect": float(farm["money"]) - before_cash,
                "before_hands": before_hands,
                "after_hands": len(farm["hands"]),
                "before_hires_today": before_hires_today,
                "after_hires_today": int(farm["hires_today"]),
                "realized": len(farm["hands"]) > before_hands,
            })
            return result

        def wrapped_land(farm, board_size):
            player_id = farm_to_player.get(id(farm))
            before_cash = float(farm["money"])
            before_unlocked = list(farm["unlocked_quadrants"])
            result = original_land(farm, board_size)
            atomic_log.append({
                "player": player_id,
                "op": "BUY_LAND",
                "before_cash": before_cash,
                "after_cash": float(farm["money"]),
                "cash_effect": float(farm["money"]) - before_cash,
                "before_unlocked": before_unlocked,
                "after_unlocked": list(farm["unlocked_quadrants"]),
                "realized": len(farm["unlocked_quadrants"]) > len(before_unlocked),
            })
            return result

        pre = {
            "cash": float(obs0["farms"][0]["money"]),
            "shed": plain(obs0["private"]["shed"]),
            "seeds": plain(obs0["private"]["seeds"]),
            "market_inventory": plain(obs0["market"]["inventory"]),
            "market_prices": plain(obs0["market"]["prices"]),
            "hands": len(obs0["farms"][0].get("hands", []) or []),
            "hires_today": int(obs0["farms"][0].get("hires_today", 0) or 0),
        }

        kg._commit_unit = wrapped_commit
        kg._do_hire = wrapped_hire
        kg._do_buy_land = wrapped_land
        try:
            env.step([a0, a1])
        finally:
            kg._commit_unit = original_commit
            kg._do_hire = original_hire
            kg._do_buy_land = original_land

        obs97 = plain(shared(env, 0))
        post = {
            "cash": float(obs97["farms"][0]["money"]),
            "shed": plain(obs97["private"]["shed"]),
            "seeds": plain(obs97["private"]["seeds"]),
            "market_inventory": plain(obs97["market"]["inventory"]),
            "market_prices": plain(obs97["market"]["prices"]),
            "hands": len(obs97["farms"][0].get("hands", []) or []),
            "hires_today": int(obs97["farms"][0].get("hires_today", 0) or 0),
        }

        self_commits = [x for x in commit_log if x["player"] == 0]
        self_atomic = [x for x in atomic_log if x["player"] == 0]

        sell_units = defaultdict(int)
        sell_prices = defaultdict(list)
        sell_revenue = 0.0
        for row in self_commits:
            if row["op"] == "SELL" and row["success"]:
                sell_units[row["item"]] += 1
                sell_prices[row["item"]].append(row["quote_price"])
                sell_revenue += row["quote_price"]

        committed_cash_effect = sum(row["cash_effect"] for row in self_commits)
        atomic_cash_effect = sum(row["cash_effect"] for row in self_atomic)
        total_logged_market_cash_effect = committed_cash_effect + atomic_cash_effect

        return {
            "seed": seed,
            "step": TARGET_STEP,
            "subject_action": a0,
            "opponent_action": a1,
            "pre": pre,
            "post": post,
            "cash_delta": post["cash"] - pre["cash"],
            "self_commit_log": self_commits,
            "self_atomic_log": self_atomic,
            "successful_sell_units": dict(sell_units),
            "sell_quote_prices": {k: v for k, v in sell_prices.items()},
            "sell_revenue": sell_revenue,
            "logged_non_sell_cash_effect": total_logged_market_cash_effect - sell_revenue,
            "total_logged_market_cash_effect": total_logged_market_cash_effect,
            "logged_cash_matches_observed_delta":
                abs(total_logged_market_cash_effect - (post["cash"] - pre["cash"])) < 1e-9,
        }

    raise RuntimeError(f"step {TARGET_STEP} not reached for seed {seed}")


def main():
    runs = {label: run_transition(seed, label) for label, seed in SEEDS.items()}
    pos = runs["positive"]
    neg = runs["negative"]

    subject_action_equal = pos["subject_action"] == neg["subject_action"]
    opponent_action_equal = pos["opponent_action"] == neg["opponent_action"]
    pre_cash_equal = pos["pre"]["cash"] == neg["pre"]["cash"]
    realized_sell_units_equal = pos["successful_sell_units"] == neg["successful_sell_units"]
    non_sell_cash_effect_equal = (
        pos["logged_non_sell_cash_effect"] == neg["logged_non_sell_cash_effect"]
    )

    observed_cash_delta_difference = pos["cash_delta"] - neg["cash_delta"]
    sell_revenue_difference = pos["sell_revenue"] - neg["sell_revenue"]

    summary = {
        "subject_action_equal": subject_action_equal,
        "opponent_action_equal": opponent_action_equal,
        "pre_cash_equal": pre_cash_equal,
        "positive_pre_cash": pos["pre"]["cash"],
        "negative_pre_cash": neg["pre"]["cash"],
        "positive_post_cash": pos["post"]["cash"],
        "negative_post_cash": neg["post"]["cash"],
        "positive_cash_delta": pos["cash_delta"],
        "negative_cash_delta": neg["cash_delta"],
        "observed_cash_delta_difference_positive_minus_negative":
            observed_cash_delta_difference,
        "positive_successful_sell_units": pos["successful_sell_units"],
        "negative_successful_sell_units": neg["successful_sell_units"],
        "realized_sell_units_equal": realized_sell_units_equal,
        "positive_sell_quote_prices": pos["sell_quote_prices"],
        "negative_sell_quote_prices": neg["sell_quote_prices"],
        "positive_sell_revenue": pos["sell_revenue"],
        "negative_sell_revenue": neg["sell_revenue"],
        "sell_revenue_difference_positive_minus_negative": sell_revenue_difference,
        "positive_non_sell_cash_effect": pos["logged_non_sell_cash_effect"],
        "negative_non_sell_cash_effect": neg["logged_non_sell_cash_effect"],
        "non_sell_cash_effect_equal": non_sell_cash_effect_equal,
        "cash_delta_difference_equals_sell_revenue_difference":
            abs(observed_cash_delta_difference - sell_revenue_difference) < 1e-9,
        "positive_logged_cash_matches_observed_delta":
            pos["logged_cash_matches_observed_delta"],
        "negative_logged_cash_matches_observed_delta":
            neg["logged_cash_matches_observed_delta"],
    }

    print("CASH_96_97_SUMMARY " + json.dumps(summary, separators=(",", ":")))
    print("CASH_96_97_POSITIVE " + json.dumps(pos, separators=(",", ":")))
    print("CASH_96_97_NEGATIVE " + json.dumps(neg, separators=(",", ":")))

    out = {
        "schema": "adaptive-replay-96-97-cash-emergence-v0",
        "question": (
            "At the first observed self-Cash divergence, does the same Replay "
            "SELL realize a different Cash return because market quotes differ?"
        ),
        "engine_commit": "d7729da06cc1382eb742d6980dc3180aa85caa28",
        "world": "Seyamalam v21 / subject seat0",
        "summary": summary,
        "runs": runs,
        "boundary": {
            "one_transition_only": "step96 pre-state -> step97 post-state",
            "no_new_guard": True,
            "no_recovery": True,
            "no_downstream_causal_claim": True,
            "interpretation_rule": (
                "Only if actions and realized SELL units are equal, non-SELL "
                "Cash effects are equal, and Cash-delta difference equals "
                "SELL-revenue difference may this probe close the "
                "Price -> same SELL -> Cash link."
            ),
        },
    }

    Path("adaptive_replay_96_97_cash_emergence_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
