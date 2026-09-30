#!/usr/bin/env python3
"""Adaptive Replay step96->97 Cash Emergence Probe v0.

Question:
    At the first observed candidate self-Cash divergence, does the same Replay
    SELL realize a different Cash return because the per-unit market quotes
    differ during execution?

Scope:
    Contract Runtime v0 only
    Seyamalam v21, subject seat0
    positive seed 93803004
    negative seed 93803005
    one transition only: step96 pre-state -> step97 post-state

Important:
    Kaggriculture SELL is executed per unit. A displayed pre-step price can be
    equal while later units in the same SELL order receive different quotes as
    market inventory changes. This probe reconstructs the exact subject SELL
    quote sequence from the pinned official market_price() function and verifies
    it against realized shed, HIRE, and Cash changes.

No new Guard, Recovery, policy, or downstream explanation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
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


def first_sell_order(action):
    for order in action.get("market", []) or []:
        if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
            return {
                "item": order[1],
                "requested": int(order[2]),
            }
    return None


def count_hires(action):
    return sum(
        1
        for order in (action.get("market", []) or [])
        if isinstance(order, list) and order and order[0] == "HIRE"
    )


def first_market_order(action):
    market = action.get("market", []) or []
    return market[0] if market else None


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

        pre = {
            "cash": float(obs0["farms"][0]["money"]),
            "shed": plain(obs0["private"]["shed"]),
            "market_inventory": plain(obs0["market"]["inventory"]),
            "market_prices": plain(obs0["market"]["prices"]),
            "hands": len(obs0["farms"][0].get("hands", []) or []),
            "hires_today": int(obs0["farms"][0].get("hires_today", 0) or 0),
        }

        env.step([a0, a1])

        obs97 = plain(shared(env, 0))
        post = {
            "cash": float(obs97["farms"][0]["money"]),
            "shed": plain(obs97["private"]["shed"]),
            "market_inventory": plain(obs97["market"]["inventory"]),
            "market_prices": plain(obs97["market"]["prices"]),
            "hands": len(obs97["farms"][0].get("hands", []) or []),
            "hires_today": int(obs97["farms"][0].get("hires_today", 0) or 0),
        }

        sell = first_sell_order(a0)
        hire_requested = count_hires(a0)

        if sell is None:
            raise RuntimeError("No subject SELL order at target step")

        item = sell["item"]
        requested = sell["requested"]

        subject_first_order = first_market_order(a0)
        opponent_first_order = first_market_order(a1)

        opponent_first_same_item_sell = (
            isinstance(opponent_first_order, list)
            and len(opponent_first_order) >= 3
            and opponent_first_order[0] == "SELL"
            and opponent_first_order[1] == item
        )

        if opponent_first_same_item_sell:
            raise RuntimeError(
                "Probe arithmetic assumes opponent first order does not sell "
                "the same item during the subject first SELL order."
            )

        pre_shed_item = int(pre["shed"].get(item, 0))
        post_shed_item = int(post["shed"].get(item, 0))
        realized_sell_units = pre_shed_item - post_shed_item

        if realized_sell_units < 0:
            raise RuntimeError("Observed shed item increased across SELL transition")

        start_inventory = int(pre["market_inventory"][item])
        quote_sequence = [
            int(kg.market_price(item, start_inventory + unit_index))
            for unit_index in range(realized_sell_units)
        ]
        reconstructed_sell_revenue = sum(quote_sequence)

        hire_costs = [
            int(kg._hire_cost(pre["hires_today"] + i))
            for i in range(hire_requested)
        ]

        realized_hires = post["hands"] - pre["hands"]
        all_requested_hires_realized = realized_hires == hire_requested

        reconstructed_cash = (
            pre["cash"]
            + reconstructed_sell_revenue
            - sum(hire_costs[:realized_hires])
        )

        return {
            "seed": seed,
            "step": TARGET_STEP,
            "subject_action": a0,
            "opponent_action": a1,
            "subject_first_market_order": subject_first_order,
            "opponent_first_market_order": opponent_first_order,
            "pre": pre,
            "post": post,
            "cash_delta": post["cash"] - pre["cash"],
            "sell_item": item,
            "sell_requested_units": requested,
            "sell_realized_units": realized_sell_units,
            "sell_all_requested_realized": realized_sell_units == requested,
            "pre_display_price": int(pre["market_prices"][item]),
            "sell_quote_sequence": quote_sequence,
            "reconstructed_sell_revenue": reconstructed_sell_revenue,
            "hire_requested": hire_requested,
            "realized_hires": realized_hires,
            "all_requested_hires_realized": all_requested_hires_realized,
            "hire_cost_sequence": hire_costs,
            "realized_hire_cost_total": sum(hire_costs[:realized_hires]),
            "reconstructed_post_cash": reconstructed_cash,
            "reconstructed_cash_matches_observed":
                abs(reconstructed_cash - post["cash"]) < 1e-9,
        }

    raise RuntimeError(f"step {TARGET_STEP} not reached for seed {seed}")


def main():
    runs = {label: run_transition(seed, label) for label, seed in SEEDS.items()}
    pos = runs["positive"]
    neg = runs["negative"]

    summary = {
        "subject_action_equal": pos["subject_action"] == neg["subject_action"],
        "opponent_action_equal": pos["opponent_action"] == neg["opponent_action"],
        "pre_cash_equal": pos["pre"]["cash"] == neg["pre"]["cash"],
        "pre_display_price_equal":
            pos["pre_display_price"] == neg["pre_display_price"],
        "positive_pre_display_price": pos["pre_display_price"],
        "negative_pre_display_price": neg["pre_display_price"],
        "positive_pre_inventory": pos["pre"]["market_inventory"][pos["sell_item"]],
        "negative_pre_inventory": neg["pre"]["market_inventory"][neg["sell_item"]],
        "sell_item_equal": pos["sell_item"] == neg["sell_item"],
        "sell_requested_units_equal":
            pos["sell_requested_units"] == neg["sell_requested_units"],
        "sell_realized_units_equal":
            pos["sell_realized_units"] == neg["sell_realized_units"],
        "positive_sell_realized_units": pos["sell_realized_units"],
        "negative_sell_realized_units": neg["sell_realized_units"],
        "positive_quote_sequence": pos["sell_quote_sequence"],
        "negative_quote_sequence": neg["sell_quote_sequence"],
        "positive_sell_revenue": pos["reconstructed_sell_revenue"],
        "negative_sell_revenue": neg["reconstructed_sell_revenue"],
        "sell_revenue_difference_positive_minus_negative":
            pos["reconstructed_sell_revenue"] - neg["reconstructed_sell_revenue"],
        "hire_requested_equal": pos["hire_requested"] == neg["hire_requested"],
        "realized_hires_equal": pos["realized_hires"] == neg["realized_hires"],
        "positive_realized_hires": pos["realized_hires"],
        "negative_realized_hires": neg["realized_hires"],
        "positive_realized_hire_cost_total": pos["realized_hire_cost_total"],
        "negative_realized_hire_cost_total": neg["realized_hire_cost_total"],
        "positive_post_cash": pos["post"]["cash"],
        "negative_post_cash": neg["post"]["cash"],
        "post_cash_difference_positive_minus_negative":
            pos["post"]["cash"] - neg["post"]["cash"],
        "positive_cash_delta": pos["cash_delta"],
        "negative_cash_delta": neg["cash_delta"],
        "cash_delta_difference_positive_minus_negative":
            pos["cash_delta"] - neg["cash_delta"],
        "positive_reconstruction_matches":
            pos["reconstructed_cash_matches_observed"],
        "negative_reconstruction_matches":
            neg["reconstructed_cash_matches_observed"],
        "cash_difference_equals_sell_revenue_difference":
            abs(
                (pos["post"]["cash"] - neg["post"]["cash"])
                - (
                    pos["reconstructed_sell_revenue"]
                    - neg["reconstructed_sell_revenue"]
                )
            ) < 1e-9,
    }

    print("CASH_96_97_SUMMARY " + json.dumps(summary, separators=(",", ":")))
    print("CASH_96_97_POSITIVE " + json.dumps(pos, separators=(",", ":")))
    print("CASH_96_97_NEGATIVE " + json.dumps(neg, separators=(",", ":")))

    out = {
        "schema": "adaptive-replay-96-97-cash-emergence-v0",
        "question": (
            "At the first observed self-Cash divergence, does the same Replay "
            "SELL realize a different Cash return because per-unit market "
            "quotes differ during execution?"
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
            "important_distinction": (
                "Displayed pre-step price equality does not imply equal "
                "per-unit SELL quote sequences because market inventory changes "
                "after each successful unit."
            ),
        },
    }

    Path("adaptive_replay_96_97_cash_emergence_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
