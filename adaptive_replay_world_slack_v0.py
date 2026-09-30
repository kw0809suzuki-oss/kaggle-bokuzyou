"""Adaptive Replay + World Slack v0.

Purpose:
Keep the proven Adaptive Replay Contract Runtime as the source of action.
Open exactly one narrow Current-World slack:

- never invent a new market operation;
- never remove an operation;
- never change quantity;
- only move ONE existing SELL earlier in the same market queue;
- only when a self-only Official-Rule projection shows that the move realizes
  strictly more of the replay's own non-SELL market intent.

This is an experiment, not a promoted model.
"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "adaptive_replay_contract_runtime_v0.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load(BASE, "_adaptive_replay_world_slack_base")

slack_trigger_count = 0
last_slack_event = None


def _plain(v: Any):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return [_plain(x) for x in v]
    return v


def _cfg(configuration, key: str, default):
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(key, default)
    return getattr(configuration, key, default)


def reset_agent():
    global slack_trigger_count, last_slack_event
    slack_trigger_count = 0
    last_slack_event = None
    if hasattr(base, "reset_agent"):
        base.reset_agent()


def _unit_price(op: str, item: str, market: dict):
    params = market.get("params")
    if op == "SELL" and item in rules.PRODUCTS:
        return rules.market_price(item, market["inventory"][item], params)
    if op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
        return rules.market_price(item, market["inventory"][item] - 1, params)
    if op == "BUY_SEED" and item in rules.CROPS:
        return int(rules.CROPS[item]["seed"])
    if op == "BUY_ANIMAL" and item in rules.ANIMALS:
        return int(rules.ANIMALS[item]["cost"])
    return None


def _simulate_market(obs, market_orders, configuration):
    """Self-only Official-rule projection of this market queue.

    It deliberately does NOT predict the opponent. Its only use is to ask:
    under the Current World already visible to us, does the replay queue itself
    fail to realize some of its own requested market intent?
    """
    raw = _plain(obs)
    player = int(raw["player"])
    farm = deepcopy(raw["farms"][player])
    private = deepcopy(raw["private"])
    market = deepcopy(raw["market"])

    board_size = int(_cfg(configuration, "boardSize", 10))
    hire_mult = int(_cfg(configuration, "farmHandCostMult", 1))
    shed_capacity = int(_cfg(configuration, "shedCapacity", 100))
    max_orders = max(1, int(_cfg(configuration, "maxMarketOrdersPerTurn", 10)))

    details = []
    for slot, order in enumerate(list(market_orders or [])[:max_orders]):
        parsed = rules._parse_order(order)
        if parsed is None:
            details.append({
                "slot": slot,
                "order": _plain(order),
                "type": "INVALID",
                "requested": 0,
                "succeeded": 0,
            })
            continue

        op = parsed["type"]
        if op == "HIRE":
            before = len(farm.get("hands", []) or [])
            rules._do_hire(farm, private, board_size, hire_mult)
            after = len(farm.get("hands", []) or [])
            succeeded = int(after > before)
            details.append({
                "slot": slot,
                "order": _plain(order),
                "type": op,
                "requested": 1,
                "succeeded": succeeded,
            })
            rules._refresh_prices(market)
            continue

        if op == "BUY_LAND":
            before = len(farm.get("unlocked_quadrants", []) or [])
            rules._do_buy_land(farm, board_size)
            after = len(farm.get("unlocked_quadrants", []) or [])
            succeeded = int(after > before)
            details.append({
                "slot": slot,
                "order": _plain(order),
                "type": op,
                "requested": 1,
                "succeeded": succeeded,
            })
            rules._refresh_prices(market)
            continue

        requested = int(parsed.get("remaining", 0) or 0)
        item = str(parsed.get("item"))
        succeeded = 0
        for _ in range(requested):
            price = _unit_price(op, item, market)
            if price is None:
                break
            ok = rules._commit_unit(
                op,
                item,
                price,
                farm,
                private,
                market,
                shed_capacity,
            )
            if not ok:
                break
            succeeded += 1

        details.append({
            "slot": slot,
            "order": _plain(order),
            "type": op,
            "requested": requested,
            "succeeded": succeeded,
        })
        rules._refresh_prices(market)

    non_sell_requested = sum(
        int(r["requested"]) for r in details
        if r["type"] not in ("SELL", "INVALID")
    )
    non_sell_succeeded = sum(
        int(r["succeeded"]) for r in details
        if r["type"] not in ("SELL", "INVALID")
    )
    sell_succeeded = sum(
        int(r["succeeded"]) for r in details if r["type"] == "SELL"
    )

    return {
        "details": details,
        "non_sell_requested": non_sell_requested,
        "non_sell_succeeded": non_sell_succeeded,
        "sell_succeeded": sell_succeeded,
        "end_cash": float(farm.get("money", 0) or 0),
    }


def _move_one_sell_earlier(orders, sell_index, insert_index):
    moved = [deepcopy(x) for x in orders]
    sell = moved.pop(sell_index)
    moved.insert(insert_index, sell)
    return moved


def _slack_candidate(obs, action, configuration):
    orders = list(action.get("market", []) or [])
    if len(orders) < 2:
        return None

    baseline = _simulate_market(obs, orders, configuration)
    if baseline["non_sell_succeeded"] >= baseline["non_sell_requested"]:
        return None

    candidates = []
    for sell_index, order in enumerate(orders):
        if not (
            isinstance(order, (list, tuple))
            and len(order) >= 1
            and order[0] == "SELL"
        ):
            continue
        for insert_index in range(sell_index):
            proposal = _move_one_sell_earlier(
                orders,
                sell_index=sell_index,
                insert_index=insert_index,
            )
            projected = _simulate_market(obs, proposal, configuration)

            # The slack exists to realize replay intent, not to trade away
            # already-realized sales for a different plan.
            if projected["sell_succeeded"] < baseline["sell_succeeded"]:
                continue
            gain = (
                projected["non_sell_succeeded"]
                - baseline["non_sell_succeeded"]
            )
            if gain <= 0:
                continue

            candidates.append({
                "gain": int(gain),
                "distance": int(sell_index - insert_index),
                "sell_index": int(sell_index),
                "insert_index": int(insert_index),
                "orders": proposal,
                "projection": projected,
            })

    if not candidates:
        return None

    # Minimal edit first among equally useful realizations.
    candidates.sort(
        key=lambda c: (
            -c["gain"],
            c["distance"],
            c["sell_index"],
            c["insert_index"],
        )
    )
    best = candidates[0]
    return {
        "baseline": baseline,
        "candidate": best,
    }


def agent(obs, configuration=None):
    global slack_trigger_count, last_slack_event

    action = deepcopy(base.agent(obs, configuration))
    found = _slack_candidate(obs, action, configuration)
    if found is None:
        return action

    before = _plain(action.get("market", []) or [])
    after = _plain(found["candidate"]["orders"])
    action["market"] = after

    slack_trigger_count += 1
    last_slack_event = {
        "step": int(obs.get("step", 0) or 0),
        "before": before,
        "after": after,
        "moved_sell_from": found["candidate"]["sell_index"],
        "moved_sell_to": found["candidate"]["insert_index"],
        "non_sell_realized_before": found["baseline"]["non_sell_succeeded"],
        "non_sell_requested": found["baseline"]["non_sell_requested"],
        "non_sell_realized_after": found["candidate"]["projection"]["non_sell_succeeded"],
        "sell_realized_before": found["baseline"]["sell_succeeded"],
        "sell_realized_after": found["candidate"]["projection"]["sell_succeeded"],
    }
    return action
