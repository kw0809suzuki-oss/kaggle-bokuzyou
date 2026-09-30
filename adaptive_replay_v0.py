"""Adaptive Replay v0.

Design:
    Replay decides the operating skeleton.
    Current World only guards a proven execution failure.

Frozen skeleton:
    decem_replay_distilled_157026_v0.py
    Episode 115303987 / source terminal self 157026.

Active adaptation:
    Step 24 HIRE Effect Guard only.
    Source requested HIRE x3 but source realized effect was HIRE x1.
    If the current state would allow >1 hires to succeed, cap this turn's
    HIRE orders to one. Otherwise leave the replay action unchanged.

No other replay differences are repaired.
No recovery policy is active in v0.
Terminal self decides adoption.
"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "decem_replay_distilled_157026_v0.py"

_SOURCE_HIRE_GUARD_STEP = 24
_SOURCE_REALIZED_HIRES = 1

spec = importlib.util.spec_from_file_location("_adaptive_replay_v0_body", REPLAY)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)

trigger_count = 0
last_guard_event = None


def _fib(n: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, n - 1)):
        a, b = b, a + b
    return a


def _count_feasible_hires(money: float, hires_today: int, requested: int, cost_mult: float) -> int:
    cash = float(money)
    done = 0
    for offset in range(requested):
        nth = int(hires_today) + offset + 1
        cost = float(cost_mult) * _fib(nth)
        if cash < cost:
            break
        cash -= cost
        done += 1
    return done


def reset_agent():
    global trigger_count, last_guard_event
    trigger_count = 0
    last_guard_event = None
    if hasattr(replay, "reset_agent"):
        replay.reset_agent()


def agent(obs, configuration=None):
    global trigger_count, last_guard_event

    action = deepcopy(replay.agent(obs, configuration))
    step = int(obs.get("step", 0) or 0)
    if step != _SOURCE_HIRE_GUARD_STEP:
        return action

    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    market_orders = list(action.get("market", []) or [])
    requested_hires = sum(
        1 for order in market_orders
        if isinstance(order, (list, tuple)) and len(order) >= 1 and order[0] == "HIRE"
    )
    if requested_hires <= _SOURCE_REALIZED_HIRES:
        return action

    money = float(farm.get("money", 0) or 0)
    hires_today = int(farm.get("hires_today", 0) or 0)
    cost_mult = 1.0
    if configuration is not None:
        if isinstance(configuration, dict):
            cost_mult = float(configuration.get("farmHandCostMult", 1) or 1)
        else:
            cost_mult = float(getattr(configuration, "farmHandCostMult", 1) or 1)

    feasible = _count_feasible_hires(money, hires_today, requested_hires, cost_mult)
    if feasible <= _SOURCE_REALIZED_HIRES:
        return action

    kept = 0
    guarded_market = []
    for order in market_orders:
        is_hire = isinstance(order, (list, tuple)) and len(order) >= 1 and order[0] == "HIRE"
        if not is_hire:
            guarded_market.append(order)
            continue
        if kept < _SOURCE_REALIZED_HIRES:
            guarded_market.append(order)
            kept += 1

    action["market"] = guarded_market
    trigger_count += 1
    last_guard_event = {
        "step": step,
        "money": money,
        "hires_today": hires_today,
        "requested_hires": requested_hires,
        "current_feasible_hires": feasible,
        "source_realized_hires": _SOURCE_REALIZED_HIRES,
    }
    return action
