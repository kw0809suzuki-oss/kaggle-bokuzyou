"""Independent Day0 MELON12 Candidate v1.

Baseline -> v1 intervention:
- Day0 h0: add six MELON seeds to the existing MELON seed order.
- Day0 h20/h22: replace the existing five WHEAT PLANT actions with MELON PLANT.
- No other rescue logic.

v0 -> v1 difference:
- exactly one additional MELON seed (+80 Cash at the official seed price).

This file only defines the intervention. Strength is judged by Official World battles.
"""
from copy import deepcopy

import astra_flow_independent_distilled_v0 as baseline


def reset_agent():
    if hasattr(baseline, "reset_agent"):
        baseline.reset_agent()


def agent(obs, configuration=None):
    action = deepcopy(baseline.agent(obs, configuration))
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)

    if day == 0 and hour == 0:
        market = list(action.get("market", []) or [])
        for order in market:
            if (
                isinstance(order, list)
                and len(order) >= 3
                and order[0] == "BUY_SEED"
                and order[1] == "MELON"
            ):
                order[2] = int(order[2]) + 6
                break
        action["market"] = market

    if day == 0 and hour in (20, 22):
        if action.get("farmer") == ["PLANT", "WHEAT"]:
            action["farmer"] = ["PLANT", "MELON"]

        hands = []
        for hand_action in list(action.get("hands", []) or []):
            if hand_action == ["PLANT", "WHEAT"]:
                hands.append(["PLANT", "MELON"])
            else:
                hands.append(hand_action)
        action["hands"] = hands

    return action
