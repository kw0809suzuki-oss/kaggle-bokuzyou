from copy import deepcopy

import astra_flow_independent_distilled_v0 as base


def reset_agent():
    if hasattr(base, "reset_agent"):
        base.reset_agent()


def _wheat_total(private):
    total = int((private.get("shed") or {}).get("WHEAT", 0) or 0)
    for inv in private.get("inventories", []) or []:
        total += int((inv or {}).get("WHEAT", 0) or 0)
    return total


def _unfed_animals(farm):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict) or not tile.get("animal"):
                continue
            if not tile.get("fed_today", False):
                n += 1
    return n


def agent(obs, configuration=None):
    action = deepcopy(base.agent(obs, configuration))
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    private = obs.get("private") or {}

    reserve = _unfed_animals(farm)
    wheat_owned = _wheat_total(private)

    filtered = []
    for order in list(action.get("market", []) or []):
        if not isinstance(order, list) or not order:
            filtered.append(order)
            continue

        op = order[0]
        if op == "SELL" and len(order) >= 3 and order[1] == "WHEAT":
            requested = int(order[2])
            sellable_surplus = max(0, wheat_owned - reserve)
            qty = min(requested, sellable_surplus)
            if qty > 0:
                filtered.append(["SELL", "WHEAT", qty])
                wheat_owned -= qty
            continue

        if op == "BUY_PRODUCT" and len(order) >= 3 and order[1] == "WHEAT":
            requested = int(order[2])
            shortage = max(0, reserve - wheat_owned)
            qty = min(requested, shortage)
            if qty > 0:
                filtered.append(["BUY_PRODUCT", "WHEAT", qty])
                wheat_owned += qty
            continue

        filtered.append(order)

    action["market"] = filtered
    return action
