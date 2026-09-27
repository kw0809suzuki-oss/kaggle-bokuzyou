from copy import deepcopy

import astra_flow_independent_distilled_v0 as base


def reset_agent():
    if hasattr(base, "reset_agent"):
        base.reset_agent()


def agent(obs, configuration=None):
    action = deepcopy(base.agent(obs, configuration))
    player = int(obs.get("player", 0) or 0)
    hires_today = int(obs["farms"][player].get("hires_today", 0) or 0)

    filtered = []
    planned_hires = hires_today
    for order in list(action.get("market", []) or []):
        if isinstance(order, list) and order and order[0] == "HIRE":
            if planned_hires >= 11:
                continue
            planned_hires += 1
        filtered.append(order)

    action["market"] = filtered
    return action
