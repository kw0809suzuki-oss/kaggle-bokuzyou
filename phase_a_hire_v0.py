import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))

import whole_flow_control_agent as base


def fib_hire_cost(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def agent(obs):
    action = copy.deepcopy(base.agent(obs))
    day = obs["day"]
    if not (7 <= day <= 10):
        return action

    me = obs["farms"][obs["player"]]
    hands = len(me.get("hands", []))
    hires_today = me.get("hires_today", 0)
    money = me.get("money", 0)

    # Phase A v0: use already-available cash to increase productive labor.
    # Do not predict sell proceeds; do not change land, seeds, animals, or unit actions.
    target_hands = 12
    reserve = 650

    market = list(action.get("market", []))
    existing_hires = sum(1 for order in market if order and order[0] == "HIRE")
    projected_hands = hands + existing_hires
    projected_money = money

    # Account only for HIREs already requested by the base model.
    n = hires_today
    for _ in range(existing_hires):
        projected_money -= fib_hire_cost(n)
        n += 1

    extra = []
    while projected_hands < target_hands and len(market) + len(extra) < 10:
        cost = fib_hire_cost(n)
        if projected_money - cost < reserve:
            break
        extra.append(["HIRE"])
        projected_money -= cost
        projected_hands += 1
        n += 1

    if not extra:
        return action

    # Preserve SELL first so the base ordering is minimally disturbed.
    sells = [o for o in market if o and o[0] == "SELL"]
    rest = [o for o in market if not (o and o[0] == "SELL")]
    action["market"] = (sells + extra + rest)[:10]
    return action
