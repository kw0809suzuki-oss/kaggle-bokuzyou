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


def count_plants(farm):
    return sum(
        1
        for row in farm.get("tiles", [])
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )


def agent(obs):
    action = copy.deepcopy(base.agent(obs))
    day, hour = obs["day"], obs["hour"]
    if not (7 <= day <= 10 and hour <= 4):
        return action

    pid = obs["player"]
    me = obs["farms"][pid]
    opp = obs["farms"][1 - pid]
    private = obs["private"]

    if count_plants(me) >= count_plants(opp):
        return action

    empty = sum(1 for row in me.get("tiles", []) for tile in row if tile is None)
    seeds = sum(private.get("seeds", {}).values())
    if empty <= 0 or seeds <= 0:
        return action

    market = list(action.get("market", []))
    room = 10 - len(market)
    if room <= 0:
        return action

    existing_hires = sum(1 for o in market if o and o[0] == "HIRE")
    my_projected_hands = len(me.get("hands", [])) + existing_hires
    opp_hands = len(opp.get("hands", []))

    # Pair rule: only close an observed labor gap; never invent a fixed target.
    need = max(0, opp_hands - my_projected_hands)
    extra_n = min(need, 3, room)
    if extra_n <= 0:
        return action

    cash = me.get("money", 0)
    reserve = 650
    n = me.get("hires_today", 0) + existing_hires
    extra = []
    for _ in range(extra_n):
        cost = fib_hire_cost(n)
        if cash - cost < reserve:
            break
        extra.append(["HIRE"])
        cash -= cost
        n += 1

    if not extra:
        return action

    sells = [o for o in market if o and o[0] == "SELL"]
    rest = [o for o in market if not (o and o[0] == "SELL")]
    action["market"] = sells + extra + rest
    return action
