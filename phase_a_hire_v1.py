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
    n = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                n += 1
    return n


def agent(obs):
    action = copy.deepcopy(base.agent(obs))
    day = obs["day"]
    hour = obs["hour"]
    if not (7 <= day <= 10 and hour <= 3):
        return action

    pid = obs["player"]
    me = obs["farms"][pid]
    opp = obs["farms"][1 - pid]
    private = obs["private"]

    my_plants = count_plants(me)
    opp_plants = count_plants(opp)
    empty = sum(1 for row in me.get("tiles", []) for tile in row if tile is None)
    seeds = sum(private.get("seeds", {}).values())

    # Phase A only: extra labor when visible productive capacity is still behind
    # and there is something plantable to convert.
    if my_plants >= opp_plants or empty <= 0 or seeds <= 0:
        return action

    market = list(action.get("market", []))
    room = max(0, 10 - len(market))
    if room <= 0:
        return action

    existing_hires = sum(1 for o in market if o and o[0] == "HIRE")
    hands = len(me.get("hands", []))
    hires_today = me.get("hires_today", 0)
    projected_hands = hands + existing_hires

    target_hands = 12
    max_extra_this_turn = min(3, room)
    reserve = 650
    cash = me.get("money", 0)

    # Only charge costs we explicitly add. If the official market later cannot
    # afford them, it will reject them; no synthetic world is introduced.
    n = hires_today + existing_hires
    extra = []
    while (
        projected_hands < target_hands
        and len(extra) < max_extra_this_turn
    ):
        cost = fib_hire_cost(n)
        if cash - cost < reserve:
            break
        extra.append(["HIRE"])
        cash -= cost
        projected_hands += 1
        n += 1

    if not extra:
        return action

    sells = [o for o in market if o and o[0] == "SELL"]
    rest = [o for o in market if not (o and o[0] == "SELL")]
    # No base order is dropped: extra orders fit only in unused market slots.
    action["market"] = sells + extra + rest
    return action
