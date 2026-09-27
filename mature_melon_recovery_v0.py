"""Mature MELON Recovery v0.

Single intervention on current Independent Distilled v0:
If a worker is already standing on a harvestable MELON tile at max yield 6,
and the scheduled action is not HARVEST, replace only that worker action with HARVEST.

No change to hiring, routing, planting, watering, buying, selling, land, or market actions.
"""
from copy import deepcopy

import astra_flow_independent_distilled_v0 as baseline


def reset_agent():
    if hasattr(baseline, "reset_agent"):
        baseline.reset_agent()


def _harvestable_max_melon(obs, player, pos):
    if not pos or len(pos) != 2:
        return False
    x, y = int(pos[0]), int(pos[1])
    farm = obs["farms"][player]
    tile = farm["tiles"][y][x]
    if not isinstance(tile, dict):
        return False
    if tile.get("kind") != "PLANT" or tile.get("crop") != "MELON":
        return False
    if int(tile.get("yield_units", 0) or 0) != 6:
        return False

    day = int(obs.get("day", 0) or 0)
    raw_planted_day = tile.get("planted_day")
    planted_day = day if raw_planted_day is None else int(raw_planted_day)
    return day - planted_day >= 10


def agent(obs, configuration=None):
    action = deepcopy(baseline.agent(obs, configuration))
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]

    farmer_pos = farm.get("farmer")
    if _harvestable_max_melon(obs, player, farmer_pos):
        if action.get("farmer") != ["HARVEST"]:
            action["farmer"] = ["HARVEST"]

    hand_positions = list(farm.get("hands", []) or [])
    hand_actions = list(action.get("hands", []) or [])
    for i, pos in enumerate(hand_positions):
        if i >= len(hand_actions):
            break
        if _harvestable_max_melon(obs, player, pos):
            if hand_actions[i] != ["HARVEST"]:
                hand_actions[i] = ["HARVEST"]
    action["hands"] = hand_actions

    return action
