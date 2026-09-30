"""Adaptive Circulation Opportunity v0.

A one-shot local alternative layered on Adaptive Circulation Runtime v0.

This is deliberately NOT a general HARVEST priority rule.

Intervention condition:
  - current ActionBundle contains productive expansion labor
    (PLANT / BUILD_COOP / BUILD_PASTURE / animal PLACE),
  - no actor is already HARVESTing this turn,
  - at least one actor is standing on a harvestable mature crop,
  - this candidate has not intervened earlier in the episode.

Then replace exactly one such actor's unit action with HARVEST.
Everything else remains unchanged.

Purpose:
  Test one concrete labor-allocation conflict:
      mature output waits
      while labor is opening/activating new production.

A positive terminal A/B may justify a narrower future contract.
A negative result rejects only this one-shot condition/transform pair.
"""

from __future__ import annotations

from copy import deepcopy

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

import adaptive_circulation_runtime_v0 as body

trigger_count = 0
last_event = None

_EXPANSION_OPS = {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}


def reset_agent():
    global trigger_count, last_event
    trigger_count = 0
    last_event = None
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def _op(action):
    if isinstance(action, (list, tuple)) and action:
        return str(action[0])
    return "PASS"


def _positions(farm):
    return [list(farm.get("farmer"))] + [list(p) for p in (farm.get("hands", []) or [])]


def _unit_actions(bundle, actor_count):
    actions = [deepcopy(bundle.get("farmer", ["PASS"]))]
    actions.extend(deepcopy(bundle.get("hands", []) or []))
    while len(actions) < actor_count:
        actions.append(["PASS"])
    return actions[:actor_count]


def _is_expansion_action(action):
    op = _op(action)
    if op in _EXPANSION_OPS:
        return True
    if op == "PLACE" and isinstance(action, (list, tuple)) and len(action) >= 2:
        return str(action[1]) in {"GOOSE", "COW", "SHEEP"}
    return False


def _harvestable_mature_crop(tile, day):
    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
        return False
    crop = str(tile.get("crop"))
    if crop not in CROPS:
        return False
    if float(tile.get("yield_units", 0) or 0) <= 0:
        return False
    planted_day = int(tile.get("planted_day", day) or day)
    return int(day) - planted_day >= int(CROPS[crop]["first_yield_day"])


def _candidate_actor(obs, bundle):
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = _positions(farm)
    actions = _unit_actions(bundle, len(positions))

    # The candidate is specifically about an unresolved expansion-vs-return
    # conflict, not "harvest whenever possible".
    if not any(_is_expansion_action(a) for a in actions):
        return None
    if any(_op(a) == "HARVEST" for a in actions):
        return None

    day = int(obs.get("day", 0) or 0)
    tiles = farm.get("tiles", []) or []
    for idx, pos in enumerate(positions):
        if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
            continue
        x, y = int(pos[0]), int(pos[1])
        if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
            continue
        tile = tiles[y][x]
        if _harvestable_mature_crop(tile, day):
            return {
                "actor_index": idx,
                "position": [x, y],
                "tile": deepcopy(tile),
                "original_action": deepcopy(actions[idx]),
                "expansion_actions": [
                    {"actor_index": j, "action": deepcopy(a)}
                    for j, a in enumerate(actions)
                    if _is_expansion_action(a)
                ],
            }
    return None


def _replace_actor_action(bundle, actor_index, replacement):
    out = deepcopy(bundle)
    if actor_index == 0:
        out["farmer"] = deepcopy(replacement)
        return out
    hands = list(out.get("hands", []) or [])
    hand_idx = actor_index - 1
    while len(hands) <= hand_idx:
        hands.append(["PASS"])
    hands[hand_idx] = deepcopy(replacement)
    out["hands"] = hands
    return out


def agent(obs, configuration=None):
    global trigger_count, last_event

    action = deepcopy(body.agent(obs, configuration))
    if trigger_count > 0:
        return action

    candidate = _candidate_actor(obs, action)
    if candidate is None:
        return action

    transformed = _replace_actor_action(action, candidate["actor_index"], ["HARVEST"])
    trigger_count = 1
    last_event = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        **candidate,
        "replacement_action": ["HARVEST"],
    }
    return transformed
