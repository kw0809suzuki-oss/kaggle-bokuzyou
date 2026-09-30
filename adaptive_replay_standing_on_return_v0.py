"""Adaptive Replay Contract Runtime v0 + standing-on-return crop harvest.

Baseline is frozen:
    adaptive_replay_contract_runtime_v0.py

This candidate changes only one thing after the baseline action is produced:
if a self worker is already standing on a PLANT tile with yield_units > 0,
that worker performs HARVEST instead of the replay-selected unit action.

Market orders, other workers, replay state, and the step24 HIRE contract are
left unchanged. This is an experiment, not a promoted contract.
"""
from __future__ import annotations

from copy import deepcopy

import adaptive_replay_contract_runtime_v0 as base


standing_return_trigger_count = 0
last_standing_return_event = None


def reset_agent():
    global standing_return_trigger_count, last_standing_return_event
    standing_return_trigger_count = 0
    last_standing_return_event = None
    base.reset_agent()


def _harvestable_crop_at(farm, position):
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return False
    x, y = int(position[0]), int(position[1])
    tiles = farm.get("tiles", [])
    if y < 0 or y >= len(tiles):
        return False
    row = tiles[y]
    if x < 0 or x >= len(row):
        return False
    tile = row[x]
    return (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and int(tile.get("yield_units", 0) or 0) > 0
    )


def _apply_standing_on_return(action, obs):
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", [])
    if player < 0 or player >= len(farms):
        return action, None

    farm = farms[player]
    positions = [farm.get("farmer"), *list(farm.get("hands", []) or [])]
    changed = []

    out = deepcopy(action)
    hands = list(out.get("hands", []) or [])

    for idx, position in enumerate(positions):
        if not _harvestable_crop_at(farm, position):
            continue

        if idx == 0:
            previous = list(out.get("farmer", ["PASS"]))
            if previous == ["HARVEST"]:
                continue
            out["farmer"] = ["HARVEST"]
        else:
            hand_idx = idx - 1
            if hand_idx >= len(hands):
                continue
            previous = list(hands[hand_idx])
            if previous == ["HARVEST"]:
                continue
            hands[hand_idx] = ["HARVEST"]

        changed.append({
            "unit_index": idx,
            "position": [int(position[0]), int(position[1])],
            "replaced_action": previous,
        })

    if not changed:
        return action, None

    out["hands"] = hands
    return out, {
        "step": int(obs.get("step", 0) or 0),
        "changed_units": changed,
    }


def agent(obs, configuration=None):
    global standing_return_trigger_count, last_standing_return_event

    action = base.agent(obs, configuration)
    action, event = _apply_standing_on_return(action, obs)
    if event is not None:
        standing_return_trigger_count += 1
        last_standing_return_event = event
    return action
