"""Adaptive Circulation Intent Detour v0.

One fixed experiment at step252.

Observed baseline job for actor11:
    step252 EAST
    step253 EAST
    step254 PLANT STRAWBERRY at [5, 7]

Candidate:
    step252 HARVEST the mature WHEAT under actor11,
    then keep only the small observed intent:
        actor11 -> establish STRAWBERRY at [5, 7]

Important:
- We do NOT replay a saved EAST/EAST/PLANT command list.
- Every turn we start from the current Adaptive Circulation action bundle.
- Only actor11 may be overridden while the pending intent is open.
- Other workers and market orders remain the current baseline output.
- The intent closes only after Official State shows STRAWBERRY at [5, 7].
- WATER/care after planting is NOT part of the retained intent.
"""

from __future__ import annotations

from copy import deepcopy

import adaptive_circulation_runtime_v0 as body

TRIGGER_STEP = 252
ACTOR_INDEX = 11
START_POS = [3, 7]
TARGET_POS = [5, 7]
TARGET_CROP = "STRAWBERRY"

trigger_count = 0
completed_count = 0
pending_intent = None
failure_reason = None
event_log = []


def reset_agent():
    global trigger_count, completed_count, pending_intent, failure_reason, event_log
    trigger_count = 0
    completed_count = 0
    pending_intent = None
    failure_reason = None
    event_log = []
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def _positions(farm):
    return [list(farm.get("farmer"))] + [list(p) for p in (farm.get("hands", []) or [])]


def _unit_actions(bundle, actor_count):
    xs = [deepcopy(bundle.get("farmer", ["PASS"]))]
    xs.extend(deepcopy(bundle.get("hands", []) or []))
    while len(xs) < actor_count:
        xs.append(["PASS"])
    return xs[:actor_count]


def _replace_actor(bundle, actor_index, replacement):
    out = deepcopy(bundle)
    if actor_index == 0:
        out["farmer"] = deepcopy(replacement)
        return out
    hands = list(out.get("hands", []) or [])
    idx = actor_index - 1
    while len(hands) <= idx:
        hands.append(["PASS"])
    hands[idx] = deepcopy(replacement)
    out["hands"] = hands
    return out


def _tile_at(farm, pos):
    if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
        return None
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles", []) or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return tiles[y][x]
    return None


def _goal_observed(obs):
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    tile = _tile_at(farm, TARGET_POS)
    return (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and str(tile.get("crop")) == TARGET_CROP
    )


def _seed_count(obs):
    private = obs.get("private", {}) or {}
    return int((private.get("seeds", {}) or {}).get(TARGET_CROP, 0) or 0)


def _route_action(pos):
    x, y = int(pos[0]), int(pos[1])
    tx, ty = TARGET_POS
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return None


def _other_bundle_parts_unchanged(base, transformed, actor_index):
    # Market must be byte-for-byte equivalent after deepcopy semantics.
    if base.get("market", []) != transformed.get("market", []):
        return False

    base_farmer = base.get("farmer", ["PASS"])
    new_farmer = transformed.get("farmer", ["PASS"])
    if actor_index != 0 and base_farmer != new_farmer:
        return False

    b_hands = list(base.get("hands", []) or [])
    n_hands = list(transformed.get("hands", []) or [])
    max_len = max(len(b_hands), len(n_hands))
    for i in range(max_len):
        actor = i + 1
        b = b_hands[i] if i < len(b_hands) else ["PASS"]
        n = n_hands[i] if i < len(n_hands) else ["PASS"]
        if actor != actor_index and b != n:
            return False
    return True


def _record_override(obs, base, transformed, reason, replacement):
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = _positions(farm)
    base_actions = _unit_actions(base, len(positions))
    event_log.append({
        "kind": "override",
        "reason": reason,
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "actor_index": ACTOR_INDEX,
        "position": deepcopy(positions[ACTOR_INDEX]) if len(positions) > ACTOR_INDEX else None,
        "base_actor_action": deepcopy(base_actions[ACTOR_INDEX]) if len(base_actions) > ACTOR_INDEX else None,
        "replacement_action": deepcopy(replacement),
        "other_workers_and_market_unchanged": _other_bundle_parts_unchanged(
            base, transformed, ACTOR_INDEX
        ),
    })


def _fail(step, reason, extra=None):
    global pending_intent, failure_reason
    failure_reason = reason
    event = {"kind": "intent_failed", "step": int(step), "reason": reason}
    if extra:
        event.update(deepcopy(extra))
    event_log.append(event)
    pending_intent = None


def agent(obs, configuration=None):
    global trigger_count, completed_count, pending_intent

    base = deepcopy(body.agent(obs, configuration))
    step = int(obs.get("step", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    positions = _positions(farm)

    # Close only on observed Official State, not on PLANT issuance.
    if pending_intent is not None and _goal_observed(obs):
        completed_count += 1
        event_log.append({
            "kind": "intent_completed",
            "step": step,
            "target": deepcopy(TARGET_POS),
            "crop": TARGET_CROP,
        })
        pending_intent = None
        return base

    if pending_intent is not None:
        if len(positions) <= ACTOR_INDEX:
            _fail(step, "actor_missing")
            return base

        pos = positions[ACTOR_INDEX]
        if list(pos) == TARGET_POS:
            tile = _tile_at(farm, TARGET_POS)
            if tile is not None:
                _fail(step, "target_not_empty", {"target_tile": deepcopy(tile)})
                return base
            if _seed_count(obs) <= 0:
                _fail(step, "strawberry_seed_unavailable")
                return base

            replacement = ["PLANT", TARGET_CROP]
            transformed = _replace_actor(base, ACTOR_INDEX, replacement)
            _record_override(obs, base, transformed, "resume_intent_plant", replacement)
            pending_intent["plant_issued_step"] = step
            return transformed

        replacement = _route_action(pos)
        if replacement is None:
            _fail(step, "route_unavailable", {"position": deepcopy(pos)})
            return base

        transformed = _replace_actor(base, ACTOR_INDEX, replacement)
        _record_override(obs, base, transformed, "resume_intent_move", replacement)
        return transformed

    # One fixed trigger only.
    if trigger_count > 0 or step != TRIGGER_STEP:
        return base

    if len(positions) <= ACTOR_INDEX or list(positions[ACTOR_INDEX]) != START_POS:
        return base

    tile = _tile_at(farm, START_POS)
    actions = _unit_actions(base, len(positions))
    actor_action = actions[ACTOR_INDEX]

    trigger_ok = (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and str(tile.get("crop")) == "WHEAT"
        and float(tile.get("yield_units", 0) or 0) == 3.0
        and actor_action == ["EAST"]
    )
    if not trigger_ok:
        return base

    replacement = ["HARVEST"]
    transformed = _replace_actor(base, ACTOR_INDEX, replacement)
    trigger_count = 1
    pending_intent = {
        "actor_index": ACTOR_INDEX,
        "target": deepcopy(TARGET_POS),
        "operation": "PLANT",
        "crop": TARGET_CROP,
        "opened_step": step,
    }
    event_log.append({
        "kind": "detour_triggered",
        "step": step,
        "actor_index": ACTOR_INDEX,
        "position": deepcopy(START_POS),
        "harvest_tile": deepcopy(tile),
        "baseline_actor_action": deepcopy(actor_action),
        "pending_intent": deepcopy(pending_intent),
    })
    _record_override(obs, base, transformed, "detour_harvest", replacement)
    return transformed
