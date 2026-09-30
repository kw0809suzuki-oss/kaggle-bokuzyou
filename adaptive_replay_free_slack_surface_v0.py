"""Adaptive Replay Free Slack Surface v0.

Purpose
-------
Add genuine *slack* to the proven Adaptive Replay runtime without confusing a
generated alternative with Evidence.

Principle:
    GENERATE FREE
    DETERMINE WITH WARRANT

Runtime shape:
    Current Official World
        -> standing candidate: Adaptive Replay Contract Runtime
        -> free candidate surface: independent generators may emit ANY complete
           Kaggriculture ActionBundle
        -> warrant boundary
        -> Action

v0 wires World Lens v0 as the first free generator.  The generated bundle is
NOT required to preserve replay operations, quantities, ordering, actor work,
or market work.

Crucially, v0 contains no promoted warrant for replacing the standing replay
with a generated candidate.  Therefore the issued action remains the proven
Adaptive Replay action.  The free alternative is retained as an observable
candidate surface for future evidence, rather than silently promoted by a
heuristic.

The already-promoted step24 Effect Contract remains inside the standing
Adaptive Replay runtime and is unaffected.

This file is suitable as a submission-facing runtime experiment because the
free surface is non-interfering until an explicit warrant is promoted.
"""
from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "adaptive_replay_contract_runtime_v0.py"
WORLD_LENS_PATH = ROOT / "world_lens_agent_v0.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load(BASE_PATH, "_free_slack_adaptive_replay")
world_lens = _load(WORLD_LENS_PATH, "_free_slack_world_lens")

surface_count = 0
distinct_alternative_count = 0
last_surface = None
first_distinct_surface = None


def _plain(v: Any):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return [_plain(x) for x in v]
    return v


def _normalize_bundle(obs, action):
    p = int(obs["player"])
    hands = len(obs["farms"][p].get("hands", []) or [])
    raw = _plain(action) if isinstance(action, dict) else {}

    farmer = raw.get("farmer", ["PASS"])
    if not isinstance(farmer, list) or not farmer:
        farmer = ["PASS"]

    hand_actions = list(raw.get("hands", []) or [])
    hand_actions = (hand_actions + [["PASS"]] * hands)[:hands]

    market = list(raw.get("market", []) or [])
    return {
        "farmer": farmer,
        "hands": hand_actions,
        "market": market,
    }


def _canonical(action) -> str:
    return json.dumps(
        _plain(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def reset_agent():
    global surface_count, distinct_alternative_count
    global last_surface, first_distinct_surface

    surface_count = 0
    distinct_alternative_count = 0
    last_surface = None
    first_distinct_surface = None

    if hasattr(base, "reset_agent"):
        base.reset_agent()
    if hasattr(world_lens, "reset_agent"):
        world_lens.reset_agent()


def generate_candidate_surface(obs, configuration=None):
    """Return the standing proposal plus freely generated alternatives.

    A generator is allowed to emit any complete ActionBundle.  This function
    applies no replay-similarity rule.
    """
    replay_action = _normalize_bundle(obs, base.agent(obs, configuration))

    candidates = [{
        "source": "adaptive_replay",
        "role": "standing",
        "action": replay_action,
    }]

    try:
        alternative = _normalize_bundle(
            obs,
            world_lens.agent(obs, configuration),
        )
        candidates.append({
            "source": "world_lens",
            "role": "free_candidate",
            "action": alternative,
        })
    except Exception as exc:
        candidates.append({
            "source": "world_lens",
            "role": "generator_error",
            "error": repr(exc),
            "action": None,
        })

    return candidates


def _determine_with_warrant(candidates):
    """v0 warrant boundary.

    No generated alternative has terminal evidence sufficient for promotion.
    Therefore the standing candidate keeps decision authority.

    Future promoted warrants belong HERE.  Candidate generation stays free.
    """
    for row in candidates:
        if row.get("role") == "standing":
            return row
    raise RuntimeError("standing candidate missing")


def agent(obs, configuration=None):
    global surface_count, distinct_alternative_count
    global last_surface, first_distinct_surface

    candidates = generate_candidate_surface(obs, configuration)
    chosen = _determine_with_warrant(candidates)

    standing = chosen["action"]
    alternative = next(
        (
            row.get("action")
            for row in candidates
            if row.get("role") == "free_candidate"
            and row.get("action") is not None
        ),
        None,
    )

    distinct = (
        alternative is not None
        and _canonical(alternative) != _canonical(standing)
    )

    surface_count += 1
    if distinct:
        distinct_alternative_count += 1

    surface = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "standing_source": chosen["source"],
        "alternative_present": alternative is not None,
        "alternative_distinct": bool(distinct),
        "standing_action": deepcopy(standing),
        "alternative_action": deepcopy(alternative),
        "warrant": {
            "status": "NO_PROMOTED_WARRANT",
            "decision": "KEEP_STANDING",
            "reason": (
                "Generated alternatives are candidates, not Evidence. "
                "No terminal-supported replacement warrant is active."
            ),
        },
    }
    last_surface = surface
    if distinct and first_distinct_surface is None:
        first_distinct_surface = deepcopy(surface)

    return deepcopy(standing)
