"""Effect Observer v0 for Adaptive Replay Runtime.

Observation only. This module never transforms an action, promotes a Contract,
or chooses Recovery. It exposes a stable record separating Requested Action,
Current-World Contract input, and the Runtime's emitted action/event.
"""
from __future__ import annotations
from copy import deepcopy


def observe_runtime_step(*, obs, requested_action, emitted_action, contract_id=None, contract_event=None):
    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    return {
        "step": int(obs.get("step", 0) or 0),
        "player": player,
        "world_input": {
            "money": float(farm.get("money", 0) or 0),
            "hires_today": int(farm.get("hires_today", 0) or 0),
            "hands": len(farm.get("hands", []) or []),
        },
        "requested_action": deepcopy(requested_action),
        "emitted_action": deepcopy(emitted_action),
        "contract_id": contract_id,
        "contract_event": deepcopy(contract_event),
        "action_changed": requested_action != emitted_action,
    }


def observe_effect(*, pre, post, requested_action, emitted_action, label=None):
    """Store already-observed pre/post effect without interpreting causality."""
    return {
        "label": label,
        "requested_action": deepcopy(requested_action),
        "emitted_action": deepcopy(emitted_action),
        "pre": deepcopy(pre),
        "post": deepcopy(post),
    }
