"""Step250 PASS probe.

Purpose:
Separate two effects mixed in the earlier WATER -> HARVEST result:
  1) not doing the original WATER this turn
  2) using that freed turn for HARVEST

This candidate changes exactly the same first opportunity selected by
adaptive_circulation_opportunity_v0, but replaces that actor action with PASS.
After that one turn, the agent immediately returns to the existing runtime.

No intent retention, no compensation, no rescue condition.
"""

from __future__ import annotations

from copy import deepcopy

import adaptive_circulation_runtime_v0 as body
import adaptive_circulation_opportunity_v0 as harvest_probe

trigger_count = 0
last_event = None


def reset_agent():
    global trigger_count, last_event
    trigger_count = 0
    last_event = None
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def agent(obs, configuration=None):
    global trigger_count, last_event

    action = deepcopy(body.agent(obs, configuration))
    if trigger_count > 0:
        return action

    candidate = harvest_probe._candidate_actor(obs, action)
    if candidate is None:
        return action

    transformed = harvest_probe._replace_actor_action(
        action, candidate["actor_index"], ["PASS"]
    )
    trigger_count = 1
    last_event = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        **candidate,
        "replacement_action": ["PASS"],
    }
    return transformed
