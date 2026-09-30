"""Price Board v0.

One-shot use of the visible market price board at the same first opportunity
used by the step250 WATER/PASS/HARVEST probes.

No fitted seed-specific threshold.

For the harvestable crop under the selected actor:
- compute its current price / base price
- compute the mean current price / base price across all market products
- target above mean -> HARVEST
- target below mean -> keep the existing action (normally WATER)
- exact tie -> PASS

This tests whether the game's directly visible current valuation can improve
the existing runtime at this already-observed decision point.
"""

from __future__ import annotations

from copy import deepcopy

from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS, PRODUCTS

import adaptive_circulation_runtime_v0 as body
import adaptive_circulation_opportunity_v0 as opportunity

trigger_count = 0
last_event = None


def reset_agent():
    global trigger_count, last_event
    trigger_count = 0
    last_event = None
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def _decision(obs, candidate):
    tile = candidate.get("tile") or {}
    crop = str(tile.get("crop", ""))
    prices = ((obs.get("market") or {}).get("prices") or {})
    if crop not in MARKET_PARAMS or crop not in prices:
        return "ORIGINAL", None

    ratios = {}
    for item in PRODUCTS:
        if item in prices and item in MARKET_PARAMS:
            base = float(MARKET_PARAMS[item]["base"])
            ratios[item] = float(prices[item]) / base if base else 0.0

    if not ratios or crop not in ratios:
        return "ORIGINAL", None

    target = ratios[crop]
    mean_ratio = sum(ratios.values()) / len(ratios)

    eps = 1e-12
    if target > mean_ratio + eps:
        choice = "HARVEST"
    elif target < mean_ratio - eps:
        choice = "ORIGINAL"
    else:
        choice = "PASS"

    return choice, {
        "crop": crop,
        "prices": {k: float(prices[k]) for k in sorted(ratios)},
        "normalized_price": ratios,
        "target_ratio": target,
        "mean_ratio": mean_ratio,
    }


def agent(obs, configuration=None):
    global trigger_count, last_event

    action = deepcopy(body.agent(obs, configuration))
    if trigger_count > 0:
        return action

    candidate = opportunity._candidate_actor(obs, action)
    if candidate is None:
        return action

    choice, price_view = _decision(obs, candidate)
    transformed = action

    if choice == "HARVEST":
        transformed = opportunity._replace_actor_action(
            action, candidate["actor_index"], ["HARVEST"]
        )
    elif choice == "PASS":
        transformed = opportunity._replace_actor_action(
            action, candidate["actor_index"], ["PASS"]
        )

    trigger_count = 1
    last_event = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        **candidate,
        "price_choice": choice,
        "price_view": price_view,
        "replacement_action": (
            ["HARVEST"] if choice == "HARVEST"
            else ["PASS"] if choice == "PASS"
            else deepcopy(candidate["original_action"])
        ),
    }
    return transformed
