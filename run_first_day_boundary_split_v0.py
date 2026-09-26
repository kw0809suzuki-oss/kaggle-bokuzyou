#!/usr/bin/env python3
"""First Day Boundary Split v0.

Window:
    Initial State -> turn 23 post-state (Day1 h0 boundary)

Question:
Within the same first-day window, what is the earliest Official World
transition at which Candidate-uniform and Kind-uniform visibly separate?

This observer does not explain the split with Plan names.
It records, for both fixed runs in the same format:

    Pre-State -> ActionBundle -> Official World delta -> Post-State -> Next State

Tracked World coordinates are descriptive only:
cash, seeds, carried inventory, shed, tile/plant state, WEED, watered flags,
and yield deltas.

No causal attribution to WATER / maintenance / Selection / continuation.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
import run_selection_kind_uniform_ab_v0 as ab
import run_first_cash_return_window_v0 as fw
from plan_generator_entrance_v0 import bind_official_state

WINDOW_LAST_TURN = 23


def plain(x):
    return body.plain(x)


def nonzero_mapping(m):
    return {str(k): float(v) for k, v in (m or {}).items() if float(v) != 0.0}


def normalized_delta(d):
    """Keep only tracked observed changes, without Plan interpretation."""
    return {
        "cash_delta": float(d.get("cash_delta", 0.0)),
        "seed_delta": nonzero_mapping(d.get("seed_delta", {})),
        "carried_delta": nonzero_mapping(d.get("carried_delta", {})),
        "shed_delta": nonzero_mapping(d.get("shed_delta", {})),
        "surface_transition_counts": nonzero_mapping(
            d.get("surface_transition_counts", {})
        ),
        "plant_established_by_crop": nonzero_mapping(
            d.get("plant_established_by_crop", {})
        ),
        "plant_to_weed_by_crop": nonzero_mapping(
            d.get("plant_to_weed_by_crop", {})
        ),
        "watered_false_to_true_by_crop": nonzero_mapping(
            d.get("watered_false_to_true_by_crop", {})
        ),
        "watered_true_to_false_by_crop": nonzero_mapping(
            d.get("watered_true_to_false_by_crop", {})
        ),
        "positive_yield_delta_by_crop": nonzero_mapping(
            d.get("positive_yield_delta_by_crop", {})
        ),
        "negative_yield_delta_by_crop": nonzero_mapping(
            d.get("negative_yield_delta_by_crop", {})
        ),
    }


def live_plant_count(s):
    return int((s.get("surface") or {}).get("PLANT", 0) or 0)


def tracked_state(s):
    return {
        "day": int(s["day"]),
        "hour": int(s["hour"]),
        "cash": float(s["cash"]),
        "seeds": plain(s.get("seeds", {})),
        "carried": plain(s.get("carried", {})),
        "shed": plain(s.get("shed", {})),
        "surface": plain(s.get("surface", {})),
        "plants_by_crop": plain(s.get("plants_by_crop", {})),
        "watered_plants": int(s.get("watered_plants", 0) or 0),
        "unwatered_plants": int(s.get("unwatered_plants", 0) or 0),
        "yield_units_by_crop": plain(s.get("yield_units_by_crop", {})),
        "yield_units_total": float(s.get("yield_units_total", 0.0) or 0.0),
    }


def replay_window(run):
    env = make("kaggriculture", configuration={"seed": ab.ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    rows = []
    all_hashes_match = True

    # Replay one turn beyond the window so turn23 can expose Next State.
    max_turn = min(WINDOW_LAST_TURN + 1, len(run["turns"]) - 1)

    for i, src in enumerate(run["turns"]):
        if i > max_turn:
            break

        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != src["pre_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} pre hash mismatch at turn {i}")

        pre_state = fw.world_snapshot(pre)
        action = plain(src["action_bundle"])
        opp = plain(src["opponent_action_bundle"])

        env.step([action, opp])

        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != src["post_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} post hash mismatch at turn {i}")

        post_state = fw.world_snapshot(post)
        rows.append({
            "turn": i,
            "pre_state": tracked_state(pre_state),
            "action_bundle": action,
            "world_delta": normalized_delta(fw.event_delta(pre, post)),
            "post_state": tracked_state(post_state),
            "next_state": None,
        })

    for i in range(min(WINDOW_LAST_TURN + 1, len(rows))):
        if i + 1 < len(rows):
            rows[i]["next_state"] = rows[i + 1]["pre_state"]

    visible = rows[: WINDOW_LAST_TURN + 1]
    return {
        "rows": visible,
        "guard": {
            "all_replayed_pre_post_hashes_match": all_hashes_match,
            "window_turn_count": len(visible),
            "expected_window_turn_count": WINDOW_LAST_TURN + 1,
            "passed": (
                all_hashes_match
                and len(visible) == WINDOW_LAST_TURN + 1
            ),
        },
    }


def lifecycle_signature(d):
    keys = (
        "surface_transition_counts",
        "plant_established_by_crop",
        "plant_to_weed_by_crop",
        "watered_false_to_true_by_crop",
        "watered_true_to_false_by_crop",
        "positive_yield_delta_by_crop",
        "negative_yield_delta_by_crop",
    )
    return {k: d[k] for k in keys}


def first_index(predicate, pairs):
    for pair in pairs:
        if predicate(pair):
            return pair["turn"]
    return None


def make_pairs(b, p):
    pairs = []
    for rb, rp in zip(b["rows"], p["rows"]):
        if rb["turn"] != rp["turn"]:
            raise RuntimeError("turn alignment mismatch")
        pairs.append({
            "turn": rb["turn"],
            "pre_state_equal": rb["pre_state"] == rp["pre_state"],
            "action_equal": rb["action_bundle"] == rp["action_bundle"],
            "world_delta_equal": rb["world_delta"] == rp["world_delta"],
            "post_state_equal": rb["post_state"] == rp["post_state"],
            "live_plant_count": {
                "baseline": live_plant_count(rb["post_state"]),
                "kind_uniform": live_plant_count(rp["post_state"]),
            },
            "baseline": rb,
            "kind_uniform": rp,
        })
    return pairs


def compact_pair(pair):
    if pair is None:
        return None
    return {
        "turn": pair["turn"],
        "pre_state_equal": pair["pre_state_equal"],
        "action_equal": pair["action_equal"],
        "world_delta_equal": pair["world_delta_equal"],
        "post_state_equal": pair["post_state_equal"],
        "live_plant_count": pair["live_plant_count"],
        "baseline": {
            "pre_state": pair["baseline"]["pre_state"],
            "action_bundle": pair["baseline"]["action_bundle"],
            "world_delta": pair["baseline"]["world_delta"],
            "post_state": pair["baseline"]["post_state"],
            "next_state": pair["baseline"]["next_state"],
        },
        "kind_uniform": {
            "pre_state": pair["kind_uniform"]["pre_state"],
            "action_bundle": pair["kind_uniform"]["action_bundle"],
            "world_delta": pair["kind_uniform"]["world_delta"],
            "post_state": pair["kind_uniform"]["post_state"],
            "next_state": pair["kind_uniform"]["next_state"],
        },
    }


def main():
    baseline_run = body.run_policy("exclude_blocked")
    kind_run = ab.run_kind_uniform()

    b = replay_window(baseline_run)
    p = replay_window(kind_run)

    if not b["guard"]["passed"]:
        raise RuntimeError("baseline guard failed")
    if not p["guard"]["passed"]:
        raise RuntimeError("kind-uniform guard failed")

    pairs = make_pairs(b, p)

    first_action_diff = first_index(
        lambda x: not x["action_equal"], pairs
    )
    first_any_world_delta_diff = first_index(
        lambda x: not x["world_delta_equal"], pairs
    )
    first_post_state_diff = first_index(
        lambda x: not x["post_state_equal"], pairs
    )
    first_live_count_diff = first_index(
        lambda x: (
            x["live_plant_count"]["baseline"]
            != x["live_plant_count"]["kind_uniform"]
        ),
        pairs,
    )
    first_plant_lifecycle_delta_diff = first_index(
        lambda x: (
            lifecycle_signature(x["baseline"]["world_delta"])
            != lifecycle_signature(x["kind_uniform"]["world_delta"])
        ),
        pairs,
    )

    by_turn = {x["turn"]: x for x in pairs}
    key_turns = sorted(set(
        t for t in [
            first_action_diff,
            first_any_world_delta_diff,
            first_post_state_diff,
            first_live_count_diff,
            first_plant_lifecycle_delta_diff,
            WINDOW_LAST_TURN,
        ]
        if t is not None
    ))

    result = {
        "schema": "first-day-boundary-split-v0",
        "question": (
            "From Initial State through turn23 / Day1 h0, what is the first "
            "observed Official World transition at which the two trajectories "
            "visibly separate?"
        ),
        "source": {
            "baseline": "Confirmed Circulation Body v0 / exclude_blocked",
            "kind_uniform": "Selection Kind-Uniform A/B v0 / kind_uniform",
            "environment_seed": ab.ENV_SEED,
            "policy_seed": ab.POLICY_SEED,
        },
        "window": {
            "first_turn": 0,
            "last_turn": WINDOW_LAST_TURN,
            "boundary_post_state": {
                "baseline": b["rows"][-1]["post_state"],
                "kind_uniform": p["rows"][-1]["post_state"],
            },
        },
        "first_observed_separations": {
            "first_action_bundle_difference_turn": first_action_diff,
            "first_any_tracked_world_delta_difference_turn": (
                first_any_world_delta_diff
            ),
            "first_post_state_difference_turn": first_post_state_diff,
            "first_live_plant_count_difference_turn": first_live_count_diff,
            "first_plant_lifecycle_delta_difference_turn": (
                first_plant_lifecycle_delta_diff
            ),
        },
        "key_turns": {
            str(t): compact_pair(by_turn[t]) for t in key_turns
        },
        "paired_turns": pairs,
        "guards": {
            "baseline": b["guard"],
            "kind_uniform": p["guard"],
        },
        "boundary": {
            "plan_name_used_as_explanation": False,
            "water_assumed_as_cause": False,
            "maintenance_assumed_as_cause": False,
            "selection_assumed_as_cause": False,
            "continuation_assumed_as_cause": False,
            "cross_run_difference_treated_as_causal_effect": False,
            "new_policy_added": False,
        },
    }

    Path("first_day_boundary_split_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "window": result["window"],
        "first_observed_separations": result["first_observed_separations"],
        "key_turns": result["key_turns"],
        "guards": result["guards"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
