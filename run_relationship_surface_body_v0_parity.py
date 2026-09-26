#!/usr/bin/env python3
"""Parity check for relationship_surface_body_v0.

This is not a new policy experiment. It only checks that the standalone Body
emits exactly the same self ActionBundles and Official pre/post hashes as the
already-promoted Relationship Surface P1 runner.

Checked Worlds:
- 7001: fixed probe World
- 7201: one Promotion Boundary fresh World
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments import make

import relationship_surface_body_v0 as promoted
import run_current_world_relationship_surface_ab_v0 as surface
import run_exclude_confirmed_blocked_ab_v0 as legacy
from plan_generator_entrance_v0 import bind_official_state

SEEDS = [7001, 7201]


def check_seed(seed: int):
    legacy.ENV_SEED = seed
    surface.ENV_SEED = seed
    expected = surface.run_relationship_surface()

    opponent = legacy.load_opponent(f"promoted_body_parity_{seed}")
    body = promoted.RelationshipSurfaceBody(
        policy_seed=legacy.POLICY_SEED,
        max_plan_steps=legacy.MAX_PLAN_STEPS,
    )

    env = make(
        "kaggriculture",
        configuration={"seed": seed},
        debug=False,
    )
    env.reset(num_agents=2)

    first_difference = None
    turn = 0
    while not env.done:
        row = expected["turns"][turn]
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)
        pre = bind_official_state(s0["observation"])

        action = body.act(s0["observation"])
        opp_action = legacy.plain(opponent.agent(s1["observation"]))

        if first_difference is None and pre.canonical_hash != row["pre_hash"]:
            first_difference = {
                "turn": turn,
                "field": "pre_hash",
                "expected": row["pre_hash"],
                "actual": pre.canonical_hash,
            }

        if first_difference is None and action != row["action_bundle"]:
            first_difference = {
                "turn": turn,
                "field": "action_bundle",
                "expected": row["action_bundle"],
                "actual": action,
            }

        if first_difference is None and opp_action != row["opponent_action_bundle"]:
            first_difference = {
                "turn": turn,
                "field": "opponent_action_bundle",
                "expected": row["opponent_action_bundle"],
                "actual": opp_action,
            }

        env.step([action, opp_action])
        post = bind_official_state(env.state[0].observation)

        if first_difference is None and post.canonical_hash != row["post_hash"]:
            first_difference = {
                "turn": turn,
                "field": "post_hash",
                "expected": row["post_hash"],
                "actual": post.canonical_hash,
            }

        turn += 1

    final = bind_official_state(env.state[0].observation)
    actual_terminal = legacy.state_summary(final)["cash"]
    expected_terminal = expected["summary"]["terminal_self_cash"]

    passed = (
        first_difference is None
        and turn == len(expected["turns"])
        and actual_terminal == expected_terminal
    )

    return {
        "seed": seed,
        "passed": passed,
        "turns": turn,
        "expected_turns": len(expected["turns"]),
        "expected_terminal": expected_terminal,
        "actual_terminal": actual_terminal,
        "first_difference": first_difference,
    }


def main():
    rows = [check_seed(seed) for seed in SEEDS]
    result = {
        "schema": "relationship-surface-body-v0-parity",
        "meaning": (
            "Standalone Body parity only; not a new policy experiment."
        ),
        "rows": rows,
        "passed": all(r["passed"] for r in rows),
    }

    Path("relationship_surface_body_v0_parity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))

    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
