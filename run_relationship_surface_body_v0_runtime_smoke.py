#!/usr/bin/env python3
"""Standalone runtime smoke test for Relationship Surface Body v0.

Purpose:
Validate the promoted Body as a normal Kaggriculture agent, outside the
experiment runner.

Checks on one unused fresh World (seed 7301):
- module-level agent(obs) runs to Official terminal vs pinned Seyamalam
- no agent exception is hidden
- every returned ActionBundle is accepted by Official World
- terminal reward/cash are present
- after reset_agent(), the same World replays deterministically:
  identical self ActionBundle sequence, Official pre/post hashes, and terminal

This is implementation verification, not a policy-strength experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kaggle_environments import make

import relationship_surface_body_v0 as body
import run_exclude_confirmed_blocked_ab_v0 as legacy
from plan_generator_entrance_v0 import bind_official_state

ENV_SEED = 7301


def digest_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_once(tag: str):
    body.reset_agent()
    opponent = legacy.load_opponent(tag)
    env = make(
        "kaggriculture",
        configuration={"seed": ENV_SEED},
        debug=False,
    )
    env.reset(num_agents=2)

    actions = []
    pre_hashes = []
    post_hashes = []
    turns = 0

    while not env.done:
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)

        pre = bind_official_state(s0["observation"])
        pre_hashes.append(pre.canonical_hash)

        action = body.agent(s0["observation"])
        opponent_action = legacy.plain(opponent.agent(s1["observation"]))

        if not isinstance(action, dict):
            raise TypeError(f"Body returned {type(action).__name__}, not dict")
        for key in ("farmer", "hands", "market"):
            if key not in action:
                raise KeyError(f"Body ActionBundle missing {key}")

        actions.append(legacy.plain(action))
        env.step([legacy.plain(action), opponent_action])

        post = bind_official_state(env.state[0].observation)
        post_hashes.append(post.canonical_hash)
        turns += 1

    final = bind_official_state(env.state[0].observation)
    rewards = []
    for st in env.state:
        try:
            rewards.append(float(st.reward))
        except Exception:
            rewards.append(None)

    terminal_cash = legacy.state_summary(final)["cash"]

    if any(r is None for r in rewards):
        raise RuntimeError("Official World did not produce terminal rewards")
    if turns <= 0:
        raise RuntimeError("Body produced no turns")

    return {
        "turns": turns,
        "terminal_cash": terminal_cash,
        "official_rewards": rewards,
        "action_sequence_digest": digest_json(actions),
        "pre_hash_sequence_digest": digest_json(pre_hashes),
        "post_hash_sequence_digest": digest_json(post_hashes),
        "first_action": actions[0] if actions else None,
        "last_action": actions[-1] if actions else None,
    }


def main():
    first = run_once("body_smoke_first")
    second = run_once("body_smoke_second")

    deterministic = (
        first["turns"] == second["turns"]
        and first["terminal_cash"] == second["terminal_cash"]
        and first["official_rewards"] == second["official_rewards"]
        and first["action_sequence_digest"] == second["action_sequence_digest"]
        and first["pre_hash_sequence_digest"] == second["pre_hash_sequence_digest"]
        and first["post_hash_sequence_digest"] == second["post_hash_sequence_digest"]
    )

    result = {
        "schema": "relationship-surface-body-v0-runtime-smoke",
        "meaning": "Standalone Body runtime verification; not policy-strength evidence.",
        "environment_seed": ENV_SEED,
        "opponent": "Seyamalam pinned v21",
        "first_run": first,
        "second_run_after_reset": second,
        "deterministic_after_reset": deterministic,
        "passed": deterministic,
    }

    Path("relationship_surface_body_v0_runtime_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))

    if not deterministic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
