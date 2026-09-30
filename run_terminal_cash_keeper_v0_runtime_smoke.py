#!/usr/bin/env python3
"""Runtime smoke for Terminal Cash Keeper v0.

This verifies executable behavior and determinism on one World.
It is not strength evidence and it is not an A/B promotion gate.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from kaggle_environments import make

import terminal_cash_keeper_v0 as keeper
import run_exclude_confirmed_blocked_ab_v0 as legacy
from plan_generator_entrance_v0 import bind_official_state

ENV_SEED = 7391


def digest_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_once(tag):
    keeper.reset_agent()
    opponent = legacy.load_opponent(tag)
    env = make("kaggriculture", configuration={"seed": ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    actions = []
    decision_kinds = Counter()
    turns = 0

    while not env.done:
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)

        action = keeper.agent(s0["observation"], env.configuration)
        opponent_action = legacy.plain(opponent.agent(s1["observation"]))

        if not isinstance(action, dict):
            raise TypeError(f"keeper returned {type(action).__name__}, not dict")
        for key in ("farmer", "hands", "market"):
            if key not in action:
                raise KeyError(f"ActionBundle missing {key}")

        decision = keeper.last_decision or {}
        decision_kinds[str(decision.get("kind", "UNKNOWN"))] += 1
        actions.append(legacy.plain(action))
        env.step([legacy.plain(action), opponent_action])
        turns += 1

    final = bind_official_state(env.state[0].observation)
    rewards = [float(st.reward) for st in env.state]
    terminal_cash = legacy.state_summary(final)["cash"]

    return {
        "turns": turns,
        "terminal_cash": terminal_cash,
        "official_rewards": rewards,
        "action_sequence_digest": digest_json(actions),
        "decision_kind_counts": dict(sorted(decision_kinds.items())),
        "first_action": actions[0] if actions else None,
        "last_action": actions[-1] if actions else None,
    }


def main():
    first = run_once("terminal_cash_keeper_first")
    second = run_once("terminal_cash_keeper_second")

    deterministic = (
        first["turns"] == second["turns"]
        and first["terminal_cash"] == second["terminal_cash"]
        and first["official_rewards"] == second["official_rewards"]
        and first["action_sequence_digest"] == second["action_sequence_digest"]
        and first["decision_kind_counts"] == second["decision_kind_counts"]
    )

    result = {
        "schema": "terminal-cash-keeper-v0-runtime-smoke",
        "meaning": "Executable/determinism verification only; not policy-strength evidence.",
        "environment_seed": ENV_SEED,
        "opponent": "Seyamalam pinned v21",
        "first_run": first,
        "second_run_after_reset": second,
        "deterministic_after_reset": deterministic,
        "passed": deterministic,
    }

    Path("terminal_cash_keeper_v0_runtime_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))
    if not deterministic:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
