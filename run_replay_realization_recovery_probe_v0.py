#!/usr/bin/env python3
"""Replay Realization Recovery Probe v0.

Parent objective:
  terminal self を高く残す。

Baseline:
  Re-run the recorded requested actions of BOTH players from Episode 115521888.

Candidate:
  Change exactly one subject market request at action step 15:
    BUY_SEED WHEAT 1 -> SELL WHEAT 1
  Then return immediately to the recorded action stream from step 16 onward.

This is a one-shot causal probe, not a promoted policy.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

PARTS = [
    "replay_realization_fixture_v0_part0.txt",
    "replay_realization_fixture_v0_part1.txt",
    "replay_realization_fixture_v0_part2.txt",
    "replay_realization_fixture_v0_part3.txt",
]
FIXTURE_SHA256 = "be97a1ae891e36a9a387a52eae314fba2ffb8f0db12b4f0174cf29c608c3b33b"
TARGET_ACTION_STEP = 15
EXPECTED_ORIGINAL_MARKET = [["BUY_SEED", "WHEAT", 1]]
REPLACEMENT_MARKET = [["SELL", "WHEAT", 1]]
EXPECTED_SOURCE_REWARDS = [55918.0, 110521.0]


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load_fixture():
    payload = "".join(Path(p).read_text(encoding="utf-8").strip() for p in PARTS)
    raw = zlib.decompress(base64.b64decode(payload))
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise RuntimeError(f"fixture sha mismatch: {got}")
    return json.loads(raw.decode("utf-8"))


def shared(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def subject_snapshot(env):
    obs = shared(env, 0)
    farm = obs["farms"][0]
    priv = obs.get("private", {})
    return {
        "step": int(obs.get("step", 0) or 0),
        "cash": float(farm.get("money", 0) or 0),
        "wheat_seed": int((priv.get("seeds", {}) or {}).get("WHEAT", 0) or 0),
        "shed_wheat": int((priv.get("shed", {}) or {}).get("WHEAT", 0) or 0),
    }


def run(mode, fixture):
    env = make("kaggriculture", configuration={"seed": fixture["seed"]}, debug=False)
    env.reset(num_agents=2)

    intervention_count = 0
    trace = {}

    while not env.done:
        step = int(shared(env, 0).get("step", 0) or 0)
        if step >= len(fixture["actions"]):
            raise RuntimeError(f"fixture exhausted at step {step}")

        a0 = deepcopy(fixture["actions"][step][0])
        a1 = deepcopy(fixture["actions"][step][1])

        if step in (14, 15, 16, 17):
            trace[f"pre_step_{step}"] = subject_snapshot(env)

        if step == TARGET_ACTION_STEP:
            if a0.get("market", []) != EXPECTED_ORIGINAL_MARKET:
                raise RuntimeError(
                    f"unexpected source action at step15: {a0.get('market', [])}"
                )
            if mode == "candidate":
                a0["market"] = deepcopy(REPLACEMENT_MARKET)
                intervention_count += 1

        env.step([a0, a1])

        if step in (14, 15, 16, 17):
            trace[f"post_action_step_{step}"] = subject_snapshot(env)

    rewards = [float(env.state[0].reward), float(env.state[1].reward)]
    return {
        "mode": mode,
        "rewards": rewards,
        "terminal_self": rewards[0],
        "terminal_opponent": rewards[1],
        "intervention_count": intervention_count,
        "trace": trace,
    }


def main():
    fixture = load_fixture()
    baseline = run("baseline", fixture)

    baseline_exact = baseline["rewards"] == EXPECTED_SOURCE_REWARDS
    if not baseline_exact:
        out = {
            "schema": "replay-realization-recovery-probe-v0",
            "status": "INVALID_BASELINE_FIDELITY",
            "baseline": baseline,
            "expected_source_rewards": EXPECTED_SOURCE_REWARDS,
        }
        Path("replay_realization_recovery_probe_v0_result.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("REPLAY_REALIZATION_BASELINE_FAIL " + json.dumps(out, separators=(",", ":")))
        raise SystemExit(2)

    candidate = run("candidate", fixture)
    delta = candidate["terminal_self"] - baseline["terminal_self"]
    decision = "IMPROVED" if delta > 0 else "WORSENED" if delta < 0 else "SAME"

    out = {
        "schema": "replay-realization-recovery-probe-v0",
        "status": "VALID",
        "source": {
            "episode_id": fixture["source_episode_id"],
            "seed": fixture["seed"],
            "agents": fixture["agents"],
            "fixture_sha256": FIXTURE_SHA256,
            "executable_action_steps": len(fixture["actions"]),
        },
        "intervention": {
            "subject_seat": 0,
            "action_step": TARGET_ACTION_STEP,
            "original_market": EXPECTED_ORIGINAL_MARKET,
            "replacement_market": REPLACEMENT_MARKET,
            "one_shot": True,
            "return_to_recorded_stream_from_step16": True,
        },
        "baseline_fidelity": {
            "expected_rewards": EXPECTED_SOURCE_REWARDS,
            "observed_rewards": baseline["rewards"],
            "exact": baseline_exact,
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta_terminal_self": delta,
        "decision": decision,
        "boundary": {
            "same_seed": True,
            "same_recorded_opponent_requested_actions": True,
            "same_subject_requested_actions_except_one_market_replacement": True,
            "no_adaptive_policy_added": True,
            "single_world_only": True,
            "causality_not_generalized": True,
        },
    }

    Path("replay_realization_recovery_probe_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("REPLAY_REALIZATION_RECOVERY_RESULT " + json.dumps({
        "baseline_terminal_self": baseline["terminal_self"],
        "candidate_terminal_self": candidate["terminal_self"],
        "delta_terminal_self": delta,
        "decision": decision,
        "intervention_count": candidate["intervention_count"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
