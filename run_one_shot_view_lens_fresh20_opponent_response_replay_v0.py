#!/usr/bin/env python3
"""Opponent-response observation replay for One-Shot View Lens Fresh20 v0.

Question:
After the single self Selection replacement enters the shared Official World,
does the pinned opponent's observed state and/or ActionBundle diverge, and when?

This is observational replay only. Frozen Body/Lens/Harness are imported unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from kaggle_environments import make

import relationship_surface_body_v0 as promoted
import run_exclude_confirmed_blocked_ab_v0 as legacy
from one_shot_view_lens_harness_v0 import OneShotViewLensHarness
from plan_generator_entrance_v0 import bind_official_state
from run_one_shot_view_lens_fresh20_battle_replay_v0 import (
    EXPECTED_DELTA_SELF,
    LENSES,
    SEEDS,
)


def _plain(x):
    return promoted._plain(x)


def _cash_pair(snapshot):
    raw = snapshot.raw()
    farms = raw["farms"]
    return (
        float(farms[0].get("money", 0) or 0),
        float(farms[1].get("money", 0) or 0),
    )


def _run_trace(seed, lens_name=None):
    opponent = legacy.load_opponent(
        f"opponent_response_{lens_name or 'baseline'}_{seed}"
    )
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    if lens_name is None:
        agent = promoted.RelationshipSurfaceBody()
        harness = None
    else:
        harness = OneShotViewLensHarness(lens_name)
        agent = harness

    rows = []
    turn = 0
    while not env.done:
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)
        self_pre = bind_official_state(s0["observation"])
        opp_pre = bind_official_state(s1["observation"])

        self_action = agent.act(s0["observation"])
        opp_action = _plain(opponent.agent(s1["observation"]))

        env.step([_plain(self_action), opp_action])

        self_post = bind_official_state(env.state[0].observation)
        opp_post = bind_official_state(env.state[1].observation)
        if harness is not None:
            harness.observe_post(env.state[0].observation)

        rows.append({
            "turn": turn,
            "self_pre_hash": self_pre.canonical_hash,
            "opponent_pre_hash": opp_pre.canonical_hash,
            "self_action": _plain(self_action),
            "opponent_action": opp_action,
            "self_post_hash": self_post.canonical_hash,
            "opponent_post_hash": opp_post.canonical_hash,
            "pre_cash": list(_cash_pair(self_pre)),
            "post_cash": list(_cash_pair(self_post)),
        })
        turn += 1

    rewards = [float(st.reward) for st in env.state]
    return {
        "official_rewards": rewards,
        "rows": rows,
        "harness": harness.summary() if harness is not None else None,
    }


def _first(rows_a, rows_b, start, predicate):
    for i in range(start, min(len(rows_a), len(rows_b))):
        if predicate(rows_a[i], rows_b[i]):
            return i
    return None


def _row(seed, lens):
    baseline = _run_trace(seed, None)
    variant = _run_trace(seed, lens)

    if len(baseline["rows"]) != len(variant["rows"]):
        raise RuntimeError(f"seed {seed}: turn count mismatch")

    h = variant["harness"]
    if h["intervention_count"] != 1:
        raise RuntimeError(f"seed {seed}: expected exactly one intervention")
    t = int(h["intervention_log"]["turn"])

    # Exact pass-through before the intervention.
    for i in range(t):
        a, b = baseline["rows"][i], variant["rows"][i]
        if (
            a["self_pre_hash"] != b["self_pre_hash"]
            or a["opponent_pre_hash"] != b["opponent_pre_hash"]
            or a["self_action"] != b["self_action"]
            or a["opponent_action"] != b["opponent_action"]
        ):
            raise RuntimeError(
                f"seed {seed}: paired trajectory diverged before intervention at {i}"
            )

    # At the intervention boundary, both players must see the same pre-state,
    # and the opponent acts from that same input.
    ba, va = baseline["rows"][t], variant["rows"][t]
    if ba["self_pre_hash"] != va["self_pre_hash"]:
        raise RuntimeError(f"seed {seed}: self pre-state mismatch at intervention")
    if ba["opponent_pre_hash"] != va["opponent_pre_hash"]:
        raise RuntimeError(f"seed {seed}: opponent pre-state mismatch at intervention")
    if ba["opponent_action"] != va["opponent_action"]:
        raise RuntimeError(f"seed {seed}: opponent action changed before World processed Lens")

    ds = variant["official_rewards"][0] - baseline["official_rewards"][0]
    if ds != EXPECTED_DELTA_SELF[lens][seed]:
        raise RuntimeError(
            f"seed {seed} lens {lens}: Fresh20 delta_self mismatch "
            f"{ds} != {EXPECTED_DELTA_SELF[lens][seed]}"
        )

    first_world_post = _first(
        baseline["rows"], variant["rows"], t,
        lambda a, b: a["self_post_hash"] != b["self_post_hash"],
    )
    first_opp_view = _first(
        baseline["rows"], variant["rows"], t,
        lambda a, b: a["opponent_pre_hash"] != b["opponent_pre_hash"],
    )
    first_opp_action = _first(
        baseline["rows"], variant["rows"], t,
        lambda a, b: a["opponent_action"] != b["opponent_action"],
    )
    first_opp_cash = _first(
        baseline["rows"], variant["rows"], t,
        lambda a, b: a["pre_cash"][1] != b["pre_cash"][1],
    )

    delta_opp = variant["official_rewards"][1] - baseline["official_rewards"][1]

    return {
        "seed": seed,
        "intervention_turn": t,
        "first_world_post_difference_turn": first_world_post,
        "first_opponent_view_difference_turn": first_opp_view,
        "first_opponent_action_difference_turn": first_opp_action,
        "first_opponent_cash_difference_turn": first_opp_cash,
        "opponent_action_changed": first_opp_action is not None,
        "opponent_action_lag_from_intervention": (
            None if first_opp_action is None else first_opp_action - t
        ),
        "baseline_opponent_terminal": baseline["official_rewards"][1],
        "lens_opponent_terminal": variant["official_rewards"][1],
        "delta_opponent_terminal": delta_opp,
        "fresh20_self_replay_exact": True,
    }


def _stats(vals):
    if not vals:
        return None
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True, choices=LENSES)
    args = ap.parse_args()

    rows = [_row(seed, args.lens) for seed in SEEDS]
    lags = [
        r["opponent_action_lag_from_intervention"]
        for r in rows
        if r["opponent_action_lag_from_intervention"] is not None
    ]
    cash_lags = [
        r["first_opponent_cash_difference_turn"] - r["intervention_turn"]
        for r in rows
        if r["first_opponent_cash_difference_turn"] is not None
    ]

    result = {
        "schema": "one-shot-view-lens-fresh20-opponent-response-replay-v0",
        "question": (
            "After one self Selection replacement enters shared Official World, "
            "does the opponent's observed state or ActionBundle diverge, and when?"
        ),
        "lens": args.lens,
        "seeds": SEEDS,
        "aggregate": {
            "seed_count": len(rows),
            "opponent_view_changed": sum(
                r["first_opponent_view_difference_turn"] is not None for r in rows
            ),
            "opponent_action_changed": sum(
                r["opponent_action_changed"] for r in rows
            ),
            "opponent_cash_changed": sum(
                r["first_opponent_cash_difference_turn"] is not None for r in rows
            ),
            "opponent_action_lag": _stats(lags),
            "opponent_cash_lag": _stats(cash_lags),
            "fresh20_self_replay_exact_all": all(
                r["fresh20_self_replay_exact"] for r in rows
            ),
        },
        "per_seed": rows,
        "boundary": {
            "same_fresh20_worlds": True,
            "body_lens_harness_unchanged": True,
            "observation_only": True,
            "no_causal_mechanism_claim": True,
        },
    }

    out = Path(
        f"one_shot_view_lens_fresh20_opponent_response_{args.lens}_v0.json"
    )
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
