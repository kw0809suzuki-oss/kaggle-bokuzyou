#!/usr/bin/env python3
"""Battle-side observation replay for One-Shot View Lens Fresh20 v0.

This is not a new Lens experiment. It replays the exact Fresh20 worlds and
interventions to recover the opponent side that the original artifact dropped.

Frozen components are imported unchanged. The replay is accepted only when
per-seed delta_self exactly matches the already observed Fresh20 delta_self.
"""

from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

from run_one_shot_view_lens_fresh5_v0 import _run_episode

SEEDS = list(range(7501, 7521))
LENSES = ("nearby_density", "relationship_age", "exit_proximity")

EXPECTED_DELTA_SELF = {
    "nearby_density": {
        7501:-647.0,7502:-887.0,7503:-724.0,7504:-15.0,7505:-649.0,
        7506:-677.0,7507:-796.0,7508:447.0,7509:366.0,7510:-1568.0,
        7511:-390.0,7512:-1415.0,7513:-67.0,7514:-739.0,7515:-201.0,
        7516:-829.0,7517:-751.0,7518:7.0,7519:552.0,7520:-630.0,
    },
    "relationship_age": {
        7501:-772.0,7502:-506.0,7503:-745.0,7504:-22.0,7505:-639.0,
        7506:-622.0,7507:-740.0,7508:-95.0,7509:-494.0,7510:-1609.0,
        7511:-168.0,7512:-1171.0,7513:-62.0,7514:-760.0,7515:-69.0,
        7516:-766.0,7517:-759.0,7518:68.0,7519:-20.0,7520:-726.0,
    },
    "exit_proximity": {
        7501:-794.0,7502:-876.0,7503:-746.0,7504:1635.0,7505:882.0,
        7506:-672.0,7507:-750.0,7508:-260.0,7509:-516.0,7510:-1601.0,
        7511:-377.0,7512:-1413.0,7513:-61.0,7514:-823.0,7515:28.0,
        7516:-779.0,7517:-781.0,7518:1608.0,7519:-138.0,7520:-792.0,
    },
}

def _outcome(self_reward, opp_reward):
    if self_reward > opp_reward:
        return "WIN"
    if self_reward < opp_reward:
        return "LOSS"
    return "TIE"

def _stats(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }

def _row(seed, lens):
    baseline = _run_episode(seed, None)
    variant = _run_episode(seed, lens)

    b_self = float(baseline["terminal"])
    l_self = float(variant["terminal"])
    b_rewards = [float(x) for x in baseline["official_rewards"]]
    l_rewards = [float(x) for x in variant["official_rewards"]]

    if b_self != b_rewards[0]:
        raise RuntimeError(
            f"seed {seed}: baseline terminal self != Official reward[0]"
        )
    if l_self != l_rewards[0]:
        raise RuntimeError(
            f"seed {seed}: lens terminal self != Official reward[0]"
        )

    delta_self = l_self - b_self
    expected = EXPECTED_DELTA_SELF[lens][seed]
    if delta_self != expected:
        raise RuntimeError(
            f"seed {seed} lens {lens}: Fresh20 self replay mismatch "
            f"{delta_self} != {expected}"
        )

    b_opp = b_rewards[1]
    l_opp = l_rewards[1]
    b_margin = b_self - b_opp
    l_margin = l_self - l_opp
    delta_opp = l_opp - b_opp
    delta_margin = l_margin - b_margin

    b_outcome = _outcome(b_self, b_opp)
    l_outcome = _outcome(l_self, l_opp)

    return {
        "seed": seed,
        "baseline_self": b_self,
        "baseline_opponent": b_opp,
        "lens_self": l_self,
        "lens_opponent": l_opp,
        "delta_self": delta_self,
        "delta_opponent": delta_opp,
        "baseline_margin": b_margin,
        "lens_margin": l_margin,
        "delta_margin": delta_margin,
        "baseline_outcome": b_outcome,
        "lens_outcome": l_outcome,
        "outcome_changed": b_outcome != l_outcome,
        "fresh20_self_replay_exact": True,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True, choices=LENSES)
    args = ap.parse_args()

    rows = [_row(seed, args.lens) for seed in SEEDS]
    ds = [r["delta_self"] for r in rows]
    do = [r["delta_opponent"] for r in rows]
    dm = [r["delta_margin"] for r in rows]

    aggregate = {
        "seed_count": len(rows),
        "fresh20_self_replay_exact_all": all(
            r["fresh20_self_replay_exact"] for r in rows
        ),
        "delta_self": {
            "improved": sum(x > 0 for x in ds),
            "worse": sum(x < 0 for x in ds),
            "same": sum(x == 0 for x in ds),
            **_stats(ds),
        },
        "delta_opponent": {
            "increased": sum(x > 0 for x in do),
            "decreased": sum(x < 0 for x in do),
            "same": sum(x == 0 for x in do),
            **_stats(do),
        },
        "delta_margin": {
            "improved": sum(x > 0 for x in dm),
            "worse": sum(x < 0 for x in dm),
            "same": sum(x == 0 for x in dm),
            **_stats(dm),
        },
        "outcome_changes": sum(r["outcome_changed"] for r in rows),
        "baseline_outcomes": {
            k: sum(r["baseline_outcome"] == k for r in rows)
            for k in ("WIN", "LOSS", "TIE")
        },
        "lens_outcomes": {
            k: sum(r["lens_outcome"] == k for r in rows)
            for k in ("WIN", "LOSS", "TIE")
        },
    }

    result = {
        "schema": "one-shot-view-lens-fresh20-battle-observation-replay-v0",
        "meaning": (
            "Same Fresh20 worlds and frozen Lens intervention replayed only to "
            "recover opponent terminal, margin, and battle outcome."
        ),
        "source_fresh20_run_id": 36257952351,
        "lens": args.lens,
        "seeds": SEEDS,
        "primary_observation": [
            "baseline_self",
            "baseline_opponent",
            "lens_self",
            "lens_opponent",
            "delta_self",
            "delta_opponent",
            "baseline_margin",
            "lens_margin",
            "delta_margin",
            "baseline_outcome",
            "lens_outcome",
        ],
        "aggregate": aggregate,
        "per_seed": rows,
        "boundary": {
            "not_a_new_lens_experiment": True,
            "same_worlds_as_fresh20": True,
            "same_lens_definitions": True,
            "same_one_shot_rules": True,
            "mechanism_interpretation": False,
        },
    }

    out = Path(
        f"one_shot_view_lens_fresh20_battle_replay_{args.lens}_v0.json"
    )
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps({
        "lens": args.lens,
        "aggregate": aggregate,
        "per_seed": rows,
    }, separators=(",", ":")))

if __name__ == "__main__":
    main()
