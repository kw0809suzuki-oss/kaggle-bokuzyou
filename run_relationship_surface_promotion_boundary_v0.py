#!/usr/bin/env python3
"""Promotion Boundary Probe v0 for Current-World Relationship Surface.

Question:
Does the already-fixed Relationship Surface effect reproduce broadly enough
to use it as the next observation baseline Body?

This is NOT a mechanism probe.
This does NOT claim the Relationship Surface is a true/correct policy.

Frozen comparison:
Baseline:
  frozen exclude_blocked Body / uniform selectable Candidate selection

P1:
  same selectable Candidate set; uniformly select relationship signature,
  then uniformly select one Candidate inside the selected signature group

Fresh Worlds:
  7201..7230 (30 previously unused environment seeds)

Primary judgment:
  terminal self

Pre-registered promotion rule:
  PROMOTE_AS_OBSERVATION_BASELINE iff ALL are true:
    1) improved_count > worse_count
    2) median_delta_terminal > 0
    3) mean_delta_terminal > 0
  otherwise HOLD.

The rule intentionally contains no effect-size threshold. Promotion only means
"adequate as the next observation baseline under current evidence".

Supplementary observations:
  per-seed terminal delta
  cash outflow / cash return
  plant established / plant->weed / harvest-like

No Candidate / divergence / inner-mechanism analysis is performed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import run_exclude_confirmed_blocked_ab_v0 as body
import run_current_world_relationship_surface_ab_v0 as surface

SEEDS = list(range(7201, 7231))
POLICY_SEED = body.POLICY_SEED

PROMOTION_RULE = {
    "rule_id": "promotion-boundary-v0",
    "decision": (
        "PROMOTE_AS_OBSERVATION_BASELINE iff "
        "improved_count > worse_count AND "
        "median_delta_terminal > 0 AND "
        "mean_delta_terminal > 0; otherwise HOLD"
    ),
    "meaning_of_promotion": (
        "Use Relationship Surface as the next observation baseline Body "
        "under current evidence; not a claim of truth or mechanism."
    ),
}


def run_one(seed: int):
    body.ENV_SEED = seed
    surface.ENV_SEED = seed

    baseline = body.run_policy("exclude_blocked")
    p1 = surface.run_relationship_surface()

    bm = surface.replay_world_metrics(baseline)
    pm = surface.replay_world_metrics(p1)

    if baseline["summary"].get("execution_error_count", 0) != 0:
        raise RuntimeError(f"baseline execution error seed {seed}")
    if p1["summary"].get("execution_error_count", 0) != 0:
        raise RuntimeError(f"p1 execution error seed {seed}")
    if not baseline["summary"].get("ran_to_terminal"):
        raise RuntimeError(f"baseline not terminal seed {seed}")
    if not p1["summary"].get("ran_to_terminal"):
        raise RuntimeError(f"p1 not terminal seed {seed}")

    delta_terminal = pm["terminal_self"] - bm["terminal_self"]

    return {
        "seed": seed,
        "baseline_terminal": bm["terminal_self"],
        "p1_terminal": pm["terminal_self"],
        "delta_terminal": delta_terminal,
        "direction": (
            "IMPROVED" if delta_terminal > 0
            else "WORSE" if delta_terminal < 0
            else "SAME"
        ),
        "baseline": {
            "cash_outflow_total": bm["cash_outflow_total"],
            "cash_return_total": bm["cash_return_total"],
            "first_cash_return_turn": bm["first_cash_return_turn"],
            "plant_established_units": bm["plant_established_units"],
            "plant_to_weed_units": bm["plant_to_weed_units"],
            "harvest_like_events":
                bm["plant_to_null_harvest_like_events"],
        },
        "p1": {
            "cash_outflow_total": pm["cash_outflow_total"],
            "cash_return_total": pm["cash_return_total"],
            "first_cash_return_turn": pm["first_cash_return_turn"],
            "plant_established_units": pm["plant_established_units"],
            "plant_to_weed_units": pm["plant_to_weed_units"],
            "harvest_like_events":
                pm["plant_to_null_harvest_like_events"],
        },
        "delta_context": {
            "cash_outflow_total":
                pm["cash_outflow_total"] - bm["cash_outflow_total"],
            "cash_return_total":
                pm["cash_return_total"] - bm["cash_return_total"],
            "plant_established_units":
                pm["plant_established_units"] - bm["plant_established_units"],
            "plant_to_weed_units":
                pm["plant_to_weed_units"] - bm["plant_to_weed_units"],
            "harvest_like_events":
                pm["plant_to_null_harvest_like_events"]
                - bm["plant_to_null_harvest_like_events"],
        },
    }


def aggregate(rows):
    deltas = [r["delta_terminal"] for r in rows]
    improved = sum(d > 0 for d in deltas)
    worse = sum(d < 0 for d in deltas)
    same = sum(d == 0 for d in deltas)
    mean_delta = statistics.fmean(deltas)
    median_delta = statistics.median(deltas)

    promotion_checks = {
        "improved_count_gt_worse_count": improved > worse,
        "median_delta_terminal_gt_zero": median_delta > 0,
        "mean_delta_terminal_gt_zero": mean_delta > 0,
    }
    promoted = all(promotion_checks.values())

    context_keys = [
        "cash_outflow_total",
        "cash_return_total",
        "plant_established_units",
        "plant_to_weed_units",
        "harvest_like_events",
    ]
    mean_context_delta = {
        key: statistics.fmean(
            r["delta_context"][key] for r in rows
        )
        for key in context_keys
    }

    return {
        "seed_count": len(rows),
        "improved": improved,
        "worse": worse,
        "same": same,
        "mean_delta_terminal": mean_delta,
        "median_delta_terminal": median_delta,
        "min_delta_terminal": min(deltas),
        "max_delta_terminal": max(deltas),
        "promotion_checks": promotion_checks,
        "promotion_decision": (
            "PROMOTE_AS_OBSERVATION_BASELINE"
            if promoted else "HOLD"
        ),
        "mean_context_delta": mean_context_delta,
    }


def main():
    rows = [run_one(seed) for seed in SEEDS]
    agg = aggregate(rows)

    result = {
        "schema": "relationship-surface-promotion-boundary-v0",
        "question": (
            "Does the fixed Relationship Surface effect reproduce broadly "
            "enough to use it as the next observation baseline Body?"
        ),
        "fixed_before_results": {
            "seeds": SEEDS,
            "policy_seed": POLICY_SEED,
            "opponent": "Seyamalam pinned v21",
            "baseline":
                "frozen exclude_blocked Body / uniform selectable Candidate",
            "p1": (
                "same selectable Candidate set; uniform relationship signature, "
                "then uniform Candidate within selected signature"
            ),
            "primary_judgment": "terminal_self",
            "promotion_rule": PROMOTION_RULE,
            "no_inner_mechanism_analysis": True,
            "no_result_dependent_rule_change": True,
        },
        "aggregate": agg,
        "per_seed": rows,
        "boundary": {
            "promotion_is_only_next_observation_baseline": True,
            "not_policy_truth_claim": True,
            "not_mechanism_claim": True,
            "not_return_priority_claim": True,
        },
    }

    Path("relationship_surface_promotion_boundary_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "promotion_rule": PROMOTION_RULE,
        "aggregate": agg,
        "per_seed_terminal": [
            {
                "seed": r["seed"],
                "baseline": r["baseline_terminal"],
                "p1": r["p1_terminal"],
                "delta": r["delta_terminal"],
                "direction": r["direction"],
            }
            for r in rows
        ],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
