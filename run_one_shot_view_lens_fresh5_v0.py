#!/usr/bin/env python3
"""Paired Fresh5 runner for One-Shot View Lens Harness v0."""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from kaggle_environments import make

import relationship_surface_body_v0 as promoted
import run_exclude_confirmed_blocked_ab_v0 as legacy
from one_shot_view_lens_harness_v0 import OneShotViewLensHarness
from plan_generator_entrance_v0 import bind_official_state

DEFAULT_SEEDS = [7401, 7402, 7403, 7404, 7405]
LENSES = ("nearby_density", "relationship_age", "exit_proximity", "disabled")

def _plain(x):
    return promoted._plain(x)

def _tile_class(tile):
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    if isinstance(tile, dict):
        if tile.get("kind") == "PLANT":
            return "PLANT"
        return str(tile.get("kind"))
    return str(tile)

def _metric_state():
    return {
        "cash_outflow": 0.0,
        "cash_return": 0.0,
        "plant": 0,
        "weed": 0,
        "harvest_like": 0,
    }

def _update_metrics(metrics, pre, post):
    a = pre.raw()
    z = post.raw()
    p = a["player"]

    pre_cash = float(a["farms"][p].get("money", 0) or 0)
    post_cash = float(z["farms"][p].get("money", 0) or 0)
    d = post_cash - pre_cash
    if d < 0:
        metrics["cash_outflow"] += -d
    elif d > 0:
        metrics["cash_return"] += d

    pre_tiles = a["farms"][p].get("tiles", []) or []
    post_tiles = z["farms"][p].get("tiles", []) or []
    for y in range(min(len(pre_tiles), len(post_tiles))):
        for x in range(min(len(pre_tiles[y]), len(post_tiles[y]))):
            before = _tile_class(pre_tiles[y][x])
            after = _tile_class(post_tiles[y][x])
            if before != "PLANT" and after == "PLANT":
                metrics["plant"] += 1
            if before == "PLANT" and after == "WEED":
                metrics["weed"] += 1
            if before == "PLANT" and post_tiles[y][x] is None:
                metrics["harvest_like"] += 1

def _run_episode(seed, lens_name=None):
    opponent = legacy.load_opponent(
        f"one_shot_{lens_name or 'baseline'}_{seed}"
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
    metrics = _metric_state()
    turn = 0

    while not env.done:
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)
        pre = bind_official_state(s0["observation"])

        action = agent.act(s0["observation"])
        opp_action = _plain(opponent.agent(s1["observation"]))
        env.step([_plain(action), opp_action])

        post = bind_official_state(env.state[0].observation)
        if harness is not None:
            harness.observe_post(env.state[0].observation)

        _update_metrics(metrics, pre, post)
        rows.append({
            "turn": turn,
            "pre_hash": pre.canonical_hash,
            "action": _plain(action),
            "post_hash": post.canonical_hash,
        })
        turn += 1

    final = bind_official_state(env.state[0].observation)
    terminal = legacy.state_summary(final)["cash"]
    rewards = []
    for st in env.state:
        try:
            rewards.append(float(st.reward))
        except Exception:
            rewards.append(None)

    if any(r is None for r in rewards):
        raise RuntimeError(f"seed {seed}: missing terminal reward")

    return {
        "terminal": terminal,
        "official_rewards": rewards,
        "metrics": metrics,
        "rows": rows,
        "harness": harness.summary() if harness is not None else None,
    }

def _first_propagation(baseline_rows, lens_rows, start_turn):
    for i in range(start_turn, min(len(baseline_rows), len(lens_rows))):
        b = baseline_rows[i]
        l = lens_rows[i]
        action_diff = b["action"] != l["action"]
        post_diff = b["post_hash"] != l["post_hash"]
        if action_diff or post_diff:
            return {
                "turn": i,
                "action_diff": action_diff,
                "post_state_diff": post_diff,
            }
    return None

def _delta_metrics(a, b):
    return {k: b[k] - a[k] for k in a}

def _paired_seed(seed, lens_name):
    baseline = _run_episode(seed, None)
    variant = _run_episode(seed, lens_name)

    if len(baseline["rows"]) != len(variant["rows"]):
        raise RuntimeError(f"seed {seed}: paired runs have different turn counts")

    if lens_name == "disabled":
        exact = (
            baseline["terminal"] == variant["terminal"]
            and baseline["official_rewards"] == variant["official_rewards"]
            and baseline["rows"] == variant["rows"]
        )
        return {
            "seed": seed,
            "disabled_parity": exact,
            "baseline_terminal": baseline["terminal"],
            "variant_terminal": variant["terminal"],
            "delta_terminal": variant["terminal"] - baseline["terminal"],
            "harness": variant["harness"],
        }

    h = variant["harness"]
    if h["intervention_count"] > 1:
        raise RuntimeError(f"seed {seed}: more than one intervention")

    reached = h["intervention_count"] == 1
    intervention = h["intervention_log"]
    if not reached:
        return {
            "seed": seed,
            "reachability": False,
            "propagation": False,
            "terminal_changed": False,
            "baseline_terminal": baseline["terminal"],
            "variant_terminal": variant["terminal"],
            "delta_terminal": variant["terminal"] - baseline["terminal"],
            "delta_outer": _delta_metrics(
                baseline["metrics"], variant["metrics"]
            ),
            "intervention": None,
            "first_propagation": None,
        }

    t = int(intervention["turn"])
    if baseline["rows"][t]["pre_hash"] != variant["rows"][t]["pre_hash"]:
        raise RuntimeError(f"seed {seed}: intervention pre-state mismatch")

    # Before the one-shot point, the wrapper must be exact pass-through.
    for i in range(t):
        if baseline["rows"][i] != variant["rows"][i]:
            raise RuntimeError(
                f"seed {seed}: trajectory diverged before intervention at {i}"
            )

    if not intervention["body_rng_unchanged"]:
        raise RuntimeError(f"seed {seed}: Body RNG changed inside lens")

    propagation = _first_propagation(
        baseline["rows"], variant["rows"], t
    )
    delta_terminal = variant["terminal"] - baseline["terminal"]

    return {
        "seed": seed,
        "reachability": True,
        "propagation": propagation is not None,
        "terminal_changed": delta_terminal != 0,
        "baseline_terminal": baseline["terminal"],
        "variant_terminal": variant["terminal"],
        "delta_terminal": delta_terminal,
        "baseline_outer": baseline["metrics"],
        "variant_outer": variant["metrics"],
        "delta_outer": _delta_metrics(
            baseline["metrics"], variant["metrics"]
        ),
        "intervention": intervention,
        "first_propagation": propagation,
    }

def _aggregate(rows, lens_name):
    if lens_name == "disabled":
        return {
            "seed_count": len(rows),
            "disabled_parity_passed": all(
                r["disabled_parity"] for r in rows
            ),
        }

    deltas = [r["delta_terminal"] for r in rows]
    return {
        "seed_count": len(rows),
        "reachable": sum(r["reachability"] for r in rows),
        "propagated": sum(r["propagation"] for r in rows),
        "terminal_changed": sum(r["terminal_changed"] for r in rows),
        "terminal_improved": sum(d > 0 for d in deltas),
        "terminal_worse": sum(d < 0 for d in deltas),
        "terminal_same": sum(d == 0 for d in deltas),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True, choices=LENSES)
    ap.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
    )
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = [_paired_seed(seed, args.lens) for seed in seeds]
    aggregate = _aggregate(rows, args.lens)

    result = {
        "schema": "one-shot-view-lens-fresh5-v0",
        "lens": args.lens,
        "seeds": seeds,
        "frozen_body": "Relationship Surface Body v0",
        "primary_judgment": "terminal_self",
        "stages": ["Reachability", "Propagation", "Terminal"],
        "aggregate": aggregate,
        "per_seed": rows,
        "boundary": {
            "one_shot_max_per_battle": True,
            "body_rng_noninterference_required": True,
            "no_mechanism_claim": True,
            "fresh5_is_world_contact_screen_not_winner_selection": True,
        },
    }

    out = Path(f"one_shot_view_lens_fresh5_{args.lens}_v0.json")
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps({
        "lens": args.lens,
        "aggregate": aggregate,
        "per_seed": [
            {
                "seed": r["seed"],
                "reachability": r.get("reachability"),
                "propagation": r.get("propagation"),
                "delta_terminal": r["delta_terminal"],
                "disabled_parity": r.get("disabled_parity"),
                "intervention_turn": (
                    r.get("intervention") or {}
                ).get("turn"),
            }
            for r in rows
        ],
    }, separators=(",", ":")))

    if args.lens == "disabled" and not aggregate["disabled_parity_passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
