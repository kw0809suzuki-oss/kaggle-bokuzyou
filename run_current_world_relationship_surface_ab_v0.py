#!/usr/bin/env python3
"""Current-World Relationship Surface A/B v0.

Question:
Does exposing the Current World relationship structure already present in
Candidate.source_paths to Selection change Whole-Farm motion?

Baseline:
    frozen exclude_blocked Body:
    uniform Candidate Selection over selectable Candidates.

P1:
    same selectable Candidate set + Current-World Relationship Lens v0.
    At fresh Selection:
      1) derive each selectable Candidate's relationship signature from
         resolved existing source_paths,
      2) choose one signature group uniformly,
      3) choose one Candidate uniformly inside that group.

No relationship group is preferred. UNKNOWN is not penalized. Candidate kinds
may share a relationship group. No score/value/effect prediction is added.

Frozen:
Generator / BLOCKED rule / Candidate semantics / continuation / Projector /
Completion / Official World.

This is a fixed single-World probe, not an adoption decision.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
from current_world_relationship_lens_v0 import annotate_candidates
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_plan_match,
)

ENV_SEED = body.ENV_SEED
POLICY_SEED = body.POLICY_SEED
MAX_PLAN_STEPS = body.MAX_PLAN_STEPS


def plain(x):
    return body.plain(x)


def semantic_matches(plans, spec):
    return [
        p for p in plans
        if semantic_plan_match(p, kind=spec["kind"], target=spec["target"])
    ]


def signature_tuple(annotation):
    return tuple(annotation["derived_relationship_signature"])


def signature_text(sig):
    return json.dumps(list(sig), ensure_ascii=False, separators=(",", ":"))


def live_plant_count(raw):
    p = raw["player"]
    n = 0
    for row in raw["farms"][p].get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                n += 1
    return n


def nonzero_dict(d):
    return {
        str(k): float(v)
        for k, v in (d or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v
    }


def carried_total(private):
    total = 0.0
    by_item = Counter()
    for inv in private.get("inventories", []) or []:
        if not isinstance(inv, dict):
            continue
        for k, v in inv.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                total += float(v)
                by_item[str(k)] += float(v)
    return total, dict(by_item)


def tile_class(tile):
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    if isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT":
            return "PLANT:" + str(tile.get("crop"))
        return str(kind)
    return str(tile)


def run_relationship_surface():
    opponent = body.load_opponent("relationship_surface_v0")
    rng = random.Random(POLICY_SEED)
    env = make("kaggriculture", configuration={"seed": ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    active = None
    active_steps = 0
    plan_sequence = 0

    selected_counts = Counter()
    completed_counts = Counter()
    invalidated_counts = Counter()
    timeout_counts = Counter()
    blocked_active_drops = Counter()
    pass_turns = 0
    execution_errors = []

    selected_signature_counts = Counter()
    selected_kind_by_signature = defaultdict(Counter)
    selection_surfaces = []
    turns = []

    turn = 0
    while not env.done:
        s0 = env._Environment__get_shared_state(0)
        s1 = env._Environment__get_shared_state(1)
        pre = bind_official_state(s0["observation"])

        plans = generate_plans(pre)
        statuses = body.candidate_status_rows(pre, plans)
        blocked_ids = {
            r["candidate_id"] for r in statuses if r["status"] == "BLOCKED"
        }
        selectable = [p for p in plans if p.candidate_id not in blocked_ids]

        transition_reason = None
        selection_reason = None
        selected_candidate_id = None
        selected_signature = None

        if active is not None:
            matches = semantic_matches(plans, active)
            if active_steps >= MAX_PLAN_STEPS:
                timeout_counts[active["kind"]] += 1
                transition_reason = "active_plan_step_limit"
                active = None
                active_steps = 0
            elif not matches:
                invalidated_counts[active["kind"]] += 1
                transition_reason = (
                    "active_plan_no_longer_present_before_observed_completion"
                )
                active = None
                active_steps = 0
            elif body.is_blocked(pre, matches[0]):
                blocked_active_drops[active["kind"]] += 1
                transition_reason = "active_plan_now_confirmed_blocked"
                active = None
                active_steps = 0

        if active is None and selectable:
            annotations = annotate_candidates(pre, selectable)
            by_id = {a["candidate_id"]: a for a in annotations}

            groups = {}
            group_order = []
            for candidate in selectable:
                sig = signature_tuple(by_id[candidate.candidate_id])
                if sig not in groups:
                    groups[sig] = []
                    group_order.append(sig)
                groups[sig].append(candidate)

            selected_signature = group_order[rng.randrange(len(group_order))]
            within = groups[selected_signature]
            chosen = within[rng.randrange(len(within))]

            selected_candidate_id = chosen.candidate_id
            selected_sig_text = signature_text(selected_signature)
            selected_signature_counts[selected_sig_text] += 1
            selected_kind_by_signature[selected_sig_text][chosen.kind] += 1

            plan_sequence += 1
            active = {
                "sequence": plan_sequence,
                "kind": chosen.kind,
                "target": plain(chosen.target),
            }
            active_steps = 0
            selected_counts[chosen.kind] += 1
            selection_reason = (
                "uniform_relationship_signature_then_uniform_candidate_within_signature"
            )

            selection_surfaces.append({
                "turn": turn,
                "pre_hash": pre.canonical_hash,
                "candidate_count": len(plans),
                "blocked_count": len(blocked_ids),
                "selectable_count": len(selectable),
                "relationship_group_count": len(group_order),
                "relationship_groups": [
                    {
                        "signature": list(sig),
                        "candidate_count": len(groups[sig]),
                        "candidate_ids": [p.candidate_id for p in groups[sig]],
                        "candidate_kind_counts": dict(
                            Counter(p.kind for p in groups[sig])
                        ),
                    }
                    for sig in group_order
                ],
                "selected_signature": list(selected_signature),
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_kind": chosen.kind,
                "selected_candidate_target": plain(chosen.target),
            })

        current_plan = None
        current_status = None
        if active is not None:
            matches = semantic_matches(plans, active)
            if matches:
                current_plan = matches[0]
                current_status = body.first_action_status(pre, current_plan)

        if current_plan is None:
            bundle = baseline_pass_bundle(pre)
            pass_turns += 1
            action_mode = "pass_no_selected_plan"
        else:
            try:
                bundle = project_short_plan(pre, current_plan)
                action_mode = "project_active_plan"
            except Exception as exc:
                execution_errors.append({
                    "turn": turn,
                    "day": pre.raw()["day"],
                    "hour": pre.raw()["hour"],
                    "active": plain(active),
                    "error": repr(exc),
                })
                bundle = baseline_pass_bundle(pre)
                pass_turns += 1
                action_mode = "pass_projector_error"
                active = None
                active_steps = 0

        opp_bundle = plain(opponent.agent(s1["observation"]))
        pre_hash = pre.canonical_hash
        active_before = plain(active) if active is not None else None

        env.step([plain(bundle), opp_bundle])
        post = bind_official_state(env.state[0].observation)
        completion = None

        if current_plan is not None and active is not None:
            completion = completion_from_states(current_plan, pre, post)
            active_steps += 1
            if completion.get("complete"):
                completed_counts[active["kind"]] += 1
                active = None
                active_steps = 0

        turns.append({
            "turn": turn,
            "pre_hash": pre_hash,
            "candidate_count": len(plans),
            "blocked_count": len(blocked_ids),
            "selectable_count": len(selectable),
            "selected_relationship_signature":
                list(selected_signature) if selected_signature is not None else None,
            "selected_candidate_id": selected_candidate_id,
            "active_before_action": active_before,
            "current_status": plain(current_status),
            "selection_reason": selection_reason,
            "transition_reason": transition_reason,
            "action_mode": action_mode,
            "action_bundle": plain(bundle),
            "opponent_action_bundle": opp_bundle,
            "completion": plain(completion),
            "post_hash": post.canonical_hash,
        })
        turn += 1

    final = bind_official_state(env.state[0].observation)
    rewards = []
    for st in env.state:
        try:
            rewards.append(float(st.reward))
        except Exception:
            rewards.append(None)

    return {
        "mode": "relationship_surface_uniform",
        "summary": {
            "ran_to_terminal": bool(env.done),
            "turns": turn,
            "terminal_self_cash": body.state_summary(final)["cash"],
            "official_rewards": rewards,
            "selected_plans_by_kind": dict(selected_counts),
            "completed_plans_by_kind": dict(completed_counts),
            "invalidated_plans_by_kind": dict(invalidated_counts),
            "timed_out_plans_by_kind": dict(timeout_counts),
            "blocked_active_drops_by_kind": dict(blocked_active_drops),
            "pass_turns": pass_turns,
            "execution_error_count": len(execution_errors),
            "selected_relationship_signatures":
                dict(selected_signature_counts),
            "selected_kind_by_relationship_signature": {
                k: dict(v) for k, v in selected_kind_by_signature.items()
            },
        },
        "selection_surfaces": selection_surfaces,
        "turns": turns,
        "execution_errors": execution_errors,
    }


def replay_world_metrics(run):
    env = make("kaggriculture", configuration={"seed": ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    cash_outflow = 0.0
    cash_return = 0.0
    first_cash_return_turn = None
    first_cash_return_day_hour = None

    planted = Counter()
    weeded = Counter()
    watered = Counter()
    harvest_like = Counter()
    carried_positive_units = Counter()
    shed_positive_units = Counter()

    daily_boundary_live = []
    active_turns_by_kind = Counter()
    nonactive_turns = 0

    max_carried_units = 0.0
    max_shed_units = 0.0

    for i, row in enumerate(run["turns"]):
        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != row["pre_hash"]:
            raise RuntimeError(f"{run['mode']} replay pre mismatch at {i}")

        pre_raw = pre.raw()
        p = pre_raw["player"]
        pre_cash = float(pre_raw["farms"][p].get("money", 0) or 0)
        pre_carried_total, pre_carried = carried_total(pre_raw["private"])
        pre_shed = nonzero_dict(pre_raw["private"].get("shed", {}) or {})

        env.step([
            plain(row["action_bundle"]),
            plain(row["opponent_action_bundle"]),
        ])
        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != row["post_hash"]:
            raise RuntimeError(f"{run['mode']} replay post mismatch at {i}")

        post_raw = post.raw()
        post_cash = float(post_raw["farms"][p].get("money", 0) or 0)
        cash_delta = post_cash - pre_cash
        if cash_delta < 0:
            cash_outflow += -cash_delta
        elif cash_delta > 0:
            cash_return += cash_delta
            if first_cash_return_turn is None:
                first_cash_return_turn = i
                first_cash_return_day_hour = [
                    int(pre_raw["day"]),
                    int(pre_raw["hour"]),
                ]

        pre_tiles = pre_raw["farms"][p].get("tiles", []) or []
        post_tiles = post_raw["farms"][p].get("tiles", []) or []
        for y in range(len(pre_tiles)):
            for x in range(len(pre_tiles[y])):
                a = pre_tiles[y][x]
                z = post_tiles[y][x]
                ac = tile_class(a)
                zc = tile_class(z)

                if not ac.startswith("PLANT:") and zc.startswith("PLANT:"):
                    planted[zc.split(":", 1)[1]] += 1
                if ac.startswith("PLANT:") and zc == "WEED":
                    weeded[ac.split(":", 1)[1]] += 1
                if (
                    isinstance(a, dict)
                    and a.get("kind") == "PLANT"
                    and isinstance(z, dict)
                    and z.get("kind") == "PLANT"
                    and a.get("crop") == z.get("crop")
                    and not bool(a.get("watered_today", False))
                    and bool(z.get("watered_today", False))
                ):
                    watered[str(z.get("crop"))] += 1
                if ac.startswith("PLANT:") and z is None:
                    harvest_like[ac.split(":", 1)[1]] += 1

        post_carried_total, post_carried = carried_total(post_raw["private"])
        post_shed = nonzero_dict(post_raw["private"].get("shed", {}) or {})

        for item in set(pre_carried) | set(post_carried):
            d = float(post_carried.get(item, 0)) - float(pre_carried.get(item, 0))
            if d > 0:
                carried_positive_units[item] += d
        for item in set(pre_shed) | set(post_shed):
            d = float(post_shed.get(item, 0)) - float(pre_shed.get(item, 0))
            if d > 0:
                shed_positive_units[item] += d

        max_carried_units = max(max_carried_units, post_carried_total)
        max_shed_units = max(max_shed_units, sum(post_shed.values()))

        if int(post_raw["hour"]) == 0:
            daily_boundary_live.append({
                "after_turn": i,
                "day": int(post_raw["day"]),
                "live_plants": live_plant_count(post_raw),
                "cash": post_cash,
            })

        active = row.get("active_before_action")
        if active is None:
            nonactive_turns += 1
        else:
            active_turns_by_kind[str(active["kind"])] += 1

    final = bind_official_state(env.state[0].observation)
    final_raw = final.raw()
    final_p = final_raw["player"]
    final_carried_total, final_carried = carried_total(final_raw["private"])
    final_shed = nonzero_dict(final_raw["private"].get("shed", {}) or {})

    return {
        "terminal_self": body.state_summary(final)["cash"],
        "cash_outflow_total": cash_outflow,
        "cash_return_total": cash_return,
        "first_cash_return_turn": first_cash_return_turn,
        "first_cash_return_day_hour": first_cash_return_day_hour,
        "plant_established_units": sum(planted.values()),
        "plant_established_by_crop": dict(planted),
        "plant_to_weed_units": sum(weeded.values()),
        "plant_to_weed_by_crop": dict(weeded),
        "watered_false_to_true_events": sum(watered.values()),
        "watered_false_to_true_by_crop": dict(watered),
        "plant_to_null_harvest_like_events": sum(harvest_like.values()),
        "plant_to_null_harvest_like_by_crop": dict(harvest_like),
        "carried_positive_units": float(sum(carried_positive_units.values())),
        "carried_positive_by_item": dict(carried_positive_units),
        "shed_positive_units": float(sum(shed_positive_units.values())),
        "shed_positive_by_item": dict(shed_positive_units),
        "max_carried_units": max_carried_units,
        "max_shed_units": max_shed_units,
        "final_carried_units": final_carried_total,
        "final_carried_by_item": final_carried,
        "final_shed_units": float(sum(final_shed.values())),
        "final_shed_by_item": final_shed,
        "active_turns_by_kind": dict(active_turns_by_kind),
        "active_turns_total": int(sum(active_turns_by_kind.values())),
        "nonactive_turns": nonactive_turns,
        "daily_boundary_live": daily_boundary_live,
    }


def first_action_divergence(baseline, p1):
    for rb, rp in zip(baseline["turns"], p1["turns"]):
        if rb["action_bundle"] != rp["action_bundle"]:
            t = rb["turn"]
            surface = next(
                (s for s in p1["selection_surfaces"] if s["turn"] == t),
                None,
            )
            return {
                "turn": t,
                "same_pre_hash": rb["pre_hash"] == rp["pre_hash"],
                "same_opponent_action":
                    rb["opponent_action_bundle"]
                    == rp["opponent_action_bundle"],
                "baseline": {
                    "active": rb["active_before_action"],
                    "action": rb["action_bundle"],
                    "candidate_count": rb["candidate_count"],
                    "blocked_count": rb["blocked_count"],
                    "selectable_count": rb["selectable_count"],
                },
                "p1": {
                    "active": rp["active_before_action"],
                    "action": rp["action_bundle"],
                    "selected_relationship_signature":
                        rp["selected_relationship_signature"],
                    "selected_candidate_id": rp["selected_candidate_id"],
                    "relationship_surface": surface,
                },
            }
    return None


def metric_delta(a, b, key):
    return b[key] - a[key]


def main():
    baseline = body.run_policy("exclude_blocked")
    p1 = run_relationship_surface()

    bm = replay_world_metrics(baseline)
    pm = replay_world_metrics(p1)
    divergence = first_action_divergence(baseline, p1)

    delta = {
        key: metric_delta(bm, pm, key)
        for key in [
            "terminal_self",
            "cash_outflow_total",
            "cash_return_total",
            "plant_established_units",
            "plant_to_weed_units",
            "watered_false_to_true_events",
            "plant_to_null_harvest_like_events",
            "carried_positive_units",
            "shed_positive_units",
            "active_turns_total",
            "nonactive_turns",
        ]
    }

    result = {
        "schema": "current-world-relationship-surface-ab-v0",
        "question": (
            "Does exposing Current World relationship structure already "
            "present in Candidate.source_paths to Selection change Whole-Farm motion?"
        ),
        "environment": {
            "seed": ENV_SEED,
            "policy_seed": POLICY_SEED,
            "opponent": "Seyamalam pinned v21",
        },
        "comparison": {
            "baseline":
                "frozen exclude_blocked Body / uniform selectable Candidate selection",
            "p1": (
                "same selectable Candidate set; uniform relationship-signature "
                "group, then uniform Candidate within selected group"
            ),
            "generator_changed": False,
            "blocked_rule_changed": False,
            "candidate_semantics_changed": False,
            "active_plan_continuation_changed": False,
            "projector_changed": False,
            "completion_changed": False,
            "value_score_added": False,
            "effect_prediction_added": False,
            "relationship_group_preference_added": False,
        },
        "baseline": {
            "summary": baseline["summary"],
            "world_metrics": bm,
        },
        "p1": {
            "summary": p1["summary"],
            "world_metrics": pm,
        },
        "delta": delta,
        "first_action_divergence": divergence,
        "boundary": {
            "single_fixed_world_only": True,
            "not_an_adoption_decision": True,
            "does_not_establish_relationship_surface_as_correct_policy": True,
            "does_not_establish_return_aware_policy": True,
            "terminal_is_final_judgment_metric": True,
        },
    }

    Path("current_world_relationship_surface_ab_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "baseline": bm,
        "p1": pm,
        "delta": delta,
        "p1_selected_relationship_signatures":
            p1["summary"]["selected_relationship_signatures"],
        "p1_selected_kind_by_relationship_signature":
            p1["summary"]["selected_kind_by_relationship_signature"],
        "first_action_divergence": divergence,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
