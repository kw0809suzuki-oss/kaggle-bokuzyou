#!/usr/bin/env python3
"""Current-World Relationship Lens Shadow v0.

Runs the frozen exclude_blocked Body unchanged in behavior, while invoking
Current-World Relationship Lens v0 only at fresh Selection opportunities.

The Lens is observation-only. It resolves existing candidate.source_paths and
stores raw Current World provenance plus a derived relationship signature.

Required non-interference guards:
- Candidate ID order unchanged by Lens
- selectable Candidate ID order unchanged by Lens
- selected Candidate equals frozen Body
- ActionBundle sequence equals frozen Body
- Official pre/post hashes equal frozen Body
- terminal equals frozen Body
- RNG state unchanged across Lens invocation

No relationship signature is used for Selection in this Shadow run.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
from current_world_relationship_lens_v0 import (
    LensIntegrityError,
    annotate_candidates,
)
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


def state_digest(rng):
    return hashlib.sha256(repr(rng.getstate()).encode("utf-8")).hexdigest()


def semantic_matches(plans, spec):
    return [
        p for p in plans
        if semantic_plan_match(p, kind=spec["kind"], target=spec["target"])
    ]


def signature_key(annotation):
    sig = annotation["derived_relationship_signature"]
    return json.dumps(sig, ensure_ascii=False, separators=(",", ":"))


def run_shadow():
    opponent = body.load_opponent("current_world_relationship_lens_shadow_v0")
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

    selection_surfaces = []
    turns = []

    all_candidate_order_unchanged = True
    all_selectable_order_unchanged = True
    all_rng_unchanged_by_lens = True
    lens_integrity_failure_count = 0

    signature_counts = Counter()
    source_path_type_counts = Counter()
    normalized_fact_counts = Counter()
    candidates_with_unknown_source_path_type = 0
    annotated_candidate_count = 0

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
            # Shadow-only Lens invocation. Selection itself remains frozen.
            candidate_ids_before = [p.candidate_id for p in plans]
            selectable_ids_before = [p.candidate_id for p in selectable]
            rng_before = state_digest(rng)

            try:
                annotations = annotate_candidates(pre, selectable)
            except LensIntegrityError:
                lens_integrity_failure_count += 1
                raise

            rng_after = state_digest(rng)
            candidate_ids_after = [p.candidate_id for p in plans]
            selectable_ids_after = [p.candidate_id for p in selectable]

            candidate_order_unchanged = (
                candidate_ids_before == candidate_ids_after
            )
            selectable_order_unchanged = (
                selectable_ids_before == selectable_ids_after
            )
            rng_unchanged = rng_before == rng_after

            all_candidate_order_unchanged &= candidate_order_unchanged
            all_selectable_order_unchanged &= selectable_order_unchanged
            all_rng_unchanged_by_lens &= rng_unchanged

            if not candidate_order_unchanged:
                raise RuntimeError("Lens changed Candidate ID order")
            if not selectable_order_unchanged:
                raise RuntimeError("Lens changed selectable Candidate ID order")
            if not rng_unchanged:
                raise RuntimeError("Lens consumed or mutated RNG state")

            for a in annotations:
                annotated_candidate_count += 1
                signature_counts[signature_key(a)] += 1
                if a["has_unknown_source_path_type"]:
                    candidates_with_unknown_source_path_type += 1
                for ref in a["authoritative_provenance"]:
                    source_path_type_counts[ref["source_path_type"]] += 1
                    normalized_fact_counts[ref["normalized_fact"]] += 1

            # Exact frozen exclude_blocked Selection:
            # same full C_t RNG stream; reject confirmed BLOCKED only.
            while True:
                drawn = plans[rng.randrange(len(plans))]
                if drawn.candidate_id not in blocked_ids:
                    chosen = drawn
                    break

            selected_candidate_id = chosen.candidate_id
            plan_sequence += 1
            active = {
                "sequence": plan_sequence,
                "kind": chosen.kind,
                "target": plain(chosen.target),
            }
            active_steps = 0
            selected_counts[chosen.kind] += 1
            selection_reason = "coupled_rng_reject_only_confirmed_blocked"

            selection_surfaces.append({
                "turn": turn,
                "pre_hash": pre.canonical_hash,
                "candidate_ids_in_order": candidate_ids_before,
                "blocked_candidate_ids": sorted(blocked_ids),
                "selectable_candidate_ids_in_order": selectable_ids_before,
                "annotations": annotations,
                "selected_candidate_id": selected_candidate_id,
                "guards": {
                    "candidate_id_order_unchanged_by_lens":
                        candidate_order_unchanged,
                    "selectable_id_order_unchanged_by_lens":
                        selectable_order_unchanged,
                    "rng_state_unchanged_by_lens": rng_unchanged,
                },
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

    return {
        "mode": "current_world_relationship_lens_shadow_v0",
        "summary": {
            "ran_to_terminal": bool(env.done),
            "turns": turn,
            "terminal_self_cash": body.state_summary(final)["cash"],
            "selected_plans_by_kind": dict(selected_counts),
            "completed_plans_by_kind": dict(completed_counts),
            "invalidated_plans_by_kind": dict(invalidated_counts),
            "timed_out_plans_by_kind": dict(timeout_counts),
            "blocked_active_drops_by_kind": dict(blocked_active_drops),
            "pass_turns": pass_turns,
            "execution_error_count": len(execution_errors),
        },
        "lens_summary": {
            "selection_surface_count": len(selection_surfaces),
            "annotated_candidate_count": annotated_candidate_count,
            "signature_counts": dict(signature_counts),
            "source_path_type_counts": dict(source_path_type_counts),
            "normalized_fact_counts": dict(normalized_fact_counts),
            "candidates_with_unknown_source_path_type":
                candidates_with_unknown_source_path_type,
            "lens_integrity_failure_count": lens_integrity_failure_count,
        },
        "internal_noninterference": {
            "all_candidate_id_orders_unchanged_by_lens":
                all_candidate_order_unchanged,
            "all_selectable_id_orders_unchanged_by_lens":
                all_selectable_order_unchanged,
            "all_rng_states_unchanged_by_lens":
                all_rng_unchanged_by_lens,
        },
        "selection_surfaces": selection_surfaces,
        "turns": turns,
        "execution_errors": execution_errors,
    }


def matching_candidate_id(catalog, active_spec):
    if active_spec is None:
        return None
    matches = [
        a["candidate_id"] for a in catalog
        if a["kind"] == active_spec["kind"]
        and a["target"] == active_spec["target"]
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one selected Candidate match, got {len(matches)}"
        )
    return matches[0]


def compare_to_frozen(frozen, shadow):
    if len(frozen["turns"]) != len(shadow["turns"]):
        raise RuntimeError("turn count differs from frozen Body")

    surface_by_turn = {
        row["turn"]: row for row in shadow["selection_surfaces"]
    }

    action_equal = True
    pre_hash_equal = True
    post_hash_equal = True
    opponent_action_equal = True
    active_spec_equal = True
    selected_candidate_equal = True
    selected_candidate_checks = 0
    first_difference = None

    for fr, sh in zip(frozen["turns"], shadow["turns"]):
        t = fr["turn"]
        checks = {
            "pre_hash": fr["pre_hash"] == sh["pre_hash"],
            "action_bundle": fr["action_bundle"] == sh["action_bundle"],
            "post_hash": fr["post_hash"] == sh["post_hash"],
            "opponent_action":
                fr["opponent_action_bundle"] == sh["opponent_action_bundle"],
            "active_spec":
                fr["active_before_action"] == sh["active_before_action"],
            "candidate_count": fr["candidate_count"] == sh["candidate_count"],
            "blocked_count": fr["blocked_count"] == sh["blocked_count"],
            "selectable_count": fr["selectable_count"] == sh["selectable_count"],
        }

        pre_hash_equal &= checks["pre_hash"]
        action_equal &= checks["action_bundle"]
        post_hash_equal &= checks["post_hash"]
        opponent_action_equal &= checks["opponent_action"]
        active_spec_equal &= checks["active_spec"]

        if fr.get("selection_reason") is not None:
            selected_candidate_checks += 1
            surface = surface_by_turn.get(t)
            if surface is None:
                selected_candidate_equal = False
            else:
                frozen_id = matching_candidate_id(
                    surface["annotations"],
                    fr["active_before_action"],
                )
                same = frozen_id == sh["selected_candidate_id"]
                selected_candidate_equal &= same
                checks["selected_candidate_id"] = same

        if first_difference is None and not all(checks.values()):
            first_difference = {"turn": t, "checks": checks}

    terminal_equal = (
        frozen["summary"]["terminal_self_cash"]
        == shadow["summary"]["terminal_self_cash"]
    )

    internal = shadow["internal_noninterference"]
    guard = {
        "candidate_id_order_unchanged_by_lens":
            internal["all_candidate_id_orders_unchanged_by_lens"],
        "selectable_id_order_unchanged_by_lens":
            internal["all_selectable_id_orders_unchanged_by_lens"],
        "rng_state_unchanged_by_lens":
            internal["all_rng_states_unchanged_by_lens"],
        "selected_candidate_sequence_equal_to_frozen":
            selected_candidate_equal,
        "selected_candidate_checks": selected_candidate_checks,
        "action_bundle_sequence_equal_to_frozen": action_equal,
        "official_pre_hash_sequence_equal_to_frozen": pre_hash_equal,
        "official_post_hash_sequence_equal_to_frozen": post_hash_equal,
        "opponent_action_sequence_equal_to_frozen": opponent_action_equal,
        "active_plan_sequence_equal_to_frozen": active_spec_equal,
        "terminal_equal_to_frozen": terminal_equal,
        "lens_integrity_failure_count":
            shadow["lens_summary"]["lens_integrity_failure_count"],
        "first_difference": first_difference,
    }
    guard["passed"] = (
        guard["candidate_id_order_unchanged_by_lens"]
        and guard["selectable_id_order_unchanged_by_lens"]
        and guard["rng_state_unchanged_by_lens"]
        and guard["selected_candidate_sequence_equal_to_frozen"]
        and guard["action_bundle_sequence_equal_to_frozen"]
        and guard["official_pre_hash_sequence_equal_to_frozen"]
        and guard["official_post_hash_sequence_equal_to_frozen"]
        and guard["opponent_action_sequence_equal_to_frozen"]
        and guard["active_plan_sequence_equal_to_frozen"]
        and guard["terminal_equal_to_frozen"]
        and guard["lens_integrity_failure_count"] == 0
        and guard["first_difference"] is None
    )
    return guard


def main():
    frozen = body.run_policy("exclude_blocked")
    shadow = run_shadow()
    guard = compare_to_frozen(frozen, shadow)

    if not guard["passed"]:
        raise RuntimeError(
            "Current-World Relationship Lens Shadow non-interference failed: "
            + json.dumps(guard, ensure_ascii=False)
        )

    result = {
        "schema": "current-world-relationship-lens-shadow-v0",
        "purpose": (
            "Externalize the Current World provenance already present in "
            "Candidate.source_paths without changing frozen Body behavior."
        ),
        "lens_contract": {
            "authoritative_output":
                "resolved existing source_path + Current World value",
            "derived_signature_from_current_state_only": True,
            "effect_prediction_added": False,
            "value_score_added": False,
            "candidate_added_or_removed_by_lens": False,
            "selection_uses_relationship_signature": False,
        },
        "frozen_body": {
            "terminal_self_cash": frozen["summary"]["terminal_self_cash"],
            "turns": len(frozen["turns"]),
        },
        "shadow": shadow,
        "noninterference_guard": guard,
        "boundary": {
            "shadow_only": True,
            "not_a_policy_probe": True,
            "not_an_adoption_decision": True,
            "return_aware_not_claimed": True,
        },
    }

    Path("current_world_relationship_lens_shadow_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "frozen_terminal": frozen["summary"]["terminal_self_cash"],
        "shadow_terminal": shadow["summary"]["terminal_self_cash"],
        "lens_summary": shadow["lens_summary"],
        "noninterference_guard": guard,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
