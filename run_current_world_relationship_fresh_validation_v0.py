#!/usr/bin/env python3
"""Fresh Terminal Validation v0 for Current-World Relationship Surface.

Purpose:
Ask fresh Official Worlds whether the fixed-world effect of exposing
Current-World relationship signatures to Selection reaches terminal self.

The observation rule is fixed before seeing fresh results.

Outer / judgment surface:
- terminal self
- cash outflow / cash return / first cash return
- plant established / plant->weed / harvest-like
- per-turn tracked Official projection + canonical hashes

Inner / exploration surface:
- Candidate Set
- BLOCKED-filtered selectable set
- relationship groups from existing source_paths
- selected Candidate
- active Plan origin
- actual Action

Synchronization:
- turn
- Official pre_hash

No causal claim is made by temporal ordering alone.

Usage:
  KAGGRI_ENV_SEEDS=7101 python run_current_world_relationship_fresh_validation_v0.py
  KAGGRI_ENV_SEEDS=7102,7103,... python ...

Policy seed and opponent remain fixed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
import run_current_world_relationship_surface_ab_v0 as surface
from current_world_relationship_lens_v0 import annotate_candidates
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import semantic_plan_match

POLICY_SEED = body.POLICY_SEED


def plain(x):
    return body.plain(x)


def parse_seeds():
    raw = os.environ.get("KAGGRI_ENV_SEEDS", "7101")
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("KAGGRI_ENV_SEEDS produced no seeds")
    if len(set(out)) != len(out):
        raise ValueError("duplicate seeds are not allowed")
    return out


def inventory_totals(private):
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


def tracked_outer(snapshot):
    raw = snapshot.raw()
    p = raw["player"]
    farm = raw["farms"][p]
    private = raw["private"]

    plants = Counter()
    weeds = 0
    empty = 0
    locked = 0
    watered = 0
    unwatered = 0

    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile is None:
                empty += 1
            elif tile == "LOCKED":
                locked += 1
            elif isinstance(tile, dict):
                if tile.get("kind") == "WEED":
                    weeds += 1
                elif tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop"))
                    plants[crop] += 1
                    if bool(tile.get("watered_today", False)):
                        watered += 1
                    else:
                        unwatered += 1

    carried_total, carried_by_item = inventory_totals(private)
    shed = {
        str(k): float(v)
        for k, v in (private.get("shed", {}) or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v
    }
    seeds = {
        str(k): float(v)
        for k, v in (private.get("seeds", {}) or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v
    }

    return {
        "day": int(raw["day"]),
        "hour": int(raw["hour"]),
        "cash": float(farm.get("money", 0) or 0),
        "farmer": plain(farm.get("farmer")),
        "live_plants": int(sum(plants.values())),
        "plants_by_crop": dict(plants),
        "watered_live": watered,
        "unwatered_live": unwatered,
        "weeds": weeds,
        "empty_tiles": empty,
        "locked_tiles": locked,
        "seeds": seeds,
        "carried_total": carried_total,
        "carried_by_item": carried_by_item,
        "shed_total": float(sum(shed.values())),
        "shed_by_item": shed,
    }


def signature_tuple(annotation):
    return tuple(annotation["derived_relationship_signature"])


def signature_text(sig):
    return json.dumps(list(sig), ensure_ascii=False, separators=(",", ":"))


def selected_candidate_from_active(selectable, active):
    if active is None:
        return None
    matches = [
        p for p in selectable
        if semantic_plan_match(p, kind=active["kind"], target=active["target"])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "fresh Selection active spec did not resolve to exactly one "
            f"selectable Candidate: {len(matches)}"
        )
    return matches[0]


def instrument_run(run, env_seed):
    env = make(
        "kaggriculture",
        configuration={"seed": env_seed},
        debug=False,
    )
    env.reset(num_agents=2)

    selection_events = {}
    trace = []
    all_hashes_match = True
    source_path_integrity = True

    for i, row in enumerate(run["turns"]):
        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != row["pre_hash"]:
            all_hashes_match = False
            raise RuntimeError(
                f"{run['mode']} seed {env_seed} pre hash mismatch turn {i}"
            )

        plans = generate_plans(pre)
        statuses = body.candidate_status_rows(pre, plans)
        blocked_ids = {
            r["candidate_id"] for r in statuses if r["status"] == "BLOCKED"
        }
        selectable = [p for p in plans if p.candidate_id not in blocked_ids]

        fresh_selection = row.get("selection_reason") is not None
        selection_event = None

        if fresh_selection:
            annotations = annotate_candidates(pre, selectable)
            if len(annotations) != len(selectable):
                source_path_integrity = False
                raise RuntimeError("annotation count != selectable count")

            by_id = {a["candidate_id"]: a for a in annotations}
            groups = {}
            group_order = []
            for cand in selectable:
                ann = by_id[cand.candidate_id]
                sig = signature_tuple(ann)
                if sig not in groups:
                    groups[sig] = []
                    group_order.append(sig)
                groups[sig].append(cand)

            chosen = selected_candidate_from_active(
                selectable, row["active_before_action"]
            )
            chosen_ann = by_id[chosen.candidate_id]
            chosen_sig = signature_tuple(chosen_ann)

            sequence = int(row["active_before_action"]["sequence"])
            selection_event = {
                "selection_turn": i,
                "pre_hash": pre.canonical_hash,
                "sequence": sequence,
                "candidate_count": len(plans),
                "blocked_count": len(blocked_ids),
                "selectable_count": len(selectable),
                "candidate_ids_in_order": [p.candidate_id for p in plans],
                "selectable_candidate_ids_in_order": [
                    p.candidate_id for p in selectable
                ],
                "relationship_groups": [
                    {
                        "signature": list(sig),
                        "candidate_count": len(groups[sig]),
                        "candidate_kind_counts": dict(
                            Counter(p.kind for p in groups[sig])
                        ),
                    }
                    for sig in group_order
                ],
                "selected_relationship_signature": list(chosen_sig),
                "selected_candidate_id": chosen.candidate_id,
                "selected_candidate_kind": chosen.kind,
                "selected_candidate_target": plain(chosen.target),
            }
            selection_events[sequence] = selection_event

        active = row.get("active_before_action")
        active_origin = None
        if active is not None:
            seq = int(active["sequence"])
            origin = selection_events.get(seq)
            if origin is not None:
                active_origin = {
                    "selection_turn": origin["selection_turn"],
                    "pre_hash": origin["pre_hash"],
                    "selected_relationship_signature":
                        origin["selected_relationship_signature"],
                    "selected_candidate_id": origin["selected_candidate_id"],
                    "selected_candidate_kind": origin["selected_candidate_kind"],
                    "selected_candidate_target":
                        origin["selected_candidate_target"],
                }

        pre_outer = tracked_outer(pre)

        env.step([
            plain(row["action_bundle"]),
            plain(row["opponent_action_bundle"]),
        ])
        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != row["post_hash"]:
            all_hashes_match = False
            raise RuntimeError(
                f"{run['mode']} seed {env_seed} post hash mismatch turn {i}"
            )

        trace.append({
            "turn": i,
            "pre_hash": pre.canonical_hash,
            "pre_outer": pre_outer,
            "fresh_selection": fresh_selection,
            "selection_event": selection_event,
            "active_before_action": plain(active),
            "active_origin": active_origin,
            "action_bundle": plain(row["action_bundle"]),
            "post_hash": post.canonical_hash,
            "post_outer": tracked_outer(post),
        })

    return {
        "trace": trace,
        "selection_events": [
            selection_events[k] for k in sorted(selection_events)
        ],
        "guard": {
            "all_replayed_hashes_match": all_hashes_match,
            "source_path_annotation_integrity": source_path_integrity,
            "turn_count": len(trace),
            "passed": all_hashes_match and source_path_integrity
                and len(trace) == len(run["turns"]),
        },
    }


def first_difference(btrace, ptrace, key_fn):
    for b, p in zip(btrace, ptrace):
        if key_fn(b) != key_fn(p):
            return b["turn"]
    return None


def first_fresh_selection_difference(btrace, ptrace):
    for b, p in zip(btrace, ptrace):
        be = b.get("selection_event")
        pe = p.get("selection_event")
        if be is None and pe is None:
            continue
        if (be is None) != (pe is None):
            return {
                "turn": b["turn"],
                "baseline": be,
                "p1": pe,
                "kind": "fresh_selection_presence_difference",
            }
        if (
            be["selected_candidate_id"] != pe["selected_candidate_id"]
            or be["selected_relationship_signature"]
            != pe["selected_relationship_signature"]
        ):
            return {
                "turn": b["turn"],
                "same_pre_hash": b["pre_hash"] == p["pre_hash"],
                "baseline": be,
                "p1": pe,
                "kind": "selected_candidate_or_relationship_difference",
            }
    return None


def first_action_difference(btrace, ptrace):
    for b, p in zip(btrace, ptrace):
        if b["action_bundle"] != p["action_bundle"]:
            return {
                "turn": b["turn"],
                "same_pre_hash": b["pre_hash"] == p["pre_hash"],
                "baseline": {
                    "active_origin": b["active_origin"],
                    "action": b["action_bundle"],
                },
                "p1": {
                    "active_origin": p["active_origin"],
                    "action": p["action_bundle"],
                },
            }
    return None


def divergence_layers(btrace, ptrace):
    first_selected = first_fresh_selection_difference(btrace, ptrace)
    first_action = first_action_difference(btrace, ptrace)

    first_pre_hash = first_difference(
        btrace, ptrace, lambda r: r["pre_hash"]
    )
    first_post_hash = first_difference(
        btrace, ptrace, lambda r: r["post_hash"]
    )
    first_outer_post = first_difference(
        btrace, ptrace, lambda r: r["post_outer"]
    )
    first_cash_post = first_difference(
        btrace, ptrace, lambda r: r["post_outer"]["cash"]
    )

    times = {
        "fresh_selection_difference":
            first_selected["turn"] if first_selected else None,
        "action_difference":
            first_action["turn"] if first_action else None,
        "full_official_pre_state_difference": first_pre_hash,
        "full_official_post_state_difference": first_post_hash,
        "tracked_outer_post_state_difference": first_outer_post,
        "cash_post_state_difference": first_cash_post,
    }

    observed_turn_gaps = {}
    keys = list(times)
    for a in keys:
        for b in keys:
            if (
                a < b
                and times[a] is not None
                and times[b] is not None
            ):
                observed_turn_gaps[f"{a}__to__{b}"] = (
                    times[b] - times[a]
                )

    return {
        "times": times,
        "observed_turn_gaps_no_causal_claim": observed_turn_gaps,
        "first_fresh_selection_difference": first_selected,
        "first_action_difference": first_action,
    }


def one_seed(seed):
    # Same Body and same policy RNG seed; only Official World seed changes.
    body.ENV_SEED = seed
    surface.ENV_SEED = seed

    baseline = body.run_policy("exclude_blocked")
    p1 = surface.run_relationship_surface()

    bm = surface.replay_world_metrics(baseline)
    pm = surface.replay_world_metrics(p1)

    bi = instrument_run(baseline, seed)
    pi = instrument_run(p1, seed)
    if not bi["guard"]["passed"] or not pi["guard"]["passed"]:
        raise RuntimeError(f"instrumentation guard failed seed {seed}")

    layers = divergence_layers(bi["trace"], pi["trace"])

    terminal_delta = pm["terminal_self"] - bm["terminal_self"]

    return {
        "seed": seed,
        "baseline": {
            "summary": baseline["summary"],
            "world_metrics": bm,
            "instrument": bi,
        },
        "p1": {
            "summary": p1["summary"],
            "world_metrics": pm,
            "instrument": pi,
        },
        "delta": {
            "terminal_self": terminal_delta,
            "cash_outflow_total":
                pm["cash_outflow_total"] - bm["cash_outflow_total"],
            "cash_return_total":
                pm["cash_return_total"] - bm["cash_return_total"],
            "plant_established_units":
                pm["plant_established_units"] - bm["plant_established_units"],
            "plant_to_weed_units":
                pm["plant_to_weed_units"] - bm["plant_to_weed_units"],
            "plant_to_null_harvest_like_events":
                pm["plant_to_null_harvest_like_events"]
                - bm["plant_to_null_harvest_like_events"],
        },
        "direction": (
            "IMPROVED" if terminal_delta > 0
            else "WORSE" if terminal_delta < 0
            else "SAME"
        ),
        "divergence_layers": layers,
    }


def aggregate(results):
    ds = [r["delta"]["terminal_self"] for r in results]
    improved = sum(1 for x in ds if x > 0)
    worse = sum(1 for x in ds if x < 0)
    same = sum(1 for x in ds if x == 0)
    return {
        "seed_count": len(results),
        "improved": improved,
        "worse": worse,
        "same": same,
        "mean_delta_terminal": sum(ds) / len(ds),
        "min_delta_terminal": min(ds),
        "max_delta_terminal": max(ds),
        "per_seed_terminal_delta": {
            str(r["seed"]): r["delta"]["terminal_self"]
            for r in results
        },
    }


def main():
    seeds = parse_seeds()
    results = [one_seed(seed) for seed in seeds]

    result = {
        "schema": "current-world-relationship-fresh-validation-v0",
        "question": (
            "Does exposing Current World relationship structure to Selection "
            "reach terminal self in fresh Official Worlds?"
        ),
        "fixed_before_results": {
            "environment_seeds": seeds,
            "policy_seed": POLICY_SEED,
            "opponent": "Seyamalam pinned v21",
            "baseline":
                "frozen exclude_blocked Body / uniform selectable Candidate",
            "p1": (
                "same selectable Candidate set; uniform relationship signature, "
                "then uniform Candidate within selected signature"
            ),
            "sync_keys": ["turn", "pre_hash"],
            "outer_judgment": "terminal_self",
            "outer_context_metrics": [
                "cash_outflow_total",
                "cash_return_total",
                "first_cash_return",
                "plant_established_units",
                "plant_to_weed_units",
                "plant_to_null_harvest_like_events",
            ],
            "inner_exploration": [
                "Candidate Set",
                "BLOCKED-filtered selectable set",
                "relationship groups",
                "selected Candidate",
                "active Plan origin",
                "Action",
            ],
            "causal_claim_from_temporal_order": False,
        },
        "aggregate": aggregate(results),
        "results": results,
        "boundary": {
            "fresh_world_validation": True,
            "terminal_is_judgment_surface": True,
            "inner_trace_is_exploration_surface": True,
            "no_policy_adoption_decision_encoded": True,
            "no_result_dependent_instrument_change": True,
        },
    }

    Path("current_world_relationship_fresh_validation_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "fixed_before_results": result["fixed_before_results"],
        "aggregate": result["aggregate"],
        "per_seed": [
            {
                "seed": r["seed"],
                "direction": r["direction"],
                "delta": r["delta"],
                "baseline_terminal":
                    r["baseline"]["world_metrics"]["terminal_self"],
                "p1_terminal":
                    r["p1"]["world_metrics"]["terminal_self"],
                "baseline_first_return":
                    r["baseline"]["world_metrics"]["first_cash_return_turn"],
                "p1_first_return":
                    r["p1"]["world_metrics"]["first_cash_return_turn"],
                "divergence_times":
                    r["divergence_layers"]["times"],
            }
            for r in results
        ],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
