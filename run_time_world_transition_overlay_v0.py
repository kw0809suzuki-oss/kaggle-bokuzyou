#!/usr/bin/env python3
"""Time x World Transition Overlay v0.

Question:
Under the two already-observed time allocations, what actually happened in
the Official World?

This observer does NOT claim that an active Plan caused a state change.
It only places, on the same turn:
    active Plan / actual ActionBundle / Official pre-state / Official post-state
and records the raw self-relevant state delta.

No new policy is added. No downstream causal chain is inferred.

Compared runs:
- Candidate-uniform baseline: frozen Confirmed Circulation Body v0 / exclude_blocked
- Kind-uniform: Selection Kind-Uniform A/B v0 / kind_uniform

Boundary:
The runs diverge into different World trajectories. Cross-run differences are
descriptive trajectory differences, not paired causal effect estimates.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
import run_selection_kind_uniform_ab_v0 as ab
from plan_generator_entrance_v0 import bind_official_state


MISSING = {"__missing__": True}


def plain(x):
    return body.plain(x)


def self_view(raw):
    p = raw["player"]
    return {
        "farm": plain(raw["farms"][p]),
        "private": plain(raw["private"]),
    }


def path_string(parts):
    out = ""
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            if out:
                out += "."
            out += str(p)
    return out


def path_template(parts):
    out = ""
    for p in parts:
        token = "[]" if isinstance(p, int) else str(p)
        if token == "[]":
            out += token
        else:
            if out:
                out += "."
            out += token
    return out


def scalar_equal(a, b):
    # bool is intentionally distinct from int semantics in transition labels,
    # but Python equality is sufficient for exact observed-state comparison.
    return a == b


def recursive_diff(a, b, parts=()):
    diffs = []

    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) | set(b), key=lambda x: str(x))
        for k in keys:
            av = a[k] if k in a else MISSING
            bv = b[k] if k in b else MISSING
            diffs.extend(recursive_diff(av, bv, parts + (str(k),)))
        return diffs

    if isinstance(a, list) and isinstance(b, list):
        n = max(len(a), len(b))
        for i in range(n):
            av = a[i] if i < len(a) else MISSING
            bv = b[i] if i < len(b) else MISSING
            diffs.extend(recursive_diff(av, bv, parts + (i,)))
        return diffs

    if scalar_equal(a, b):
        return diffs

    row = {
        "path": path_string(parts),
        "path_template": path_template(parts),
        "before": plain(a),
        "after": plain(b),
    }
    if (
        isinstance(a, (int, float))
        and not isinstance(a, bool)
        and isinstance(b, (int, float))
        and not isinstance(b, bool)
    ):
        row["numeric_delta"] = float(b) - float(a)
    diffs.append(row)
    return diffs


def compact_value(v):
    if isinstance(v, dict) and v.get("__missing__") is True:
        return "<MISSING>"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (str, int, float)):
        return str(v)
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def transition_label(d):
    return (
        d["path_template"]
        + " | "
        + compact_value(d["before"])
        + " -> "
        + compact_value(d["after"])
    )


def bucket_for(row):
    active = row.get("active_before_action")
    if active is None:
        return "__NONACTIVE__"
    return str(active.get("kind"))


def aggregate_turn_rows(rows):
    by_bucket = {}
    raw_buckets = defaultdict(list)
    for row in rows:
        raw_buckets[row["active_bucket"]].append(row)

    for bucket, items in sorted(raw_buckets.items()):
        field_change_counts = Counter()
        field_changed_turn_counts = Counter()
        value_transition_counts = Counter()
        numeric_delta_sums = defaultdict(float)
        action_mode_counts = Counter()
        selection_turns = 0
        turns_with_any_delta = 0

        for row in items:
            action_mode_counts[str(row.get("action_mode"))] += 1
            if row.get("fresh_selection"):
                selection_turns += 1
            if row["observed_delta"]:
                turns_with_any_delta += 1

            seen_templates = set()
            for d in row["observed_delta"]:
                template = d["path_template"]
                field_change_counts[template] += 1
                seen_templates.add(template)
                value_transition_counts[transition_label(d)] += 1
                if "numeric_delta" in d:
                    numeric_delta_sums[template] += float(d["numeric_delta"])

            for template in seen_templates:
                field_changed_turn_counts[template] += 1

        by_bucket[bucket] = {
            "turns": len(items),
            "fresh_selection_turns": selection_turns,
            "turns_with_any_self_state_delta": turns_with_any_delta,
            "action_mode_counts": dict(action_mode_counts),
            "field_change_counts": dict(field_change_counts),
            "field_changed_turn_counts": dict(field_changed_turn_counts),
            "numeric_delta_sums": {
                k: v for k, v in sorted(numeric_delta_sums.items())
            },
            "value_transition_counts": dict(value_transition_counts),
        }

    return by_bucket


def replay_overlay(run):
    env = make("kaggriculture", configuration={"seed": ab.ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    rows = []
    all_hashes_match = True

    for i, row in enumerate(run["turns"]):
        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != row["pre_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} pre hash mismatch at turn {i}")

        pre_raw = pre.raw()
        pre_self = self_view(pre_raw)

        self_bundle = plain(row["action_bundle"])
        opp_bundle = plain(row["opponent_action_bundle"])

        env.step([self_bundle, opp_bundle])

        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != row["post_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} post hash mismatch at turn {i}")

        post_raw = post.raw()
        post_self = self_view(post_raw)
        observed_delta = recursive_diff(pre_self, post_self)

        active = plain(row.get("active_before_action"))
        rows.append({
            "turn": i,
            "day": int(pre_raw["day"]),
            "hour": int(pre_raw["hour"]),
            "active_bucket": (
                str(active.get("kind")) if active is not None else "__NONACTIVE__"
            ),
            "active_plan": active,
            "fresh_selection": row.get("selection_reason") is not None,
            "selection_reason": row.get("selection_reason"),
            "transition_reason": row.get("transition_reason"),
            "action_mode": row.get("action_mode"),
            "action_bundle": self_bundle,
            "observed_delta": observed_delta,
        })

    if not env.done:
        raise RuntimeError(f"{run['mode']} replay did not reach terminal")

    final = bind_official_state(env.state[0].observation)
    final_cash = body.state_summary(final)["cash"]
    terminal_equal = final_cash == run["summary"]["terminal_self_cash"]
    if not terminal_equal:
        raise RuntimeError(
            f"{run['mode']} terminal mismatch: {final_cash} != "
            f"{run['summary']['terminal_self_cash']}"
        )

    return {
        "rows": rows,
        "by_active_plan_kind": aggregate_turn_rows(rows),
        "guard": {
            "all_pre_post_hashes_match": all_hashes_match,
            "terminal_cash_equal": terminal_equal,
            "turn_count_equal": len(rows) == len(run["turns"]),
            "passed": (
                all_hashes_match
                and terminal_equal
                and len(rows) == len(run["turns"])
            ),
        },
    }


def run_summary(run, overlay):
    active_turns = Counter()
    for row in overlay["rows"]:
        active_turns[row["active_bucket"]] += 1

    all_field_changes = Counter()
    all_field_changed_turns = Counter()
    all_numeric = defaultdict(float)

    for bucket in overlay["by_active_plan_kind"].values():
        all_field_changes.update(bucket["field_change_counts"])
        all_field_changed_turns.update(bucket["field_changed_turn_counts"])
        for k, v in bucket["numeric_delta_sums"].items():
            all_numeric[k] += float(v)

    return {
        "turns": len(run["turns"]),
        "terminal_self": run["summary"]["terminal_self_cash"],
        "active_turns_by_plan_kind": dict(active_turns),
        "all_field_change_counts": dict(all_field_changes),
        "all_field_changed_turn_counts": dict(all_field_changed_turns),
        "all_numeric_delta_sums": {
            k: v for k, v in sorted(all_numeric.items())
        },
    }


def main():
    baseline = body.run_policy("exclude_blocked")
    kind_uniform = ab.run_kind_uniform()

    b = replay_overlay(baseline)
    p = replay_overlay(kind_uniform)

    if not b["guard"]["passed"]:
        raise RuntimeError("baseline replay guard failed")
    if not p["guard"]["passed"]:
        raise RuntimeError("kind-uniform replay guard failed")

    b_summary = run_summary(baseline, b)
    p_summary = run_summary(kind_uniform, p)

    result = {
        "schema": "time-world-transition-overlay-v0",
        "question": (
            "Under the two observed time allocations, what self-relevant "
            "Official World state changes actually occurred on each turn?"
        ),
        "source": {
            "baseline": "Confirmed Circulation Body v0 / exclude_blocked",
            "p1": "Selection Kind-Uniform A/B v0 / kind_uniform",
            "environment_seed": ab.ENV_SEED,
            "policy_seed": ab.POLICY_SEED,
        },
        "observation_definition": {
            "turn_overlay": (
                "active Plan + actual ActionBundle + exact pre/post self-relevant "
                "Official state delta on the same turn"
            ),
            "self_relevant_state": (
                "raw farms[player] and raw private only; clock is retained as "
                "row metadata, not counted as a state delta"
            ),
            "state_delta": (
                "recursive exact before/after differences; no Plan-to-effect "
                "causal attribution"
            ),
        },
        "baseline": {
            "summary": b_summary,
            "by_active_plan_kind": b["by_active_plan_kind"],
            "rows": b["rows"],
            "guard": b["guard"],
        },
        "kind_uniform": {
            "summary": p_summary,
            "by_active_plan_kind": p["by_active_plan_kind"],
            "rows": p["rows"],
            "guard": p["guard"],
        },
        "boundary": {
            "new_policy_added": False,
            "generator_changed": False,
            "continuation_changed": False,
            "projector_changed": False,
            "completion_changed": False,
            "active_plan_does_not_imply_cause_of_same_turn_delta": True,
            "cross_run_difference_is_not_a_paired_causal_effect": True,
            "runs_diverge_into_different_world_trajectories": True,
            "no_cash_return_chain_inferred": True,
        },
    }

    Path("time_world_transition_overlay_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep console output compact enough to inspect before reading the artifact.
    def compact(summary, overlay):
        return {
            "summary": summary,
            "by_active_plan_kind": {
                kind: {
                    "turns": row["turns"],
                    "fresh_selection_turns": row["fresh_selection_turns"],
                    "turns_with_any_self_state_delta": (
                        row["turns_with_any_self_state_delta"]
                    ),
                    "field_changed_turn_counts": row["field_changed_turn_counts"],
                    "numeric_delta_sums": row["numeric_delta_sums"],
                }
                for kind, row in overlay["by_active_plan_kind"].items()
            },
            "guard": overlay["guard"],
        }

    print("SUMMARY " + json.dumps({
        "baseline": compact(b_summary, b),
        "kind_uniform": compact(p_summary, p),
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
