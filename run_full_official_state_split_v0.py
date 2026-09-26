#!/usr/bin/env python3
"""Full Official State Split v0.

Question:
Was the Official World really still identical immediately before turn3?

Window:
    turn0 -> turn3 inclusive

Method:
Replay the already-fixed Candidate-uniform baseline and Kind-uniform run.
At each turn, compare the COMPLETE Official bound state, not an economic/plant
projection:

    Full Pre-State -> actual ActionBundle -> Full Post-State

Then report the first Action difference, first full Pre-State difference, and
first full Post-State difference, together with exact raw field paths.

No Selection / maintenance / WATER explanation is added.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
import run_selection_kind_uniform_ab_v0 as ab
from plan_generator_entrance_v0 import bind_official_state

LAST_TURN = 3
MISSING = {"__missing__": True}


def plain(x):
    return body.plain(x)


def full_raw(snapshot):
    return plain(snapshot.raw())


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


def recursive_diff(a, b, parts=()):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=lambda x: str(x)):
            av = a[k] if k in a else MISSING
            bv = b[k] if k in b else MISSING
            out.extend(recursive_diff(av, bv, parts + (str(k),)))
        return out

    if isinstance(a, list) and isinstance(b, list):
        n = max(len(a), len(b))
        for i in range(n):
            av = a[i] if i < len(a) else MISSING
            bv = b[i] if i < len(b) else MISSING
            out.extend(recursive_diff(av, bv, parts + (i,)))
        return out

    if a != b:
        out.append({
            "path": path_string(parts),
            "baseline": plain(a),
            "kind_uniform": plain(b),
        })
    return out


def replay(run):
    env = make("kaggriculture", configuration={"seed": ab.ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    rows = []
    all_hashes_match = True

    for i, src in enumerate(run["turns"]):
        if i > LAST_TURN:
            break

        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != src["pre_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} pre hash mismatch at turn {i}")

        pre_raw = full_raw(pre)
        action = plain(src["action_bundle"])
        opp = plain(src["opponent_action_bundle"])

        env.step([action, opp])

        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != src["post_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} post hash mismatch at turn {i}")

        rows.append({
            "turn": i,
            "pre_hash": pre.canonical_hash,
            "pre_state": pre_raw,
            "action_bundle": action,
            "opponent_action_bundle": opp,
            "post_hash": post.canonical_hash,
            "post_state": full_raw(post),
        })

    return {
        "rows": rows,
        "guard": {
            "all_replayed_hashes_match_source": all_hashes_match,
            "turn_count": len(rows),
            "expected_turn_count": LAST_TURN + 1,
            "passed": all_hashes_match and len(rows) == LAST_TURN + 1,
        },
    }


def first_turn(pairs, key):
    for p in pairs:
        if not p[key]:
            return p["turn"]
    return None


def compact_pair(p):
    return {
        "turn": p["turn"],
        "pre_hash_equal": p["pre_hash_equal"],
        "action_equal": p["action_equal"],
        "post_hash_equal": p["post_hash_equal"],
        "pre_state_diff": p["pre_state_diff"],
        "post_state_diff": p["post_state_diff"],
        "baseline_action": p["baseline_action"],
        "kind_uniform_action": p["kind_uniform_action"],
    }


def main():
    baseline_run = body.run_policy("exclude_blocked")
    kind_run = ab.run_kind_uniform()

    b = replay(baseline_run)
    k = replay(kind_run)

    if not b["guard"]["passed"]:
        raise RuntimeError("baseline guard failed")
    if not k["guard"]["passed"]:
        raise RuntimeError("kind-uniform guard failed")

    pairs = []
    for rb, rk in zip(b["rows"], k["rows"]):
        if rb["turn"] != rk["turn"]:
            raise RuntimeError("turn alignment mismatch")

        pre_diff = recursive_diff(rb["pre_state"], rk["pre_state"])
        post_diff = recursive_diff(rb["post_state"], rk["post_state"])

        pairs.append({
            "turn": rb["turn"],
            "pre_hash_equal": rb["pre_hash"] == rk["pre_hash"],
            "pre_state_equal": not pre_diff,
            "action_equal": rb["action_bundle"] == rk["action_bundle"],
            "post_hash_equal": rb["post_hash"] == rk["post_hash"],
            "post_state_equal": not post_diff,
            "pre_state_diff": pre_diff,
            "post_state_diff": post_diff,
            "baseline_action": rb["action_bundle"],
            "kind_uniform_action": rk["action_bundle"],
        })

    first_action = first_turn(pairs, "action_equal")
    first_pre = first_turn(pairs, "pre_state_equal")
    first_post = first_turn(pairs, "post_state_equal")

    key_turns = sorted(set(
        t for t in [first_action, first_pre, first_post, 2, 3]
        if t is not None
    ))
    by_turn = {p["turn"]: p for p in pairs}

    result = {
        "schema": "full-official-state-split-v0",
        "question": (
            "Was the complete Official World still identical immediately "
            "before turn3?"
        ),
        "source": {
            "baseline": "Confirmed Circulation Body v0 / exclude_blocked",
            "kind_uniform": "Selection Kind-Uniform A/B v0 / kind_uniform",
            "environment_seed": ab.ENV_SEED,
            "policy_seed": ab.POLICY_SEED,
        },
        "first_observed_separations": {
            "first_action_bundle_difference_turn": first_action,
            "first_full_official_pre_state_difference_turn": first_pre,
            "first_full_official_post_state_difference_turn": first_post,
        },
        "key_turns": {
            str(t): compact_pair(by_turn[t]) for t in key_turns
        },
        "paired_turns": [compact_pair(p) for p in pairs],
        "guards": {
            "baseline": b["guard"],
            "kind_uniform": k["guard"],
        },
        "boundary": {
            "full_official_state_compared": True,
            "tracked_projection_used_for_equality": False,
            "selection_used_as_explanation": False,
            "maintenance_used_as_explanation": False,
            "water_used_as_explanation": False,
            "causal_claim_made": False,
            "new_policy_added": False,
        },
    }

    Path("full_official_state_split_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "first_observed_separations": result["first_observed_separations"],
        "key_turns": result["key_turns"],
        "guards": result["guards"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
