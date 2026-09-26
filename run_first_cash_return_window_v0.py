#!/usr/bin/env python3
"""First Cash Return Window v0.

Question:
Before the first positive Cash transition appears, what did each whole farm
actually become in the Official World?

This is a World-first observer.
It does not use Plan kind as the explanatory coordinate.

Compared runs:
- Candidate-uniform baseline: frozen Confirmed Circulation Body v0 / exclude_blocked
- Kind-uniform: Selection Kind-Uniform A/B v0 / kind_uniform

We replay both fixed runs, preserve exact Official pre/post hashes, and record
only compact whole-farm World state plus raw observed deltas.

Two views are emitted:
1) each run from Initial State -> its own first positive Cash transition;
2) same-clock comparisons at both first-return turns.

Boundary:
- A raw World delta is observation, not a causal attribution.
- Cross-run differences are descriptive trajectory differences.
- Similar inventory quantities are not treated as identity-tracked objects.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
import run_selection_kind_uniform_ab_v0 as ab
from plan_generator_entrance_v0 import bind_official_state


def plain(x):
    return body.plain(x)


def nz_numeric(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
            out[str(k)] = float(v)
    return out


def aggregate_inventory(private):
    out = Counter()
    for inv in private.get("inventories", []) or []:
        if not isinstance(inv, dict):
            continue
        for k, v in inv.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v:
                out[str(k)] += float(v)
    return out


def tile_class(tile):
    if tile is None:
        return "EMPTY"
    if tile == "LOCKED":
        return "LOCKED"
    if isinstance(tile, dict):
        kind = str(tile.get("kind"))
        if kind == "PLANT":
            return "PLANT:" + str(tile.get("crop"))
        return kind
    return str(tile)


def surface_summary(raw):
    p = raw["player"]
    c = Counter()
    plants_by_crop = Counter()
    watered = 0
    unwatered = 0
    positive_yield_by_crop = Counter()

    for row in raw["farms"][p].get("tiles", []) or []:
        for tile in row:
            cls = tile_class(tile)
            if cls.startswith("PLANT:"):
                c["PLANT"] += 1
                crop = cls.split(":", 1)[1]
                plants_by_crop[crop] += 1
                if bool(tile.get("watered_today", False)):
                    watered += 1
                else:
                    unwatered += 1
                y = float(tile.get("yield_units", 0) or 0)
                if y > 0:
                    positive_yield_by_crop[crop] += y
            else:
                c[cls] += 1

    return {
        "surface": dict(c),
        "plants_by_crop": dict(plants_by_crop),
        "watered_plants": watered,
        "unwatered_plants": unwatered,
        "yield_units_by_crop": dict(positive_yield_by_crop),
        "yield_units_total": float(sum(positive_yield_by_crop.values())),
    }


def world_snapshot(snapshot):
    raw = snapshot.raw()
    p = raw["player"]
    farm = raw["farms"][p]
    private = raw["private"]
    surf = surface_summary(raw)

    return {
        "day": int(raw["day"]),
        "hour": int(raw["hour"]),
        "cash": float(farm.get("money", 0) or 0),
        "seeds": nz_numeric(private.get("seeds", {}) or {}),
        "carried": dict(aggregate_inventory(private)),
        "shed": nz_numeric(private.get("shed", {}) or {}),
        **surf,
    }


def map_delta(before, after):
    out = {}
    keys = sorted(set(before) | set(after))
    for k in keys:
        a = float(before.get(k, 0) or 0)
        b = float(after.get(k, 0) or 0)
        d = b - a
        if d:
            out[str(k)] = d
    return out


def tile_deltas(pre_raw, post_raw):
    p = pre_raw["player"]
    before = pre_raw["farms"][p].get("tiles", []) or []
    after = post_raw["farms"][p].get("tiles", []) or []

    surface = Counter()
    plant_established = Counter()
    plant_to_weed = Counter()
    watered_false_to_true = Counter()
    watered_true_to_false = Counter()
    yield_positive = Counter()
    yield_negative = Counter()

    for y in range(min(len(before), len(after))):
        for x in range(min(len(before[y]), len(after[y]))):
            a = before[y][x]
            b = after[y][x]
            ac = tile_class(a)
            bc = tile_class(b)

            if ac != bc:
                surface[f"{ac}->{bc}"] += 1

            if not ac.startswith("PLANT:") and bc.startswith("PLANT:"):
                plant_established[bc.split(":", 1)[1]] += 1

            if ac.startswith("PLANT:") and bc == "WEED":
                plant_to_weed[ac.split(":", 1)[1]] += 1

            if (
                isinstance(a, dict) and a.get("kind") == "PLANT"
                and isinstance(b, dict) and b.get("kind") == "PLANT"
                and str(a.get("crop")) == str(b.get("crop"))
            ):
                crop = str(a.get("crop"))

                aw = bool(a.get("watered_today", False))
                bw = bool(b.get("watered_today", False))
                if (not aw) and bw:
                    watered_false_to_true[crop] += 1
                elif aw and (not bw):
                    watered_true_to_false[crop] += 1

                ay = float(a.get("yield_units", 0) or 0)
                by = float(b.get("yield_units", 0) or 0)
                yd = by - ay
                if yd > 0:
                    yield_positive[crop] += yd
                elif yd < 0:
                    yield_negative[crop] += -yd

    return {
        "surface_transition_counts": dict(surface),
        "plant_established_by_crop": dict(plant_established),
        "plant_to_weed_by_crop": dict(plant_to_weed),
        "watered_false_to_true_by_crop": dict(watered_false_to_true),
        "watered_true_to_false_by_crop": dict(watered_true_to_false),
        "positive_yield_delta_by_crop": dict(yield_positive),
        "negative_yield_delta_by_crop": dict(yield_negative),
    }


def event_delta(pre, post):
    pre_raw = pre.raw()
    post_raw = post.raw()
    p = pre_raw["player"]

    pre_snap = world_snapshot(pre)
    post_snap = world_snapshot(post)

    return {
        "cash_delta": post_snap["cash"] - pre_snap["cash"],
        "seed_delta": map_delta(pre_snap["seeds"], post_snap["seeds"]),
        "carried_delta": map_delta(pre_snap["carried"], post_snap["carried"]),
        "shed_delta": map_delta(pre_snap["shed"], post_snap["shed"]),
        **tile_deltas(pre_raw, post_raw),
    }


def add_counter(dst, mapping):
    for k, v in (mapping or {}).items():
        dst[str(k)] += float(v)


def cumulative_rows(rows):
    cash_in = 0.0
    cash_out = 0.0
    positive_cash_events = 0
    established = Counter()
    weeded = Counter()
    watered = Counter()
    positive_yield = Counter()
    carried_in = Counter()
    shed_in = Counter()

    out = []
    for row in rows:
        d = row["delta"]
        cd = float(d["cash_delta"])
        if cd > 0:
            cash_in += cd
            positive_cash_events += 1
        elif cd < 0:
            cash_out += -cd

        add_counter(established, d["plant_established_by_crop"])
        add_counter(weeded, d["plant_to_weed_by_crop"])
        add_counter(watered, d["watered_false_to_true_by_crop"])
        add_counter(positive_yield, d["positive_yield_delta_by_crop"])

        for k, v in d["carried_delta"].items():
            if v > 0:
                carried_in[k] += float(v)
        for k, v in d["shed_delta"].items():
            if v > 0:
                shed_in[k] += float(v)

        out.append({
            "turn": row["turn"],
            "after_day": row["post"]["day"],
            "after_hour": row["post"]["hour"],
            "cash_return_total": cash_in,
            "cash_outflow_total": cash_out,
            "positive_cash_events": positive_cash_events,
            "plant_established_total": float(sum(established.values())),
            "plant_established_by_crop": dict(established),
            "plant_to_weed_total": float(sum(weeded.values())),
            "plant_to_weed_by_crop": dict(weeded),
            "watered_false_to_true_total": float(sum(watered.values())),
            "watered_false_to_true_by_crop": dict(watered),
            "positive_yield_delta_total": float(sum(positive_yield.values())),
            "positive_yield_delta_by_crop": dict(positive_yield),
            "carried_positive_delta_total": float(sum(carried_in.values())),
            "carried_positive_delta_by_product": dict(carried_in),
            "shed_positive_delta_total": float(sum(shed_in.values())),
            "shed_positive_delta_by_product": dict(shed_in),
        })

    return out


def first_milestones(rows):
    names = {
        "first_plant_established": None,
        "first_plant_to_weed": None,
        "first_watered_false_to_true": None,
        "first_positive_yield_delta": None,
        "first_carried_positive_delta": None,
        "first_shed_positive_delta": None,
        "first_positive_cash_delta": None,
    }

    for row in rows:
        d = row["delta"]

        tests = {
            "first_plant_established": any(
                float(v) > 0 for v in d["plant_established_by_crop"].values()
            ),
            "first_plant_to_weed": any(
                float(v) > 0 for v in d["plant_to_weed_by_crop"].values()
            ),
            "first_watered_false_to_true": any(
                float(v) > 0 for v in d["watered_false_to_true_by_crop"].values()
            ),
            "first_positive_yield_delta": any(
                float(v) > 0 for v in d["positive_yield_delta_by_crop"].values()
            ),
            "first_carried_positive_delta": any(
                float(v) > 0 for v in d["carried_delta"].values()
            ),
            "first_shed_positive_delta": any(
                float(v) > 0 for v in d["shed_delta"].values()
            ),
            "first_positive_cash_delta": float(d["cash_delta"]) > 0,
        }

        for name, hit in tests.items():
            if hit and names[name] is None:
                names[name] = {
                    "turn": row["turn"],
                    "pre_day_hour": [row["pre"]["day"], row["pre"]["hour"]],
                    "post_day_hour": [row["post"]["day"], row["post"]["hour"]],
                    "delta": d,
                    "post_state": row["post"],
                }

    return names


def daily_end(rows):
    by_day = {}
    for row in rows:
        by_day[int(row["pre"]["day"])] = {
            "turn": row["turn"],
            "post": row["post"],
        }
    return [by_day[d] for d in sorted(by_day)]


def replay(run):
    env = make("kaggriculture", configuration={"seed": ab.ENV_SEED}, debug=False)
    env.reset(num_agents=2)

    rows = []
    all_hashes_match = True

    for i, src in enumerate(run["turns"]):
        pre = bind_official_state(
            env._Environment__get_shared_state(0)["observation"]
        )
        if pre.canonical_hash != src["pre_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} pre hash mismatch at turn {i}")

        pre_snap = world_snapshot(pre)
        self_bundle = plain(src["action_bundle"])
        opp_bundle = plain(src["opponent_action_bundle"])

        env.step([self_bundle, opp_bundle])

        post = bind_official_state(env.state[0].observation)
        if post.canonical_hash != src["post_hash"]:
            all_hashes_match = False
            raise RuntimeError(f"{run['mode']} post hash mismatch at turn {i}")

        rows.append({
            "turn": i,
            "pre": pre_snap,
            "delta": event_delta(pre, post),
            "post": world_snapshot(post),
        })

    if not env.done:
        raise RuntimeError(f"{run['mode']} replay did not reach terminal")

    final = bind_official_state(env.state[0].observation)
    terminal_equal = (
        body.state_summary(final)["cash"]
        == run["summary"]["terminal_self_cash"]
    )

    cumulative = cumulative_rows(rows)
    milestones = first_milestones(rows)
    first_cash = milestones["first_positive_cash_delta"]
    if first_cash is None:
        raise RuntimeError(f"{run['mode']} has no positive Cash transition")

    return {
        "rows": rows,
        "cumulative": cumulative,
        "milestones": milestones,
        "first_cash_return_turn": int(first_cash["turn"]),
        "daily_end": daily_end(rows),
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


def state_after_turn(obs, turn):
    return obs["rows"][turn]["post"]


def cumulative_after_turn(obs, turn):
    return obs["cumulative"][turn]


def window_until_first_return(obs):
    t = obs["first_cash_return_turn"]
    return {
        "first_cash_return_turn": t,
        "initial_state": obs["rows"][0]["pre"],
        "milestones_through_first_return": {
            k: v
            for k, v in obs["milestones"].items()
            if v is not None and int(v["turn"]) <= t
        },
        "cumulative_through_first_return": cumulative_after_turn(obs, t),
        "state_after_first_return": state_after_turn(obs, t),
        "daily_end_through_first_return": [
            r for r in obs["daily_end"] if int(r["turn"]) <= t
        ],
        "turn_rows_through_first_return": obs["rows"][: t + 1],
    }


def same_clock_pair(b, p, turn, label):
    return {
        "label": label,
        "after_turn": turn,
        "baseline": {
            "state": state_after_turn(b, turn),
            "cumulative": cumulative_after_turn(b, turn),
        },
        "kind_uniform": {
            "state": state_after_turn(p, turn),
            "cumulative": cumulative_after_turn(p, turn),
        },
    }


def compact_window(obs):
    t = obs["first_cash_return_turn"]
    c = cumulative_after_turn(obs, t)
    s = state_after_turn(obs, t)
    m = obs["milestones"]
    return {
        "first_cash_return_turn": t,
        "first_cash_return_post_day_hour": [s["day"], s["hour"]],
        "first_milestone_turns": {
            k: (v["turn"] if v is not None else None)
            for k, v in m.items()
        },
        "cumulative_through_first_return": c,
        "state_after_first_return": s,
    }


def main():
    baseline_run = body.run_policy("exclude_blocked")
    kind_run = ab.run_kind_uniform()

    b = replay(baseline_run)
    p = replay(kind_run)

    if not b["guard"]["passed"]:
        raise RuntimeError("baseline replay guard failed")
    if not p["guard"]["passed"]:
        raise RuntimeError("kind-uniform replay guard failed")

    b_first = b["first_cash_return_turn"]
    p_first = p["first_cash_return_turn"]

    anchors = [
        same_clock_pair(
            b, p, p_first,
            "after_kind_uniform_first_positive_cash_transition",
        ),
        same_clock_pair(
            b, p, b_first,
            "after_baseline_first_positive_cash_transition",
        ),
    ]

    result = {
        "schema": "first-cash-return-window-v0",
        "question": (
            "Before the first positive Cash transition appears, what did each "
            "whole farm actually become in the Official World?"
        ),
        "source": {
            "baseline": "Confirmed Circulation Body v0 / exclude_blocked",
            "kind_uniform": "Selection Kind-Uniform A/B v0 / kind_uniform",
            "environment_seed": ab.ENV_SEED,
            "policy_seed": ab.POLICY_SEED,
        },
        "baseline": {
            "window": window_until_first_return(b),
            "guard": b["guard"],
        },
        "kind_uniform": {
            "window": window_until_first_return(p),
            "guard": p["guard"],
        },
        "same_clock_anchors": anchors,
        "boundary": {
            "plan_kind_not_used_as_explanatory_coordinate": True,
            "world_delta_not_treated_as_causal_attribution": True,
            "cross_run_difference_not_treated_as_paired_causal_effect": True,
            "inventory_quantity_not_treated_as_identity_tracking": True,
            "new_policy_added": False,
            "generator_changed": False,
            "continuation_changed": False,
            "projector_changed": False,
            "completion_changed": False,
        },
    }

    Path("first_cash_return_window_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps({
        "baseline": compact_window(b),
        "kind_uniform": compact_window(p),
        "same_clock_anchors": anchors,
        "guards": {
            "baseline": b["guard"],
            "kind_uniform": p["guard"],
        },
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
