#!/usr/bin/env python3
"""Order Residual Refinery v0.

Purpose:
  Refine only the two residual failures left by Order Separator Probe v0.
  Do not redesign the 20-world classifier from scratch.

Frozen carrier:
  market.inventory.EGG <= 9961.5 -> Original
  market.inventory.EGG > 9961.5  -> Swap

Questions:
  1. Are the two residuals structured by seat / engine participation order?
  2. Does a mechanism-nearer Town shop representation explain the residual subset?
  3. Can one very small residual rule repair the frozen 18/20 carrier?

Boundary:
  Exploratory on the same 20 Worlds.
  No policy promotion. No learned model. No terminal-derived feature.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import run_order_separator_probe_v0 as base

EGG_THRESHOLD = 9961.5


def shop_name(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("name", "kind", "shop", "type", "id"):
            if x.get(k) is not None:
                return str(x[k])
        return json.dumps(x, sort_keys=True, separators=(",", ":"))
    return str(x)


def shop_signature(pre_state):
    town = pre_state.get("town", {}) or {}
    shops = [shop_name(x) for x in (town.get("unlocked_shops", []) or [])]
    return tuple(shops)


def thresholds(values):
    u = sorted(set(float(v) for v in values))
    return [(a + b) / 2.0 for a, b in zip(u, u[1:])]


def lit_value(row, spec):
    feature, op, threshold = spec
    v = row["residual_features"][feature]
    return v <= threshold if op == "<=" else v > threshold


def rule_predictions(rows, rule):
    if rule["kind"] == "one":
        raw = [lit_value(r, rule["literal"]) for r in rows]
    else:
        a = [lit_value(r, rule["left"]) for r in rows]
        b = [lit_value(r, rule["right"]) for r in rows]
        if rule["combine"] == "AND":
            raw = [x and y for x, y in zip(a, b)]
        else:
            raw = [x or y for x, y in zip(a, b)]
    return [("swap" if x == rule["swap_when_true"] else "original") for x in raw]


def accuracy(preds, rows):
    return sum(int(p == r["label"]) for p, r in zip(preds, rows))


def search_residual_rules(rows):
    # Restrict to mechanism-nearer features plus Town shop one-hots.
    names = [
        "queue.self_first",
        "self.cash",
        "opp.cash",
        "market.inventory.WHEAT",
        "market.price.WHEAT",
    ]
    names += sorted(
        n for n in rows[0]["residual_features"]
        if n.startswith("town.shop.")
    )

    literals = []
    for name in names:
        vals = [r["residual_features"][name] for r in rows]
        for t in thresholds(vals):
            for op in ("<=", ">"):
                literals.append((name, op, t))

    rules = []
    for lit in literals:
        for swap_when_true in (True, False):
            rule = {
                "kind": "one",
                "literal": lit,
                "swap_when_true": swap_when_true,
            }
            rule["correct"] = accuracy(rule_predictions(rows, rule), rows)
            rules.append(rule)

    for left, right in itertools.combinations(literals, 2):
        if left[0] == right[0]:
            continue
        for combine in ("AND", "OR"):
            for swap_when_true in (True, False):
                rule = {
                    "kind": "two",
                    "left": left,
                    "right": right,
                    "combine": combine,
                    "swap_when_true": swap_when_true,
                }
                rule["correct"] = accuracy(rule_predictions(rows, rule), rows)
                rules.append(rule)

    rules.sort(
        key=lambda r: (
            -r["correct"],
            0 if r["kind"] == "one" else 1,
            json.dumps(r, sort_keys=True),
        )
    )
    return rules


def jsonable_rule(rule):
    out = dict(rule)
    for key in ("literal", "left", "right"):
        if key in out:
            out[key] = {
                "feature": out[key][0],
                "op": out[key][1],
                "threshold": out[key][2],
            }
    return out


def main():
    rows = []
    shop_universe = set()

    for seed in base.SEEDS:
        for seat in (0, 1):
            original, captured = base.run_world(seed, seat, "original", capture=True)
            swap, _ = base.run_world(seed, seat, "swap", capture=False)
            if captured is None:
                raise RuntimeError("missing capture")

            delta = swap - original
            label = "swap" if delta > 0 else "original" if delta < 0 else "tie"
            if label == "tie":
                raise RuntimeError("unexpected tie")

            sig = shop_signature(captured["pre_state"])
            shop_universe.update(sig)

            features = dict(captured["features"])
            carrier_pred = (
                "original"
                if features["market.inventory.EGG"] <= EGG_THRESHOLD
                else "swap"
            )
            rows.append({
                "seed": seed,
                "seat": seat,
                "label": label,
                "delta_swap_minus_original": delta,
                "carrier_pred": carrier_pred,
                "carrier_correct": carrier_pred == label,
                "features": features,
                "shop_signature": list(sig),
            })

    shop_universe = sorted(shop_universe)
    for r in rows:
        rf = {
            "queue.self_first": 1.0 if r["seat"] == 0 else 0.0,
            "self.cash": r["features"]["self.cash"],
            "opp.cash": r["features"]["opp.cash"],
            "market.inventory.WHEAT": r["features"]["market.inventory.WHEAT"],
            "market.price.WHEAT": r["features"]["market.price.WHEAT"],
        }
        sig = set(r["shop_signature"])
        for shop in shop_universe:
            rf[f"town.shop.{shop}"] = 1.0 if shop in sig else 0.0
        r["residual_features"] = rf

    residual = [
        r for r in rows
        if r["features"]["market.inventory.EGG"] <= EGG_THRESHOLD
    ]
    carrier_failures = [r for r in rows if not r["carrier_correct"]]

    # Check same numeric state excluding seat. This exposes whether seat itself
    # is carrying information that the previous "identical snapshot" check hid.
    excl_seat_groups = {}
    for r in rows:
        f = {k: v for k, v in r["features"].items() if k != "seat"}
        key = json.dumps(f, sort_keys=True, separators=(",", ":"))
        excl_seat_groups.setdefault(key, []).append(r)
    conflicting_without_seat = []
    for grp in excl_seat_groups.values():
        labels = sorted(set(r["label"] for r in grp))
        if len(labels) > 1:
            conflicting_without_seat.append([
                {
                    "seed": r["seed"],
                    "seat": r["seat"],
                    "label": r["label"],
                    "delta": r["delta_swap_minus_original"],
                }
                for r in grp
            ])

    same_seed_flips = []
    for seed in base.SEEDS:
        pair = [r for r in rows if r["seed"] == seed]
        if len(set(r["label"] for r in pair)) > 1:
            same_seed_flips.append({
                "seed": seed,
                "rows": [
                    {
                        "seat": r["seat"],
                        "label": r["label"],
                        "delta": r["delta_swap_minus_original"],
                        "shop_signature": r["shop_signature"],
                    }
                    for r in pair
                ],
            })

    rules = search_residual_rules(residual)
    best = rules[0] if rules else None

    # Two-stage candidate: frozen carrier handles high-EGG worlds.
    # The residual rule is used only inside the low-EGG subset.
    final_preds = {}
    if best:
        residual_preds = rule_predictions(residual, best)
        rp = {
            (r["seed"], r["seat"]): p
            for r, p in zip(residual, residual_preds)
        }
        for r in rows:
            if r["features"]["market.inventory.EGG"] > EGG_THRESHOLD:
                final_preds[(r["seed"], r["seat"])] = "swap"
            else:
                final_preds[(r["seed"], r["seat"])] = rp[(r["seed"], r["seat"])]

    two_stage_correct = sum(
        int(final_preds.get((r["seed"], r["seat"])) == r["label"])
        for r in rows
    ) if best else 0

    # Summarize shop-signature behavior without treating it as causal proof.
    shop_groups = {}
    for r in rows:
        key = (r["seat"], tuple(r["shop_signature"]))
        shop_groups.setdefault(key, []).append(r["label"])
    shop_signature_seat_conflicts = sum(
        len(set(labels)) > 1 for labels in shop_groups.values()
    )

    summary = {
        "worlds": len(rows),
        "frozen_carrier": {
            "feature": "market.inventory.EGG",
            "threshold": EGG_THRESHOLD,
            "rule": "EGG<=threshold -> original; else swap",
            "correct": sum(r["carrier_correct"] for r in rows),
        },
        "carrier_failures": [
            {
                "seed": r["seed"],
                "seat": r["seat"],
                "label": r["label"],
                "delta": r["delta_swap_minus_original"],
                "egg": r["features"]["market.inventory.EGG"],
                "cash": r["features"]["self.cash"],
                "wheat_inventory": r["features"]["market.inventory.WHEAT"],
                "wheat_price": r["features"]["market.price.WHEAT"],
                "shop_signature": r["shop_signature"],
            }
            for r in carrier_failures
        ],
        "conflicting_numeric_snapshot_groups_when_seat_excluded":
            len(conflicting_without_seat),
        "same_seed_order_flips": same_seed_flips,
        "shop_signature_plus_seat_conflicts": shop_signature_seat_conflicts,
        "low_egg_residual_worlds": len(residual),
        "best_residual_rule": jsonable_rule(best) if best else None,
        "two_stage_correct": two_stage_correct,
        "two_stage_total": len(rows),
    }

    out = {
        "schema": "order-residual-refinery-v0",
        "question": "Can the two EGG-carrier residuals be explained by a tiny mechanism-nearer seat/World representation?",
        "summary": summary,
        "conflicting_without_seat_groups": conflicting_without_seat,
        "rows": rows,
        "top_residual_rules": [jsonable_rule(r) for r in rules[:20]],
        "boundary": {
            "same_20_worlds": True,
            "frozen_18_of_20_carrier": True,
            "residual_only_refinement": True,
            "shop_identity_added_as_representation": True,
            "exploratory_same_sample": True,
            "no_policy_promotion": True,
            "no_learned_model": True,
            "no_terminal_derived_feature": True,
        },
    }

    Path("order_residual_refinery_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("ORDER_RESIDUAL_REFINERY_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
