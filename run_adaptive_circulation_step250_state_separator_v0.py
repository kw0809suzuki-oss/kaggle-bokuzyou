#!/usr/bin/env python3
"""Observe whether step250 agent-visible Current World contains a simple separator
for HARVEST>PASS vs PASS>HARVEST in the fixed10.

This is an observation probe, not a selector-fitting exercise.

Checks:
1. full visible state equality / fingerprints
2. single scalar fields whose observed value sets are disjoint by sign group
3. numeric fields with a one-threshold perfect split on these ten cases

A found separator is sample evidence only. It is not promoted to a decision rule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]

# From the already-observed WATER/PASS/HARVEST fixed10.
HARVEST_MINUS_PASS = {
    92802001: -7657,
    92802002: -104,
    92802003: -9578,
    92802004: -50,
    92802005: -794,
    92802006: 895,
    92802007: 190,
    92802008: -53,
    92802009: 20312,
    92802010: 21568,
}


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k in sorted(obj):
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(obj[k], p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.update(flatten(v, p))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj
    return out


def capture(seed):
    model = load(MODEL, f"state_model_{seed}")
    opp = load(OPPONENT, f"state_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0 = shared(env, 0)
        if int(o0.get("step", -1)) == 250:
            return o0
        o1 = shared(env, 1)
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))
        env.step([a0, a1])

    raise RuntimeError(f"seed {seed}: step250 not reached")


def scalar_key(v):
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return canonical(v)


def disjoint_value_fields(rows):
    paths = sorted(set.intersection(*(set(r["flat"]) for r in rows)))
    pos = [r for r in rows if r["label"] == "HARVEST_GT_PASS"]
    neg = [r for r in rows if r["label"] == "PASS_GT_HARVEST"]
    found = []
    for p in paths:
        pv = {scalar_key(r["flat"][p]) for r in pos}
        nv = {scalar_key(r["flat"][p]) for r in neg}
        if pv.isdisjoint(nv):
            found.append({
                "path": p,
                "positive_values": [r["flat"][p] for r in pos],
                "negative_values": [r["flat"][p] for r in neg],
                "positive_unique": sorted(pv),
                "negative_unique": sorted(nv),
            })
    return found


def threshold_fields(rows):
    paths = sorted(set.intersection(*(set(r["flat"]) for r in rows)))
    found = []
    for p in paths:
        vals = []
        ok = True
        for r in rows:
            v = r["flat"][p]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                ok = False
                break
            vals.append((float(v), r["label"], r["seed"]))
        if not ok:
            continue

        uniq = sorted(set(v for v, _, _ in vals))
        if len(uniq) < 2:
            continue
        cuts = [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
        for cut in cuts:
            for orientation in ("GT_IS_HARVEST", "LE_IS_HARVEST"):
                good = True
                for v, label, _ in vals:
                    pred = (
                        "HARVEST_GT_PASS"
                        if ((v > cut) if orientation == "GT_IS_HARVEST" else (v <= cut))
                        else "PASS_GT_HARVEST"
                    )
                    if pred != label:
                        good = False
                        break
                if good:
                    found.append({
                        "path": p,
                        "threshold": cut,
                        "orientation": orientation,
                        "values": [
                            {"seed": s, "value": v, "label": lab}
                            for v, lab, s in sorted(vals)
                        ],
                    })
    return found


def main():
    rows = []
    for seed in SEEDS:
        state = capture(seed)
        flat = flatten(state)
        delta = HARVEST_MINUS_PASS[seed]
        label = "HARVEST_GT_PASS" if delta > 0 else "PASS_GT_HARVEST"
        rows.append({
            "seed": seed,
            "delta_harvest_minus_pass": delta,
            "label": label,
            "state_sha256": hashlib.sha256(canonical(state).encode("utf-8")).hexdigest(),
            "state": state,
            "flat": flat,
        })
        print("CAPTURE " + json.dumps({
            "seed": seed,
            "label": label,
            "delta_harvest_minus_pass": delta,
            "state_sha256": rows[-1]["state_sha256"],
            "scalar_count": len(flat),
        }, separators=(",", ":")))

    # Any exact visible states shared across opposite labels?
    cross_label_equal = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["label"] != b["label"] and a["state_sha256"] == b["state_sha256"]:
                cross_label_equal.append([a["seed"], b["seed"]])

    disjoint = disjoint_value_fields(rows)
    thresholds = threshold_fields(rows)

    # Keep output readable while retaining counts. Prefer short/simple paths first.
    disjoint_sorted = sorted(disjoint, key=lambda x: (x["path"].count(".") + x["path"].count("["), len(x["path"]), x["path"]))
    thresholds_sorted = sorted(thresholds, key=lambda x: (x["path"].count(".") + x["path"].count("["), len(x["path"]), x["path"]))

    summary = {
        "n": len(rows),
        "harvest_gt_pass_seeds": [r["seed"] for r in rows if r["label"] == "HARVEST_GT_PASS"],
        "pass_gt_harvest_seeds": [r["seed"] for r in rows if r["label"] == "PASS_GT_HARVEST"],
        "unique_full_state_fingerprints": len(set(r["state_sha256"] for r in rows)),
        "cross_label_exact_state_matches": cross_label_equal,
        "scalar_fields_common_to_all": len(set.intersection(*(set(r["flat"]) for r in rows))),
        "disjoint_single_field_count": len(disjoint),
        "perfect_numeric_threshold_count": len(thresholds),
        "disjoint_single_field_examples": disjoint_sorted[:40],
        "perfect_numeric_threshold_examples": thresholds_sorted[:40],
    }

    out = {
        "schema": "adaptive-circulation-step250-state-separator-v0",
        "question": "Can step250 Current World separate HARVEST>PASS from PASS>HARVEST in fixed10?",
        "summary": summary,
        "cases": [
            {
                "seed": r["seed"],
                "label": r["label"],
                "delta_harvest_minus_pass": r["delta_harvest_minus_pass"],
                "state_sha256": r["state_sha256"],
            }
            for r in rows
        ],
        "boundary": [
            "A separator found on these ten cases is observational and may be accidental.",
            "No candidate field is causal merely because it perfectly separates this sample.",
            "No policy or threshold is promoted by this probe.",
            "If no simple separator exists, do not invent a compound classifier in this pass."
        ],
    }

    Path("adaptive_circulation_step250_state_separator_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
