#!/usr/bin/env python3
"""Adaptive Circulation Runtime v0 — Fixed10 Operation Effect Lens.

No policy changes.

For the same fixed10 worlds already used by Adaptive Replay v0, compare
the DECEM 157,026 Source transition trace with the current runtime and ask:

1. Is the first non-cash self-effect break the same across worlds?
2. Does the first break signature separate stronger/lower terminal worlds?
3. Where Action differs but realized self-effect stays aligned?
4. Which same-Action operation first yields a structural effect difference?

The lens intentionally ignores cash-only differences as repair triggers.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import statistics
from pathlib import Path

import run_circulation_effect_trace_v0 as lens

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]


def same_action(a, b):
    return a["W"]["action"] == b["W"]["action"]


def self_effect_equal(a, b):
    return all(a[ch] == b[ch] for ch in ("L", "P", "R", "X"))


def noncash_effect_equal(a, b):
    return all(a[ch] == b[ch] for ch in ("P", "R", "X"))


def differing_channels(a, b, include_cash=False):
    channels = ("L", "P", "R", "X") if include_cash else ("P", "R", "X")
    return [ch for ch in channels if a[ch] != b[ch]]


def op_names(action):
    return lens.action_operations(action)


def touched_items(action):
    out = []
    if not isinstance(action, dict):
        return out
    for order in action.get("market", []) or []:
        if isinstance(order, (list, tuple)) and len(order) >= 2:
            op = str(order[0])
            item = str(order[1])
            if op in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL", "SELL"):
                out.append([op, item])
    return out


def context_for_items(effect, action):
    pre = effect["pre_surface"]
    c = pre["C"]
    out = {}
    for op, item in touched_items(action):
        out[item] = {
            "op": op,
            "market_inventory": c["market_inventory"].get(item),
            "market_price": c["market_prices"].get(item),
        }
    return out


def first_action_divergence_effect_aligned(source, current):
    for i, (s, c) in enumerate(zip(source, current)):
        if same_action(s, c):
            continue
        if self_effect_equal(s, c):
            return {
                "transition_index": i,
                "source_action": s["W"]["action"],
                "current_action": c["W"]["action"],
                "source_effect": {ch: s[ch] for ch in ("L", "P", "R", "X")},
                "current_effect": {ch: c[ch] for ch in ("L", "P", "R", "X")},
            }
    return None


def first_same_action_noncash_break(source, current):
    for i, (s, c) in enumerate(zip(source, current)):
        if not same_action(s, c):
            continue
        diff = differing_channels(s, c, include_cash=False)
        if not diff:
            continue
        action = s["W"]["action"]
        return {
            "transition_index": i,
            "roles": lens.flow_roles(action),
            "ops": op_names(action),
            "action": action,
            "differing_channels": diff,
            "source_effect": {ch: s[ch] for ch in diff},
            "current_effect": {ch: c[ch] for ch in diff},
            "source_pre_cash": s["pre_surface"]["L"]["cash"],
            "current_pre_cash": c["pre_surface"]["L"]["cash"],
            "source_context": context_for_items(s, action),
            "current_context": context_for_items(c, action),
        }
    return None


def first_same_action_cash_only_break(source, current):
    for i, (s, c) in enumerate(zip(source, current)):
        if not same_action(s, c):
            continue
        if s["L"] != c["L"] and noncash_effect_equal(s, c):
            action = s["W"]["action"]
            return {
                "transition_index": i,
                "roles": lens.flow_roles(action),
                "ops": op_names(action),
                "action": action,
                "source_cash_effect": s["L"],
                "current_cash_effect": c["L"],
                "source_context": context_for_items(s, action),
                "current_context": context_for_items(c, action),
            }
    return None


def first_channel(source, current, ch):
    return lens.first_divergence(source, current, ch)


def signature(row):
    if row is None:
        return "NONE"
    ops = ",".join(row.get("ops", []))
    channels = ",".join(row.get("differing_channels", []))
    return f"step{row['transition_index']}|ops={ops}|channels={channels}"


def terminal_band(rows):
    vals = sorted(r["terminal_self"] for r in rows)
    med = statistics.median(vals)
    for r in rows:
        r["terminal_band"] = "HIGH" if r["terminal_self"] >= med else "LOW"
    return med


def summarize_signature_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["first_noncash_break_signature"]].append(r)
    out = []
    for sig, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        terminals = [m["terminal_self"] for m in members]
        out.append({
            "signature": sig,
            "n": len(members),
            "seeds": [m["seed"] for m in members],
            "terminal_mean": statistics.mean(terminals),
            "terminal_min": min(terminals),
            "terminal_max": max(terminals),
            "high_band_count": sum(m["terminal_band"] == "HIGH" for m in members),
            "low_band_count": sum(m["terminal_band"] == "LOW" for m in members),
        })
    return out


def main():
    replay = lens.load_source_replay()
    source = lens.source_trace(replay)

    rows = []
    for seed in SEEDS:
        lens.CURRENT_SEED = seed
        current, terminal = lens.current_trace()

        aligned_action_div = first_action_divergence_effect_aligned(source, current)
        noncash_break = first_same_action_noncash_break(source, current)
        cash_only = first_same_action_cash_only_break(source, current)

        first_p = first_channel(source, current, "P")
        first_r = first_channel(source, current, "R")
        first_x = first_channel(source, current, "X")

        row = {
            "seed": seed,
            "terminal_self": terminal,
            "first_action_divergence_effect_aligned": aligned_action_div,
            "first_same_action_cash_only_break": cash_only,
            "first_same_action_noncash_break": noncash_break,
            "first_noncash_break_signature": signature(noncash_break),
            "first_P": None if first_p is None else first_p["transition_index"],
            "first_R": None if first_r is None else first_r["transition_index"],
            "first_X": None if first_x is None else first_x["transition_index"],
        }
        rows.append(row)
        print("FIXED10_CASE " + json.dumps({
            "seed": seed,
            "terminal_self": terminal,
            "effect_aligned_action_div_step": None if aligned_action_div is None else aligned_action_div["transition_index"],
            "first_noncash_break_signature": row["first_noncash_break_signature"],
            "first_P": row["first_P"],
            "first_R": row["first_R"],
            "first_X": row["first_X"],
        }, separators=(",", ":")))

    median_terminal = terminal_band(rows)
    groups = summarize_signature_groups(rows)

    aligned_steps = Counter(
        None if r["first_action_divergence_effect_aligned"] is None
        else r["first_action_divergence_effect_aligned"]["transition_index"]
        for r in rows
    )
    signature_counts = Counter(r["first_noncash_break_signature"] for r in rows)

    out = {
        "schema": "adaptive-circulation-fixed10-operation-effect-lens-v0",
        "meaning": "Observation only; no new operation contract or policy repair.",
        "source_episode": lens.EPISODE_ID,
        "source_terminal_self": float(lens.obs_from(replay["steps"][-1], 0)["farms"][0]["money"]),
        "seeds": SEEDS,
        "terminal_median": median_terminal,
        "summary": {
            "n": len(rows),
            "terminal_mean": statistics.mean(r["terminal_self"] for r in rows),
            "terminal_min": min(r["terminal_self"] for r in rows),
            "terminal_max": max(r["terminal_self"] for r in rows),
            "effect_aligned_action_divergence_steps": {
                str(k): v for k, v in sorted(aligned_steps.items(), key=lambda kv: str(kv[0]))
            },
            "first_noncash_break_signature_counts": dict(signature_counts),
            "first_noncash_break_group_count": len(signature_counts),
        },
        "signature_groups": groups,
        "cases": rows,
        "decision_rule": {
            "promote_to_contract_candidate_only_if": [
                "a break is reproducible across multiple worlds or separates materially different terminal outcomes",
                "the operation-level realized effect can be defined without inventing hidden state",
                "a minimal repair can be tested by terminal A/B without changing unrelated operations",
            ],
            "do_not_promote": [
                "cash-only difference",
                "a common break that does not distinguish terminal behavior",
                "resource-stage difference whose operation provenance is ambiguous",
            ],
        },
    }

    Path("adaptive_circulation_fixed10_operation_effect_lens_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SUMMARY " + json.dumps(out["summary"], separators=(",", ":")))
    print("GROUPS " + json.dumps(groups, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
