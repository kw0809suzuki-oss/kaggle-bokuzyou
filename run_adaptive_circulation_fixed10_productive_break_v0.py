#!/usr/bin/env python3
"""Fixed10 Productive-Body Break Probe.

Observation only.
Focus: why first P divergence is step120 in 6 worlds but step479 in 4.

No repair is implemented.
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


def compact_pre(effect):
    p = effect["pre_surface"]
    return {
        "cash": p["L"]["cash"],
        "hands": p["P"]["hands"],
        "plants": p["P"]["plants"],
        "animals": p["P"]["animals"],
        "seeds": p["X"]["seeds"],
        "market_prices": p["C"]["market_prices"],
    }


def main():
    replay = lens.load_source_replay()
    source = lens.source_trace(replay)
    rows = []

    for seed in SEEDS:
        lens.CURRENT_SEED = seed
        current, terminal = lens.current_trace()
        p = lens.first_divergence(source, current, "P")
        if p is None:
            row = {"seed": seed, "terminal_self": terminal, "first_P": None}
        else:
            i = p["transition_index"]
            s = source[i]
            c = current[i]
            row = {
                "seed": seed,
                "terminal_self": terminal,
                "first_P": i,
                "source_action": s["W"]["action"],
                "current_action": c["W"]["action"],
                "source_roles": lens.flow_roles(s["W"]["action"]),
                "current_roles": lens.flow_roles(c["W"]["action"]),
                "source_P_effect": s["P"],
                "current_P_effect": c["P"],
                "source_L_effect": s["L"],
                "current_L_effect": c["L"],
                "source_pre": compact_pre(s),
                "current_pre": compact_pre(c),
                "same_action": s["W"]["action"] == c["W"]["action"],
            }
        rows.append(row)
        print("P_BREAK_CASE " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))

    groups = defaultdict(list)
    for r in rows:
        groups[r["first_P"]].append(r)

    group_summary = []
    for step, members in sorted(groups.items(), key=lambda kv: (999999 if kv[0] is None else kv[0])):
        vals = [m["terminal_self"] for m in members]
        actions = Counter(json.dumps(m.get("current_action"), sort_keys=True, ensure_ascii=False) for m in members)
        effects = Counter(json.dumps(m.get("current_P_effect"), sort_keys=True, ensure_ascii=False) for m in members)
        group_summary.append({
            "first_P": step,
            "n": len(members),
            "seeds": [m["seed"] for m in members],
            "terminal_mean": statistics.mean(vals),
            "terminal_median": statistics.median(vals),
            "terminal_min": min(vals),
            "terminal_max": max(vals),
            "same_action_count": sum(bool(m.get("same_action")) for m in members),
            "current_action_variants": dict(actions),
            "current_P_effect_variants": dict(effects),
        })

    out = {
        "schema": "adaptive-circulation-fixed10-productive-break-v0",
        "meaning": "Observation only; no policy change.",
        "groups": group_summary,
        "cases": rows,
        "boundary": [
            "P-break timing is not assumed causal for terminal.",
            "Overlapping terminal ranges do not support a separator claim.",
            "No contract is promoted from timing alone.",
        ],
    }
    Path("adaptive_circulation_fixed10_productive_break_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(group_summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
