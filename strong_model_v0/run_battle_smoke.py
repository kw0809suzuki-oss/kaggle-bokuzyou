#!/usr/bin/env python3
"""Strong Model v0 candidate battle smoke.

Purpose:
- Run Strong Model v0 and Independent Distilled v0 under the same fresh seeds,
  opponent, seat placements, environment, and aggregation.
- This is a smoke/candidate test, NOT a Strength Benchmark verdict.

Output: strong_model_v0/battle_smoke_result.json
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strong_model_v0.agent import agent as strong_agent, reset_agent as reset_strong
import astra_flow_independent_distilled_v0 as independent

SEEDS = [92804001, 92804002, 92804003]
OPPONENT_PATH = ROOT / "opponents" / "seyamalam_v21.py"
OUT = Path(__file__).with_name("battle_smoke_result.json")


def _load_opponent():
    spec = importlib.util.spec_from_file_location("strong_model_v0_opponent", OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {OPPONENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset(obj: Any) -> None:
    fn = getattr(obj, "reset_agent", None)
    if callable(fn):
        fn()


def _money(final_state: list[Any], seat: int) -> float:
    obs = final_state[seat].observation
    farms = obs["farms"] if isinstance(obs, dict) else obs.farms
    farm = farms[seat]
    if isinstance(farm, dict):
        return float(farm["money"])
    return float(farm.money)


def _run_one(
    label: str,
    self_agent: Callable[..., dict[str, Any]],
    self_reset: Callable[[], None],
    seed: int,
    seat: int,
) -> dict[str, Any]:
    opponent = _load_opponent()
    self_reset()
    _reset(opponent)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    agents = [None, None]
    agents[seat] = self_agent
    agents[1 - seat] = opponent.agent
    env.run(agents)

    terminal_self = _money(env.state, seat)
    terminal_opp = _money(env.state, 1 - seat)
    margin = terminal_self - terminal_opp
    status = str(env.state[seat].status)

    return {
        "model": label,
        "seed": seed,
        "seat": seat,
        "terminal_self": terminal_self,
        "terminal_opponent": terminal_opp,
        "margin": margin,
        "win": terminal_self > terminal_opp,
        "status": status,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selfs = [r["terminal_self"] for r in rows]
    margins = [r["margin"] for r in rows]
    return {
        "n": len(rows),
        "mean_self": statistics.mean(selfs),
        "median_self": statistics.median(selfs),
        "min_self": min(selfs),
        "max_self": max(selfs),
        "mean_margin": statistics.mean(margins),
        "wins": sum(bool(r["win"]) for r in rows),
        "statuses": sorted({r["status"] for r in rows}),
    }


def main() -> None:
    rows: list[dict[str, Any]] = []

    models = [
        ("independent_distilled_v0", independent.agent, independent.reset_agent),
        ("strong_model_v0", strong_agent, reset_strong),
    ]

    for seed in SEEDS:
        for seat in (0, 1):
            for label, agent_fn, reset_fn in models:
                row = _run_one(label, agent_fn, reset_fn, seed, seat)
                rows.append(row)
                print("BATTLE " + json.dumps(row, separators=(",", ":"), sort_keys=True))

    by_model = {}
    for label, _, _ in models:
        by_model[label] = _aggregate([r for r in rows if r["model"] == label])

    paired = []
    for seed in SEEDS:
        for seat in (0, 1):
            b = next(r for r in rows if r["model"] == "independent_distilled_v0" and r["seed"] == seed and r["seat"] == seat)
            s = next(r for r in rows if r["model"] == "strong_model_v0" and r["seed"] == seed and r["seat"] == seat)
            paired.append({
                "seed": seed,
                "seat": seat,
                "independent_self": b["terminal_self"],
                "strong_self": s["terminal_self"],
                "self_diff_strong_minus_independent": s["terminal_self"] - b["terminal_self"],
                "independent_margin": b["margin"],
                "strong_margin": s["margin"],
                "margin_diff_strong_minus_independent": s["margin"] - b["margin"],
            })

    diffs = [p["self_diff_strong_minus_independent"] for p in paired]
    margin_diffs = [p["margin_diff_strong_minus_independent"] for p in paired]

    result = {
        "schema": "strong-model-v0-battle-smoke-v0",
        "purpose": "Candidate smoke only; not a Strength Benchmark or adoption verdict.",
        "conditions": {
            "seeds": SEEDS,
            "seats": [0, 1],
            "opponent": "Seyamalam pinned v21",
            "comparison": "Independent Distilled v0 vs Strong Model v0 under identical seed/seat/opponent conditions",
        },
        "by_model": by_model,
        "paired_summary": {
            "mean_self_diff_strong_minus_independent": statistics.mean(diffs),
            "median_self_diff_strong_minus_independent": statistics.median(diffs),
            "improved": sum(d > 0 for d in diffs),
            "equal": sum(d == 0 for d in diffs),
            "worsened": sum(d < 0 for d in diffs),
            "mean_margin_diff_strong_minus_independent": statistics.mean(margin_diffs),
        },
        "paired": paired,
        "rows": rows,
        "boundary": {
            "not_strength_benchmark": True,
            "not_adoption_verdict": True,
            "fresh_small_sample": True,
        },
    }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(result["paired_summary"], separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
