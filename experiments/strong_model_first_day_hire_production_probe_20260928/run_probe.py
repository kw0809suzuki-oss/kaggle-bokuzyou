#!/usr/bin/env python3
"""One-state probe: current production plan vs. the same plan plus HIRE.

The input is a real, saved Kaggriculture replay state at step 24 (day 1, hour 0).
The replay's action is stored on the following row, so the script asserts that
the trace action matches that next replay row before using the state.
"""
from __future__ import annotations

import copy
import gzip
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strong_model_v0_reimplementation.planner import (
    Settings,
    _project_one_turn,
    choose,
    rollout,
)

REPLAY = ROOT / "battle-results/repair-probe/repaired-92804001-seat0.replay.json.gz"
TRACE = ROOT / "battle-results/repair-probe/repaired-92804001-seat0.trace.json"
OUT = Path(__file__).with_name("result.json")
TARGET_STEP = 24


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    return value


def first_cash_increase(records: list[dict[str, Any]], start_cash: float):
    previous = records[0] if records else None
    for row in records[1:]:
        if float(row["cash"]) > start_cash:
            return {
                "first_observed_step": int(row["step"]),
                "cash": float(row["cash"]),
                "from_action_at_step": None if previous is None else previous["action"],
            }
        previous = row
    return None


def run() -> dict[str, Any]:
    with gzip.open(REPLAY, "rt", encoding="utf-8") as handle:
        replay = json.load(handle)
    trace = json.loads(TRACE.read_text(encoding="utf-8"))

    match_i = next(
        i for i, row in enumerate(replay["steps"])
        if int(row[0]["observation"]["step"]) == TARGET_STEP
        and int(row[0]["observation"]["player"]) == 0
    )
    obs = replay["steps"][match_i][0]["observation"]
    next_row = replay["steps"][match_i + 1][0]
    trace_row = next(row for row in trace if int(row["step"]) == TARGET_STEP)

    raw_cash = float(obs["farms"][0]["money"])
    trace_cash = float(trace_row["cash"])
    assert raw_cash == trace_cash == 463.0, (raw_cash, trace_cash)
    assert int(obs["day"]) == 1 and int(obs["hour"]) == 0
    assert int(next_row["observation"]["step"]) == TARGET_STEP + 1
    assert normalized(next_row["action"]) == normalized(trace_row["action"]), (
        "The action is recorded on the next replay row; trace and replay differ.",
        next_row["action"],
        trace_row["action"],
    )

    decision = trace_row["debug"]["last_choice"]
    after = decision["active_after"]
    before_keys = set(decision["active_before"])
    active_specs = {key: spec for key, spec in after.items() if key in before_keys}

    cfg = Settings()
    chosen, representative, active_jobs, continuation = choose(obs, cfg, active_specs)
    assert normalized(chosen.action) == normalized(trace_row["action"]), (
        "Current source did not reproduce the saved decision.",
        chosen.action,
        trace_row["action"],
    )

    chosen_jobs = list(active_jobs)
    if representative is not None:
        chosen_jobs.append(representative)
    assert {job.key for job in chosen_jobs} == set(decision["active_after"]), (
        "The selected jobs do not reconstruct the recorded active set.",
        [job.key for job in chosen_jobs],
        list(decision["active_after"]),
    )

    # Add one hire to the actual current production-start choice. This isolates
    # whether the omitted joint candidate changes forecast at this day boundary.
    variant = copy.deepcopy(chosen.action)
    assert ["HIRE"] not in variant["market"]
    assert len(variant["market"]) < cfg.maxMarketOrdersPerTurn
    variant["market"].append(["HIRE"])

    baseline_world = _project_one_turn(obs, chosen.action, cfg)
    hire_world = _project_one_turn(obs, variant, cfg)
    baseline_end, baseline_trace = rollout(
        obs, cfg, chosen_jobs, first_action=chosen.action, trace=True
    )
    hire_end, hire_trace = rollout(
        obs, cfg, chosen_jobs, first_action=variant, trace=True
    )

    player = int(obs["player"])
    baseline_farm = baseline_world["farms"][player]
    hire_farm = hire_world["farms"][player]
    chosen_spec = None if representative is None else representative.spec()
    result = {
        "probe": "day-boundary-current-production-plan-plus-hire",
        "source": {
            "replay": str(REPLAY.relative_to(ROOT)),
            "trace": str(TRACE.relative_to(ROOT)),
            "seed": 92804001,
            "seat": 0,
            "step": TARGET_STEP,
            "day": int(obs["day"]),
            "hour": int(obs["hour"]),
            "cash": raw_cash,
            "replay_trace_action_alignment": "matched on next replay row",
        },
        "baseline": {
            "action": normalized(chosen.action),
            "representative": chosen_spec,
            "scheduled": list(chosen.scheduled),
            "active_before": [job.key for job in active_jobs],
            "continuation_cash": float(continuation.envelope.central_cash),
            "terminal_cash": float(baseline_end),
            "one_turn_cash": float(baseline_farm["money"]),
            "one_turn_hands": len(baseline_farm.get("hands", [])),
            "first_cash_increase": first_cash_increase(baseline_trace, raw_cash),
        },
        "with_hire": {
            "action": normalized(variant),
            "same_production_commitments": chosen_spec,
            "terminal_cash": float(hire_end),
            "one_turn_cash": float(hire_farm["money"]),
            "one_turn_hands": len(hire_farm.get("hands", [])),
            "hires_today": int(hire_farm.get("hires_today", 0)),
            "first_cash_increase": first_cash_increase(hire_trace, raw_cash),
        },
        "differences": {
            "terminal_cash": float(hire_end - baseline_end),
            "one_turn_cash": float(hire_farm["money"] - baseline_farm["money"]),
            "one_turn_hands": len(hire_farm.get("hands", []))
                - len(baseline_farm.get("hands", [])),
        },
        "boundary": (
            "One saved state and the model's constrained rollout. This does not "
            "measure battle cash, margin, or win rate."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
