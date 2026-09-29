#!/usr/bin/env python3
"""Idle Recovery v0 — one-candidate 24-step World gate.

Baseline: ordinary Strong.
Candidate: STRONG_VARIANT=idle_recovery from the validated parallel probe code.
Question: does using only baseline-PASS units for recovery create an observable
World difference within 24 actions on the fixed seed/seat/opponent?
"""
from __future__ import annotations

import json
from pathlib import Path

from strong_model_v0_reimplementation.run_terminal_return_parallel_v0 import (
    run_one, compare_to_baseline,
)

SEED=92804001
SEAT=0
OUT=Path(__file__).with_name("idle_recovery_world_gate_v0_result.json")


def main():
    baseline=run_one("baseline",SEED,SEAT)
    candidate=run_one("idle_recovery",SEED,SEAT)
    comparison=compare_to_baseline(baseline,candidate)
    result={
        "schema":"idle-recovery-world-gate-v0",
        "conditions":{
            "seed":SEED,
            "seat":SEAT,
            "opponent":"Seyamalam pinned v21",
            "official_world_commit":"d7729da06cc1382eb742d6980dc3180aa85caa28",
            "baseline":"Strong Model v0",
            "candidate":"idle_recovery: preserve every busy baseline unit action; activate recovery only on baseline-PASS units",
            "horizon":24,
        },
        "comparison":comparison,
        "baseline":baseline,
        "candidate":candidate,
        "boundary":{
            "world_gate_only":True,
            "terminal_strength_claim":False,
            "cash_after_24_is_diagnostic_only":True,
        },
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "gate_status":comparison["gate_status"],
        "terminal_eligible":comparison["terminal_eligible"],
        "first_action_divergence_turn":comparison["first_action_divergence_turn"],
        "first_world_divergence_turn":comparison["first_world_divergence_turn"],
        "first_worker_divergence_turn":comparison["first_worker_divergence_turn"],
        "first_return_stage_divergence_turn":comparison["first_return_stage_divergence_turn"],
        "cash_delta_after_24":comparison["cash_delta_after_24"],
    },ensure_ascii=False))


if __name__=="__main__":
    main()
