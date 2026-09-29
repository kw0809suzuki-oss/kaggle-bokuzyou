#!/usr/bin/env python3
from types import SimpleNamespace
from run_world_divergence_decision_trace_v0 import (
    first_world_divergence_index,
    summarize_first_divergence,
    first_nonactive_record,
)


def _row(step,cash,fresh_label,central):
    candidate={
        "rank":1,
        "central_terminal_cash":central,
        "strict_terminal_cash":central-100,
        "action":{"farmer":["PASS"],"hands":[],"market":[]},
        "scheduled":[],
        "commitments":[],
        "representative":None,
        "harvest_now":False,
    }
    return {
        "world":{"step":step,"cash":cash},
        "debug":{
            "decision_inputs":{
                "fresh_jobs":[{"key":fresh_label}],
                "investment_jobs":[],
                "active_jobs":[],
                "operating_jobs":[],
            },
            "candidate_scores":[candidate],
            "chosen_candidate_rank":1,
            "chosen":{"central_terminal_cash":central},
        },
        "action":{"farmer":["PASS"],"hands":[],"market":[]},
    }


def main():
    baseline=[
        _row(0,3000,"same",3000),
        _row(1,2900,"baseline",3000),
        _row(2,2800,"same",3000),
    ]
    candidate=[
        _row(0,3000,"same",3000),
        _row(1,2500,"candidate",3200),
        _row(2,2800,"same",3000),
    ]

    assert first_world_divergence_index(baseline,candidate)==1
    assert first_world_divergence_index(baseline,baseline) is None

    summary=summarize_first_divergence(baseline,candidate)
    assert summary["index"]==1
    assert summary["step"]==1
    assert summary["world_diff_paths"]==["cash"]
    assert summary["decision_input_diff_paths"]
    assert summary["candidate_identity_diff_paths"]==[]
    assert summary["candidate_score_diff_paths"]==["[0].central_terminal_cash","[0].strict_terminal_cash"]
    assert summary["chosen_diff_paths"]==["central_terminal_cash"]
    assert summary["action_diff_paths"]==[]

    steps=[
        [SimpleNamespace(status="ACTIVE", observation=SimpleNamespace(step=0, remainingOverageTime=60.0))],
        [SimpleNamespace(status="TIMEOUT", observation=SimpleNamespace(step=202, remainingOverageTime=-0.2))],
        [SimpleNamespace(status="DONE", observation=SimpleNamespace(step=719, remainingOverageTime=-0.2))],
    ]
    nonactive=first_nonactive_record(steps,0)
    assert nonactive=={"step":202,"status":"TIMEOUT","remaining_overage":-0.2}

    print("WORLD_DIVERGENCE_RUNNER_HELPER_OK")


if __name__=="__main__":
    main()
