#!/usr/bin/env python3
from run_world_divergence_decision_trace_v0 import (
    candidate_identity_rows,
    first_diff_paths,
)


def main():
    assert first_diff_paths({"a":1,"b":{"c":2}}, {"a":1,"b":{"c":3}}) == ["b.c"]
    assert first_diff_paths({"a":[1,2]}, {"a":[1,3]}) == ["a[1]"]

    rows=[
        {
            "rank":1,
            "central_terminal_cash":3000,
            "strict_terminal_cash":2900,
            "action":{"farmer":["PASS"],"hands":[],"market":[]},
            "scheduled":[],
            "commitments":[],
            "representative":None,
            "harvest_now":False,
        }
    ]
    ids=candidate_identity_rows(rows)
    assert len(ids)==1
    assert "central_terminal_cash" not in ids[0]
    assert "strict_terminal_cash" not in ids[0]
    assert "rank" not in ids[0]
    assert ids[0]["action"]["farmer"]==["PASS"]
    print("WORLD_DIVERGENCE_TRACE_HELPERS_OK")


if __name__=="__main__":
    main()
