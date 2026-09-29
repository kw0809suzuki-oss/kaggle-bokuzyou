#!/usr/bin/env python3
from run_world_divergence_decision_trace_v0 import first_world_divergence_index


def main():
    baseline=[
        {"world":{"step":0,"cash":3000}},
        {"world":{"step":1,"cash":2900}},
        {"world":{"step":2,"cash":2800}},
    ]
    candidate=[
        {"world":{"step":0,"cash":3000}},
        {"world":{"step":1,"cash":2500}},
        {"world":{"step":2,"cash":2800}},
    ]
    assert first_world_divergence_index(baseline,candidate)==1
    assert first_world_divergence_index(baseline,baseline) is None
    print("WORLD_DIVERGENCE_RUNNER_HELPER_OK")


if __name__=="__main__":
    main()
