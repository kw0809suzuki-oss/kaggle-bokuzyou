#!/usr/bin/env python3
from kaggle_environments import make

from strong_model_v0_reimplementation.agent import agent, debug_state, reset_agent


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): plain(x) for k, x in v.items()}
    return v


def main():
    reset_agent()
    env = make("kaggriculture", configuration={"seed": 92804001}, debug=False)
    env.reset(num_agents=2)
    obs = env._Environment__get_shared_state(0)["observation"]

    action = agent(obs, env.configuration)
    dbg = debug_state(0)

    assert isinstance(action, dict)
    assert dbg is not None and dbg.get("last_choice") is not None
    trace = dbg["last_choice"]

    assert "candidate_scores" in trace, trace.keys()
    assert isinstance(trace["candidate_scores"], list) and trace["candidate_scores"], trace["candidate_scores"]

    row = trace["candidate_scores"][0]
    for key in ("rank", "central_terminal_cash", "strict_terminal_cash", "action", "scheduled", "commitments"):
        assert key in row, (key, row)

    assert trace["chosen_candidate_rank"] == 1, trace
    print("DECISION_TRACE_DEBUG_OK")


if __name__ == "__main__":
    main()
