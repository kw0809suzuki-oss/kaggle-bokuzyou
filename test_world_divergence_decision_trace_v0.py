#!/usr/bin/env python3
from kaggle_environments import make

from strong_model_v0_reimplementation.agent import agent, debug_state, reset_agent
from strong_model_v0_reimplementation.planner import choose, settings_from


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

    raw = plain(obs)
    cfg = settings_from(env.configuration)
    silent, _, _, _ = choose(raw, cfg, {}, diagnostics=False)
    assert not hasattr(silent, "candidate_scores")
    assert not hasattr(silent, "decision_inputs")

    diagnostic, _, _, _ = choose(raw, cfg, {}, diagnostics=True)
    assert hasattr(diagnostic, "candidate_scores")
    assert hasattr(diagnostic, "decision_inputs")

    action = agent(obs, env.configuration)
    dbg = debug_state(0)

    assert isinstance(action, dict)
    assert dbg is not None and dbg.get("last_choice") is not None
    trace = dbg["last_choice"]

    assert "decision_inputs" in trace, trace.keys()
    inputs = trace["decision_inputs"]
    for key in ("fresh_jobs", "investment_jobs", "active_jobs", "operating_jobs"):
        assert key in inputs, (key, inputs.keys())
        assert isinstance(inputs[key], list), (key, type(inputs[key]))

    assert "candidate_scores" in trace, trace.keys()
    assert isinstance(trace["candidate_scores"], list) and trace["candidate_scores"], trace["candidate_scores"]

    row = trace["candidate_scores"][0]
    for key in ("rank", "central_terminal_cash", "strict_terminal_cash", "action", "scheduled", "commitments"):
        assert key in row, (key, row)

    assert trace["chosen_candidate_rank"] == 1, trace
    print("DECISION_TRACE_DEBUG_OK")


if __name__ == "__main__":
    main()
