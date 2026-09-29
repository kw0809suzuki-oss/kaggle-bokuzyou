#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "decem_replay_distilled_157026_v0.py"
INDEPENDENT = ROOT / "astra_flow_independent_distilled_v0.py"
SEYAMALAM = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod

def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]

def run_pair(seed, left_path, right_path, left_name, right_name):
    left = load(left_path, f"{left_name}_{seed}_{os.getpid()}")
    right = load(right_path, f"{right_name}_{seed}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    turns = 0
    while not env.done:
        a0 = plain(left.agent(shared_obs(env, 0)))
        a1 = plain(right.agent(shared_obs(env, 1)))
        env.step([a0, a1])
        turns += 1
    final0 = plain(env.state[0].observation)
    cash0 = float(final0["farms"][0]["money"])
    cash1 = float(final0["farms"][1]["money"])
    return {"left": left_name, "right": right_name, "turns": turns, "cash_left": cash0, "cash_right": cash1, "margin_left": cash0-cash1}

def main():
    seed = int(os.environ["SEED"])
    seat = int(os.environ["SELF_SEAT"])
    if seat == 0:
        candidate_vs_sey = run_pair(seed, CANDIDATE, SEYAMALAM, "candidate", "seyamalam")
        independent_vs_sey = run_pair(seed, INDEPENDENT, SEYAMALAM, "independent", "seyamalam")
        candidate_vs_independent = run_pair(seed, CANDIDATE, INDEPENDENT, "candidate", "independent")
        c_self = candidate_vs_sey["cash_left"]
        i_self = independent_vs_sey["cash_left"]
        direct_c = candidate_vs_independent["cash_left"]
        direct_i = candidate_vs_independent["cash_right"]
    else:
        candidate_vs_sey = run_pair(seed, SEYAMALAM, CANDIDATE, "seyamalam", "candidate")
        independent_vs_sey = run_pair(seed, SEYAMALAM, INDEPENDENT, "seyamalam", "independent")
        candidate_vs_independent = run_pair(seed, INDEPENDENT, CANDIDATE, "independent", "candidate")
        c_self = candidate_vs_sey["cash_right"]
        i_self = independent_vs_sey["cash_right"]
        direct_c = candidate_vs_independent["cash_right"]
        direct_i = candidate_vs_independent["cash_left"]
    summary = {
        "schema":"decem-replay-distilled-157026-v0-battle",
        "seed":seed,
        "self_seat":seat,
        "candidate_vs_seyamalam":candidate_vs_sey,
        "independent_vs_seyamalam":independent_vs_sey,
        "candidate_vs_independent":candidate_vs_independent,
        "same_opponent_terminal_delta": c_self - i_self,
        "direct_candidate_minus_independent": direct_c - direct_i,
    }
    out = ROOT / f"decem_replay_distilled_seed{seed}_seat{seat}_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("DECEM_DISTILLED " + json.dumps(summary, separators=(",",":")))

if __name__ == "__main__":
    main()
