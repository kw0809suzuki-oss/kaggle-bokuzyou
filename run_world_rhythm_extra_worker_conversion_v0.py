#!/usr/bin/env python3
"""World Rhythm — Extra Worker Conversion Probe v0.

Known evidence:
  Phase Sign Probe v0:
    step168 incremental HIRE +1 yielded terminal-positive in 6/20 worlds,
    terminal-negative in 14/20 worlds.

Question:
  Is that sign split explained by whether the added 9th worker is converted
  into useful Replay work during the remainder of the same day?

Counterfactual design:
  BASE      = Adaptive Replay Runtime v0.1.
  EARLY     = BASE + one extra HIRE appended at step168.
  PASS9     = same as EARLY, but for steps169..191 only, force the 9th hand
              action to PASS while preserving the extra hire, its cost, the
              9-hand state, and the first 8 Replay hand actions.

Thus:
  worker_action_value = terminal(EARLY) - terminal(PASS9)

This probe does not assign semantic Phase labels and does not modify the
frozen Runtime.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "adaptive_replay_contract_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    94002001, 94002002, 94002003, 94002004, 94002005,
    94002006, 94002007, 94002008, 94002009, 94002010,
]
HIRE_STEP = 168
WORK_START = 169
WORK_END = 191

PRODUCTIVE_OPS = {
    "PLANT", "WATER", "HARVEST", "CARE", "FEED",
    "PICKUP", "DROP", "COLLECT_FERTILIZER", "PLACE",
}


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


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def op_name(action):
    if isinstance(action, list) and action:
        return str(action[0])
    return None


def run_one(seed: int, seat: int, mode: str):
    model = load(BASE, f"worker_conv_model_{mode}_{seed}_{seat}_{os.getpid()}")
    opp = load(OPPONENT, f"worker_conv_opp_{mode}_{seed}_{seat}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    marginal_trace = []
    hire_event = None

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        subject_obs = obs0 if seat == 0 else obs1
        opponent_obs = obs1 if seat == 0 else obs0
        step = int(subject_obs.get("step", 0) or 0)

        subject_action = plain(model.agent(subject_obs))
        opponent_action = plain(opp.agent(opponent_obs))

        if mode in ("early", "pass9") and step == HIRE_STEP:
            player = int(subject_obs.get("player", seat) or seat)
            farm = subject_obs["farms"][player]
            original_market = deepcopy(subject_action.get("market", []) or [])
            original_hires = sum(
                1 for o in original_market
                if isinstance(o, list) and o and o[0] == "HIRE"
            )
            subject_action = deepcopy(subject_action)
            subject_action.setdefault("market", []).append(["HIRE"])
            hire_event = {
                "step": step,
                "pre_cash": float(farm.get("money", 0) or 0),
                "pre_hands": len(farm.get("hands", []) or []),
                "pre_hires_today": int(farm.get("hires_today", 0) or 0),
                "original_hire_orders": original_hires,
            }

        if mode in ("early", "pass9") and WORK_START <= step <= WORK_END:
            hands = deepcopy(subject_action.get("hands", []) or [])
            marginal = deepcopy(hands[8]) if len(hands) >= 9 else None
            marginal_trace.append({
                "step": step,
                "requested_action": marginal,
                "operation": op_name(marginal),
                "classified_productive": op_name(marginal) in PRODUCTIVE_OPS,
            })
            if mode == "pass9" and len(hands) >= 9:
                hands[8] = ["PASS"]
                subject_action = deepcopy(subject_action)
                subject_action["hands"] = hands

        if seat == 0:
            env.step([subject_action, opponent_action])
        else:
            env.step([opponent_action, subject_action])

        if hire_event is not None and "post_hands" not in hire_event:
            next_obs = plain(shared(env, seat))
            if int(next_obs.get("step", 0) or 0) == HIRE_STEP + 1:
                post_farm = next_obs["farms"][seat]
                hire_event["post_cash"] = float(post_farm.get("money", 0) or 0)
                hire_event["post_hands"] = len(post_farm.get("hands", []) or [])
                hire_event["post_hires_today"] = int(post_farm.get("hires_today", 0) or 0)

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][seat]["money"]),
        "terminal_opponent": float(final["farms"][1-seat]["money"]),
        "hire_event": hire_event,
        "marginal_trace": marginal_trace,
    }


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def summarize(vals):
    return {
        "n": len(vals),
        "mean": statistics.mean(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def main():
    cases = []
    for seed in SEEDS:
        for seat in (0, 1):
            base = run_one(seed, seat, "base")
            early = run_one(seed, seat, "early")
            pass9 = run_one(seed, seat, "pass9")

            delta_early = early["terminal_self"] - base["terminal_self"]
            worker_action_value = early["terminal_self"] - pass9["terminal_self"]

            trace = early["marginal_trace"]
            productive_count = sum(x["classified_productive"] for x in trace)
            pass_count = sum(x["operation"] == "PASS" for x in trace)
            move_count = sum(x["operation"] in {"NORTH","SOUTH","EAST","WEST"} for x in trace)

            case = {
                "seed": seed,
                "seat": seat,
                "baseline_terminal_self": base["terminal_self"],
                "early_terminal_self": early["terminal_self"],
                "pass9_terminal_self": pass9["terminal_self"],
                "delta_early_vs_base": delta_early,
                "phase_sign_group": "positive" if delta_early > 0 else "negative",
                "worker_action_value": worker_action_value,
                "worker_action_value_sign": sign(worker_action_value),
                "marginal_requested_steps": len(trace),
                "marginal_productive_action_count": productive_count,
                "marginal_move_count": move_count,
                "marginal_pass_count": pass_count,
                "hire_event": early["hire_event"],
                "marginal_trace": trace,
            }
            cases.append(case)
            print("EXTRA_WORKER_CONVERSION_CASE " + json.dumps(case, separators=(",", ":")))

    positive = [c for c in cases if c["delta_early_vs_base"] > 0]
    negative = [c for c in cases if c["delta_early_vs_base"] < 0]

    def group_summary(group):
        vals = [c["worker_action_value"] for c in group]
        return {
            "n": len(group),
            "worker_action_value": summarize(vals),
            "worker_value_positive": sum(v > 0 for v in vals),
            "worker_value_negative": sum(v < 0 for v in vals),
            "worker_value_same": sum(v == 0 for v in vals),
            "mean_productive_action_count": (
                statistics.mean(c["marginal_productive_action_count"] for c in group)
                if group else None
            ),
            "mean_move_count": (
                statistics.mean(c["marginal_move_count"] for c in group)
                if group else None
            ),
            "mean_pass_count": (
                statistics.mean(c["marginal_pass_count"] for c in group)
                if group else None
            ),
        }

    summary = {
        "worlds": len(cases),
        "positive_phase_sign_worlds": len(positive),
        "negative_phase_sign_worlds": len(negative),
        "positive_group": group_summary(positive),
        "negative_group": group_summary(negative),
    }

    out = {
        "schema": "world-rhythm-extra-worker-conversion-v0",
        "question": (
            "Does the step168 sign split track the terminal value generated by "
            "the added 9th worker's own Replay actions during steps169..191?"
        ),
        "design": {
            "hire_step": HIRE_STEP,
            "work_window": [WORK_START, WORK_END],
            "counterfactual": "same extra hire, but 9th hand forced PASS",
        },
        "summary": summary,
        "cases": cases,
        "boundary": {
            "no_phase_semantics": True,
            "requested_action_classification_is_descriptive_only": True,
            "worker_action_value_is_counterfactual_terminal_difference": True,
            "no_new_agent": True,
            "no_new_contract": True,
        },
    }
    Path("world_rhythm_extra_worker_conversion_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("EXTRA_WORKER_CONVERSION_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
