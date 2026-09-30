#!/usr/bin/env python3
"""World Rhythm Rider candidate — Phase Sign Probe v0.

Question:
    Does the terminal contribution of the SAME incremental action, HIRE +1,
    change sign when placed at an early vs late world position?

Important:
    This is not a World Rhythm Rider agent.
    It is only an existence probe for phase-dependent action value.

Design:
    Base = Adaptive Replay Contract Runtime v0.1 behavior.
    Early = Base + append exactly one HIRE at step168.
    Late  = Base + append exactly one HIRE at step672.

Why these steps:
    Frozen Replay already requests exactly 8 HIREs at both step168 and step672.
    The intervention is appended as the 9th HIRE in both positions, making the
    intended marginal HIRE order comparable. We do not name either position a
    semantic phase in this probe.

Worlds:
    fresh 10 seeds x subject seat0/seat1 = 20 paired worlds.
    Same opponent: Seyamalam v21.

Evidence target:
    delta_early = terminal(Early) - terminal(Base)
    delta_late  = terminal(Late)  - terminal(Base)

Primary observation:
    sign(delta_early) vs sign(delta_late), only in worlds where the incremental
    HIRE is actually realized as +1 hand at the target transition.

No new Guard, Contract, Recovery, or semantic Phase label.
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
EARLY_STEP = 168
LATE_STEP = 672


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


def run_one(seed: int, subject_seat: int, target_step: int | None, tag: str):
    model = load(BASE, f"phase_sign_model_{tag}_{seed}_{subject_seat}_{os.getpid()}")
    opp = load(OPPONENT, f"phase_sign_opp_{tag}_{seed}_{subject_seat}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    target_event = None

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))

        if subject_seat == 0:
            subject_obs, opp_obs = obs0, obs1
        else:
            subject_obs, opp_obs = obs1, obs0

        step = int(subject_obs.get("step", 0) or 0)
        subject_action = plain(model.agent(subject_obs))
        opponent_action = plain(opp.agent(opp_obs))

        if target_step is not None and step == target_step:
            player = int(subject_obs.get("player", subject_seat) or subject_seat)
            farm = subject_obs["farms"][player]
            original_market = deepcopy(subject_action.get("market", []) or [])
            original_hires = sum(
                1 for o in original_market
                if isinstance(o, list) and o and o[0] == "HIRE"
            )
            pre_hands = len(farm.get("hands", []) or [])
            pre_cash = float(farm.get("money", 0) or 0)
            pre_hires_today = int(farm.get("hires_today", 0) or 0)

            subject_action = deepcopy(subject_action)
            subject_action.setdefault("market", [])
            subject_action["market"].append(["HIRE"])

            target_event = {
                "step": step,
                "pre_cash": pre_cash,
                "pre_hands": pre_hands,
                "pre_hires_today": pre_hires_today,
                "original_hire_orders": original_hires,
                "original_market": original_market,
                "emitted_market": deepcopy(subject_action["market"]),
            }

        if subject_seat == 0:
            env.step([subject_action, opponent_action])
        else:
            env.step([opponent_action, subject_action])

        if target_event is not None and target_event.get("post_hands") is None:
            current_step = int(plain(shared(env, subject_seat)).get("step", 0) or 0)
            if current_step == target_event["step"] + 1:
                post_obs = plain(shared(env, subject_seat))
                post_farm = post_obs["farms"][subject_seat]
                target_event["post_cash"] = float(post_farm.get("money", 0) or 0)
                target_event["post_hands"] = len(post_farm.get("hands", []) or [])
                target_event["post_hires_today"] = int(post_farm.get("hires_today", 0) or 0)

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][subject_seat]["money"]),
        "terminal_opponent": float(final["farms"][1-subject_seat]["money"]),
        "target_event": target_event,
    }


def sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    cases = []

    for seed in SEEDS:
        for seat in (0, 1):
            base = run_one(seed, seat, None, "base")
            early = run_one(seed, seat, EARLY_STEP, "early")
            late = run_one(seed, seat, LATE_STEP, "late")

            early_event = early["target_event"]
            late_event = late["target_event"]

            early_realized_extra_hands = (
                early_event["post_hands"] - early_event["pre_hands"]
                if early_event is not None else None
            )
            late_realized_extra_hands = (
                late_event["post_hands"] - late_event["pre_hands"]
                if late_event is not None else None
            )

            # Baseline replay already requests 8 HIRE orders at both chosen steps.
            # We treat the intervention as mechanically exposed only if the
            # candidate emitted 9 total HIRE orders and post hands rose by 9.
            early_exposed = bool(
                early_event
                and early_event["original_hire_orders"] == 8
                and early_realized_extra_hands == 9
            )
            late_exposed = bool(
                late_event
                and late_event["original_hire_orders"] == 8
                and late_realized_extra_hands == 9
            )

            de = early["terminal_self"] - base["terminal_self"]
            dl = late["terminal_self"] - base["terminal_self"]

            case = {
                "seed": seed,
                "seat": seat,
                "baseline_terminal_self": base["terminal_self"],
                "early_terminal_self": early["terminal_self"],
                "late_terminal_self": late["terminal_self"],
                "delta_early": de,
                "delta_late": dl,
                "sign_early": sign(de),
                "sign_late": sign(dl),
                "sign_changed": sign(de) != sign(dl),
                "early_intervention_exposed": early_exposed,
                "late_intervention_exposed": late_exposed,
                "comparable": early_exposed and late_exposed,
                "early_event": early_event,
                "late_event": late_event,
            }
            cases.append(case)
            print("PHASE_SIGN_CASE " + json.dumps(case, separators=(",", ":")))

    comparable = [c for c in cases if c["comparable"]]
    de = [c["delta_early"] for c in comparable]
    dl = [c["delta_late"] for c in comparable]

    summary = {
        "worlds": len(cases),
        "comparable_worlds": len(comparable),
        "early_exposed": sum(c["early_intervention_exposed"] for c in cases),
        "late_exposed": sum(c["late_intervention_exposed"] for c in cases),
        "sign_changed_count": sum(c["sign_changed"] for c in comparable),
        "early_positive": sum(c["delta_early"] > 0 for c in comparable),
        "early_negative": sum(c["delta_early"] < 0 for c in comparable),
        "early_same": sum(c["delta_early"] == 0 for c in comparable),
        "late_positive": sum(c["delta_late"] > 0 for c in comparable),
        "late_negative": sum(c["delta_late"] < 0 for c in comparable),
        "late_same": sum(c["delta_late"] == 0 for c in comparable),
        "early_delta": summarize(de) if de else None,
        "late_delta": summarize(dl) if dl else None,
    }

    out = {
        "schema": "world-rhythm-phase-sign-hire-v0",
        "question": (
            "Does the terminal contribution of the same incremental HIRE +1 "
            "change sign at different world positions?"
        ),
        "base_model": "adaptive_replay_contract_runtime_v0.py",
        "opponent": "Seyamalam v21",
        "early_step": EARLY_STEP,
        "late_step": LATE_STEP,
        "seeds": SEEDS,
        "summary": summary,
        "cases": cases,
        "boundary": {
            "no_phase_semantics_assigned": True,
            "no_new_agent": True,
            "no_new_contract": True,
            "no_recovery": True,
            "sign_change_is_existence_evidence_only": True,
        },
    }

    Path("world_rhythm_phase_sign_hire_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PHASE_SIGN_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
