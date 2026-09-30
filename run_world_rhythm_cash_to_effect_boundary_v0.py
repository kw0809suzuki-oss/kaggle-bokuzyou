#!/usr/bin/env python3
"""World Rhythm — Cash-to-Effect Boundary Probe v0.

Known:
  step168 extra HIRE +1 is realized in all 20 worlds.
  The 9th worker then PASSes throughout steps169..191 and forcing PASS changes
  terminal by 0/20. So the added worker's own labor path is closed.

Question:
  After step168 creates the incremental HIRE cost, where does the resulting
  Cash difference first become a different REALIZED self effect, while the
  requested subject action is otherwise the same?

Design:
  BASE  = Adaptive Replay Runtime v0.1.
  EARLY = BASE + append exactly one HIRE at step168.

Comparison:
  - Ignore EARLY's trailing 9th-hand PASS when comparing requested actions.
  - Require farmer / market / first 8 hand requests to be equal.
  - Compare transition effects on self-local resources.
  - Return the first step >168 where equal requested core action yields a
    different self-effect signature.

No Phase semantics, no policy change, no new Contract.
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
START_COMPARE_STEP = 169


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


def delta_map(pre, post):
    keys = sorted(set(pre) | set(post))
    return {
        k: post.get(k, 0) - pre.get(k, 0)
        for k in keys
        if post.get(k, 0) != pre.get(k, 0)
    }


def tile_counts(tiles):
    out = {}
    for row in tiles:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                key = "PLANT:" + str(tile.get("crop"))
            elif kind in ("COOP", "BARN", "PASTURE", "SHED"):
                key = str(kind) + ":" + str(tile.get("animal", ""))
            else:
                continue
            out[key] = out.get(key, 0) + 1
    return out


def snapshot(obs, seat):
    farm = obs["farms"][seat]
    private = obs["private"]
    return {
        "cash": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "seeds": deepcopy(private.get("seeds", {}) or {}),
        "shed": deepcopy(private.get("shed", {}) or {}),
        "unlocked_quadrants": sorted(farm.get("unlocked_quadrants", []) or []),
        "tile_counts": tile_counts(farm.get("tiles", []) or []),
    }


def effect(pre, post):
    return {
        "cash_delta": post["cash"] - pre["cash"],
        "hands_delta": post["hands"] - pre["hands"],
        "hires_today_delta": post["hires_today"] - pre["hires_today"],
        "seed_delta": delta_map(pre["seeds"], post["seeds"]),
        "shed_delta": delta_map(pre["shed"], post["shed"]),
        "unlocked_added": [
            x for x in post["unlocked_quadrants"]
            if x not in pre["unlocked_quadrants"]
        ],
        "unlocked_removed": [
            x for x in pre["unlocked_quadrants"]
            if x not in post["unlocked_quadrants"]
        ],
        "tile_count_delta": delta_map(pre["tile_counts"], post["tile_counts"]),
    }


def core_action(action):
    hands = list(action.get("hands", []) or [])
    return {
        "farmer": deepcopy(action.get("farmer")),
        "hands_first8": deepcopy(hands[:8]),
        "market": deepcopy(action.get("market", []) or []),
    }


def run_trace(seed: int, seat: int, extra_hire: bool, tag: str):
    model = load(BASE, f"cash_effect_model_{tag}_{seed}_{seat}_{os.getpid()}")
    opp = load(OPPONENT, f"cash_effect_opp_{tag}_{seed}_{seat}_{os.getpid()}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    trace = []

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        subject_obs = obs0 if seat == 0 else obs1
        opponent_obs = obs1 if seat == 0 else obs0
        step = int(subject_obs.get("step", 0) or 0)

        subject_action = plain(model.agent(subject_obs))
        opponent_action = plain(opp.agent(opponent_obs))

        if extra_hire and step == HIRE_STEP:
            subject_action = deepcopy(subject_action)
            subject_action.setdefault("market", []).append(["HIRE"])

        pre = snapshot(subject_obs, seat)

        if seat == 0:
            env.step([subject_action, opponent_action])
        else:
            env.step([opponent_action, subject_action])

        post_obs = plain(shared(env, seat))
        post = snapshot(post_obs, seat)

        trace.append({
            "step": step,
            "subject_action": deepcopy(subject_action),
            "subject_core_action": core_action(subject_action),
            "opponent_action": deepcopy(opponent_action),
            "pre": pre,
            "post": post,
            "effect": effect(pre, post),
        })

    final = plain(env.state[0].observation)
    return {
        "terminal_self": float(final["farms"][seat]["money"]),
        "terminal_opponent": float(final["farms"][1-seat]["money"]),
        "trace": trace,
    }


def first_boundary(base, early):
    by_step_b = {x["step"]: x for x in base["trace"]}
    by_step_e = {x["step"]: x for x in early["trace"]}

    first_action_divergence = None
    first_equal_action_effect_divergence = None

    for step in range(START_COMPARE_STEP, min(max(by_step_b), max(by_step_e)) + 1):
        b = by_step_b[step]
        e = by_step_e[step]
        action_equal = b["subject_core_action"] == e["subject_core_action"]
        opponent_equal = b["opponent_action"] == e["opponent_action"]

        if not action_equal and first_action_divergence is None:
            first_action_divergence = {
                "step": step,
                "baseline_core_action": b["subject_core_action"],
                "early_core_action": e["subject_core_action"],
                "opponent_action_equal": opponent_equal,
            }

        if action_equal and b["effect"] != e["effect"]:
            first_equal_action_effect_divergence = {
                "step": step,
                "subject_core_action": b["subject_core_action"],
                "opponent_action_equal": opponent_equal,
                "baseline_pre_cash": b["pre"]["cash"],
                "early_pre_cash": e["pre"]["cash"],
                "pre_cash_difference_early_minus_base": e["pre"]["cash"] - b["pre"]["cash"],
                "baseline_effect": b["effect"],
                "early_effect": e["effect"],
                "baseline_post": b["post"],
                "early_post": e["post"],
            }
            break

    return {
        "first_subject_core_action_divergence": first_action_divergence,
        "first_equal_action_realized_effect_divergence": first_equal_action_effect_divergence,
    }


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def main():
    cases = []
    for seed in SEEDS:
        for seat in (0, 1):
            base = run_trace(seed, seat, False, "base")
            early = run_trace(seed, seat, True, "early")
            delta_terminal = early["terminal_self"] - base["terminal_self"]
            boundary = first_boundary(base, early)

            post168_b = base["trace"][HIRE_STEP]["post"]
            post168_e = early["trace"][HIRE_STEP]["post"]

            case = {
                "seed": seed,
                "seat": seat,
                "baseline_terminal_self": base["terminal_self"],
                "early_terminal_self": early["terminal_self"],
                "delta_terminal": delta_terminal,
                "phase_sign_group": "positive" if delta_terminal > 0 else "negative",
                "post168_cash_base": post168_b["cash"],
                "post168_cash_early": post168_e["cash"],
                "post168_cash_difference_early_minus_base": post168_e["cash"] - post168_b["cash"],
                "post168_hands_base": post168_b["hands"],
                "post168_hands_early": post168_e["hands"],
                "post168_hires_today_base": post168_b["hires_today"],
                "post168_hires_today_early": post168_e["hires_today"],
                **boundary,
            }
            cases.append(case)
            print("CASH_TO_EFFECT_BOUNDARY_CASE " + json.dumps(case, separators=(",", ":")))

    positive = [c for c in cases if c["delta_terminal"] > 0]
    negative = [c for c in cases if c["delta_terminal"] < 0]

    def steps(group, key):
        vals = []
        for c in group:
            item = c.get(key)
            if item is not None:
                vals.append(item["step"])
        return vals

    summary = {
        "worlds": len(cases),
        "positive_terminal_worlds": len(positive),
        "negative_terminal_worlds": len(negative),
        "post168_cash_minus34_count": sum(
            c["post168_cash_difference_early_minus_base"] == -34 for c in cases
        ),
        "positive_first_effect_steps": steps(
            positive, "first_equal_action_realized_effect_divergence"
        ),
        "negative_first_effect_steps": steps(
            negative, "first_equal_action_realized_effect_divergence"
        ),
        "positive_first_action_steps": steps(
            positive, "first_subject_core_action_divergence"
        ),
        "negative_first_action_steps": steps(
            negative, "first_subject_core_action_divergence"
        ),
    }

    out = {
        "schema": "world-rhythm-cash-to-effect-boundary-v0",
        "question": (
            "After step168 HIRE +1 creates the incremental Cash cost, what is "
            "the first later transition where the same requested core action "
            "produces a different realized self effect?"
        ),
        "summary": summary,
        "cases": cases,
        "boundary": {
            "worker_path_closed_by_prior_probe": True,
            "core_action_ignores_9th_trailing_pass": True,
            "first_equal_action_effect_divergence_is_primary": True,
            "no_phase_semantics": True,
            "no_new_policy": True,
            "no_new_contract": True,
        },
    }

    Path("world_rhythm_cash_to_effect_boundary_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("CASH_TO_EFFECT_BOUNDARY_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
