#!/usr/bin/env python3
"""World Rhythm / Future Option Order — Order Swap Probe v0.

Question:
  Does changing ONLY the execution order of the same two market operations
  change the realized World transition?

Target frozen Replay action at step217:
  A -> B = HIRE -> BUY_PRODUCT WHEAT 19

Swap candidate:
  B -> A = BUY_PRODUCT WHEAT 19 -> HIRE

Everything else is held fixed:
  - same start state
  - same farmer action
  - same hand actions
  - same two market operations
  - same operation counts
  - same opponent
  - same seed and seat

Primary observation:
  Compare the single transition step217 -> step218.
  We do NOT use terminal as evidence in v0.

If post-state/effect differs, that is direct evidence that order alone changes
the realized World transition under that State.

No Phase semantics. No policy promotion.
"""
from __future__ import annotations

import importlib.util
import json
import os
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
TARGET_STEP = 217

AB = [["HIRE"], ["BUY_PRODUCT", "WHEAT", 19]]
BA = [["BUY_PRODUCT", "WHEAT", 19], ["HIRE"]]


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


def plant_count(farm, crop):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
                n += 1
    return n


def snap(obs, seat):
    farm = obs["farms"][seat]
    priv = obs["private"]
    return {
        "cash": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "seed_wheat": int((priv.get("seeds", {}) or {}).get("WHEAT", 0) or 0),
        "shed_wheat": int((priv.get("shed", {}) or {}).get("WHEAT", 0) or 0),
        "plant_wheat": plant_count(farm, "WHEAT"),
    }


def effect(pre, post):
    return {
        "cash_delta": post["cash"] - pre["cash"],
        "hands_delta": post["hands"] - pre["hands"],
        "hires_today_delta": post["hires_today"] - pre["hires_today"],
        "seed_wheat_delta": post["seed_wheat"] - pre["seed_wheat"],
        "shed_wheat_delta": post["shed_wheat"] - pre["shed_wheat"],
        "plant_wheat_delta": post["plant_wheat"] - pre["plant_wheat"],
    }


def run_to_transition(seed: int, seat: int, mode: str):
    model = load(BASE, f"order_swap_model_{mode}_{seed}_{seat}_{os.getpid()}")
    opp = load(OPPONENT, f"order_swap_opp_{mode}_{seed}_{seat}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    result = None

    while not env.done:
        o0 = plain(shared(env, 0))
        o1 = plain(shared(env, 1))
        so = o0 if seat == 0 else o1
        oo = o1 if seat == 0 else o0

        step = int(so.get("step", 0) or 0)
        sa = plain(model.agent(so))
        oa = plain(opp.agent(oo))

        if step == TARGET_STEP:
            original = deepcopy(sa)
            original_market = deepcopy(sa.get("market", []) or [])

            if original_market != AB:
                raise RuntimeError(
                    f"Unexpected step217 market action for seed={seed} seat={seat}: "
                    f"{original_market!r}"
                )

            if mode == "ba":
                sa = deepcopy(sa)
                sa["market"] = deepcopy(BA)

            pre = snap(so, seat)

            if seat == 0:
                env.step([sa, oa])
            else:
                env.step([oa, sa])

            post_obs = plain(shared(env, seat))
            post = snap(post_obs, seat)

            result = {
                "step": TARGET_STEP,
                "pre": pre,
                "original_action": original,
                "emitted_action": deepcopy(sa),
                "opponent_action": deepcopy(oa),
                "post": post,
                "effect": effect(pre, post),
            }
            break

        if seat == 0:
            env.step([sa, oa])
        else:
            env.step([oa, sa])

    if result is None:
        raise RuntimeError("Target step was not reached")
    return result


def main():
    cases = []

    for seed in SEEDS:
        for seat in (0, 1):
            ab = run_to_transition(seed, seat, "ab")
            ba = run_to_transition(seed, seat, "ba")

            pre_equal = ab["pre"] == ba["pre"]
            non_market_equal = {
                "farmer": ab["emitted_action"].get("farmer"),
                "hands": ab["emitted_action"].get("hands"),
            } == {
                "farmer": ba["emitted_action"].get("farmer"),
                "hands": ba["emitted_action"].get("hands"),
            }

            same_market_multiset = sorted(map(json.dumps, ab["emitted_action"]["market"])) == sorted(
                map(json.dumps, ba["emitted_action"]["market"])
            )

            post_equal = ab["post"] == ba["post"]
            effect_equal = ab["effect"] == ba["effect"]

            case = {
                "seed": seed,
                "seat": seat,
                "pre_state_equal": pre_equal,
                "non_market_action_equal": non_market_equal,
                "same_market_operation_multiset": same_market_multiset,
                "ab_market": ab["emitted_action"]["market"],
                "ba_market": ba["emitted_action"]["market"],
                "ab_pre": ab["pre"],
                "ba_pre": ba["pre"],
                "ab_effect": ab["effect"],
                "ba_effect": ba["effect"],
                "ab_post": ab["post"],
                "ba_post": ba["post"],
                "effect_equal": effect_equal,
                "post_state_equal": post_equal,
                "order_created_world_divergence": not post_equal,
            }
            cases.append(case)
            print("ORDER_SWAP_CASE " + json.dumps(case, separators=(",", ":")))

    valid = [
        c for c in cases
        if c["pre_state_equal"]
        and c["non_market_action_equal"]
        and c["same_market_operation_multiset"]
    ]

    summary = {
        "worlds": len(cases),
        "valid_order_only_comparisons": len(valid),
        "world_divergence_count": sum(c["order_created_world_divergence"] for c in valid),
        "effect_divergence_count": sum(not c["effect_equal"] for c in valid),
        "identical_post_count": sum(c["post_state_equal"] for c in valid),
    }

    out = {
        "schema": "world-rhythm-order-swap-v0",
        "question": (
            "With the same start State and same two market operations, does "
            "HIRE -> BUY_PRODUCT WHEAT 19 differ from "
            "BUY_PRODUCT WHEAT 19 -> HIRE in the realized step217->218 World?"
        ),
        "summary": summary,
        "cases": cases,
        "boundary": {
            "single_transition_only": True,
            "terminal_not_used": True,
            "same_operation_multiset_required": True,
            "no_phase_semantics": True,
            "no_policy_promotion": True,
        },
    }

    Path("world_rhythm_order_swap_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("ORDER_SWAP_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
