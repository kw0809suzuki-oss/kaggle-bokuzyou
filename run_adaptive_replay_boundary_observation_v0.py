#!/usr/bin/env python3
"""Boundary Observation Probe v0.

Purpose:
    Observe where the already-proven step24 HIRE Effect Contract keeps the
    same meaning as World conditions change.

This is not a strength-optimization run.
No new Guard / Recovery / policy rule is added.

Grid:
    opponent in {Seyamalam v21, frozen DECEM Replay}
    subject seat in {0, 1}
    fresh seeds x5

For every world, compare:
    frozen DECEM Replay (no Guard)
    vs
    Adaptive Replay Contract Runtime v0

Primary observation:
    step24 Contract input distribution
      (money, hires_today, replay_requested_hires, feasible_hires, hands)

Secondary observation:
    Guard response and terminal delta.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent

BASELINE = ROOT / "decem_replay_distilled_157026_v0.py"
CANDIDATE = ROOT / "adaptive_replay_contract_runtime_v0.py"
SKELETON = ROOT / "decem_replay_distilled_157026_v0.py"
WORLD_ADAPTER = ROOT / "adaptive_replay_world_adapter_v0.py"

OPPONENTS = {
    "seyamalam_v21": ROOT / "astra_flow_vendor" / "seyamalam_v21.py",
    "decem_replay": ROOT / "decem_replay_distilled_157026_v0.py",
}

SEEDS = [93803001, 93803002, 93803003, 93803004, 93803005]
SUBJECT_SEATS = [0, 1]


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


def count_market_operation(market_orders, operation):
    return sum(
        1
        for order in (market_orders or [])
        if isinstance(order, (list, tuple))
        and len(order) >= 1
        and order[0] == operation
    )


def run_one(seed, subject_seat, opponent_name, model_path, tag):
    model = load(model_path, f"subject_{tag}_{opponent_name}_{subject_seat}_{seed}_{os.getpid()}")
    opponent = load(
        OPPONENTS[opponent_name],
        f"opponent_{tag}_{opponent_name}_{subject_seat}_{seed}_{os.getpid()}",
    )
    skeleton = load(
        SKELETON,
        f"skeleton_{tag}_{opponent_name}_{subject_seat}_{seed}_{os.getpid()}",
    )
    world = load(
        WORLD_ADAPTER,
        f"world_{tag}_{opponent_name}_{subject_seat}_{seed}_{os.getpid()}",
    )

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    step24 = {
        "pre": None,
        "replay_requested_action": None,
        "issued_action": None,
        "post": None,
    }

    while not env.done:
        subject_obs = plain(shared(env, subject_seat))
        opponent_seat = 1 - subject_seat
        opponent_obs = plain(shared(env, opponent_seat))

        step = int(subject_obs.get("step", 0) or 0)

        if step == 24:
            replay_action = plain(skeleton.agent(subject_obs))
            replay_requested_hires = count_market_operation(
                replay_action.get("market", []), "HIRE"
            )

            player = int(subject_obs.get("player", subject_seat) or subject_seat)
            farm = subject_obs["farms"][player]
            money = float(farm.get("money", 0) or 0)
            hires_today = int(farm.get("hires_today", 0) or 0)
            hands = len(farm.get("hands", []) or [])

            feasible = world.count_feasible_hires(
                money=money,
                hires_today=hires_today,
                requested=replay_requested_hires,
                cost_mult=1.0,
            )

            step24["pre"] = {
                "money": money,
                "hires_today": hires_today,
                "hands": hands,
                "replay_requested_hires": replay_requested_hires,
                "feasible_hires": feasible,
            }
            step24["replay_requested_action"] = replay_action

        subject_action = plain(model.agent(subject_obs))
        opponent_action = plain(opponent.agent(opponent_obs))

        if step == 24:
            step24["issued_action"] = subject_action

        actions = [None, None]
        actions[subject_seat] = subject_action
        actions[1 - subject_seat] = opponent_action
        env.step(actions)

        if step == 24 and not env.done:
            post_obs = plain(shared(env, subject_seat))
            player = int(post_obs.get("player", subject_seat) or subject_seat)
            farm = post_obs["farms"][player]
            pre_hands = int(step24["pre"]["hands"])
            post_hands = len(farm.get("hands", []) or [])
            step24["post"] = {
                "money": float(farm.get("money", 0) or 0),
                "hires_today": int(farm.get("hires_today", 0) or 0),
                "hands": post_hands,
                "realized_hires_delta": post_hands - pre_hands,
            }

    final = plain(env.state[subject_seat].observation)
    subject_cash = float(final["farms"][subject_seat]["money"])
    opponent_cash = float(final["farms"][1 - subject_seat]["money"])

    return {
        "terminal_self": subject_cash,
        "terminal_opponent": opponent_cash,
        "margin": subject_cash - opponent_cash,
        "guard_trigger_count": int(getattr(model, "trigger_count", 0)),
        "guard_event": plain(getattr(model, "last_guard_event", None)),
        "contract_id": plain(getattr(model, "last_contract_id", None)),
        "step24": step24,
    }


def input_key(pre):
    if not pre:
        return "missing"
    return "|".join(
        str(pre[k])
        for k in [
            "money",
            "hires_today",
            "hands",
            "replay_requested_hires",
            "feasible_hires",
        ]
    )


def summarize(cases):
    deltas = [c["delta_terminal_self"] for c in cases]
    dist = Counter(input_key(c["candidate"]["step24"]["pre"]) for c in cases)

    by_world = defaultdict(list)
    for c in cases:
        by_world[(c["opponent"], c["subject_seat"])].append(c)

    world_summary = {}
    for (opponent, seat), rows in sorted(by_world.items()):
        ds = [r["delta_terminal_self"] for r in rows]
        inputs = Counter(input_key(r["candidate"]["step24"]["pre"]) for r in rows)
        world_summary[f"{opponent}|seat{seat}"] = {
            "n": len(rows),
            "improved": sum(d > 0 for d in ds),
            "worsened": sum(d < 0 for d in ds),
            "same": sum(d == 0 for d in ds),
            "mean_delta_terminal_self": statistics.mean(ds),
            "guard_triggered": sum(r["candidate"]["guard_trigger_count"] > 0 for r in rows),
            "prestate_equal_ab": sum(r["step24_pre_equal"] for r in rows),
            "input_distribution": dict(inputs),
        }

    return {
        "n": len(cases),
        "improved": sum(d > 0 for d in deltas),
        "worsened": sum(d < 0 for d in deltas),
        "same": sum(d == 0 for d in deltas),
        "mean_delta_terminal_self": statistics.mean(deltas),
        "median_delta_terminal_self": statistics.median(deltas),
        "guard_triggered_cases": sum(
            c["candidate"]["guard_trigger_count"] > 0 for c in cases
        ),
        "step24_pre_equal_ab": sum(c["step24_pre_equal"] for c in cases),
        "unique_contract_inputs": len(dist),
        "contract_input_distribution": dict(dist),
        "worlds": world_summary,
    }


def main():
    cases = []

    for opponent_name in OPPONENTS:
        for subject_seat in SUBJECT_SEATS:
            for seed in SEEDS:
                baseline = run_one(
                    seed,
                    subject_seat,
                    opponent_name,
                    BASELINE,
                    "baseline",
                )
                candidate = run_one(
                    seed,
                    subject_seat,
                    opponent_name,
                    CANDIDATE,
                    "candidate",
                )

                base_pre = baseline["step24"]["pre"]
                cand_pre = candidate["step24"]["pre"]
                row = {
                    "opponent": opponent_name,
                    "subject_seat": subject_seat,
                    "seed": seed,
                    "step24_pre_equal": base_pre == cand_pre,
                    "contract_input": cand_pre,
                    "baseline": baseline,
                    "candidate": candidate,
                    "delta_terminal_self": (
                        candidate["terminal_self"] - baseline["terminal_self"]
                    ),
                    "delta_margin": candidate["margin"] - baseline["margin"],
                }
                cases.append(row)
                print(
                    "BOUNDARY_OBSERVATION_CASE "
                    + json.dumps(row, separators=(",", ":"))
                )

    summary = summarize(cases)

    out = {
        "schema": "adaptive-replay-boundary-observation-v0",
        "objective": (
            "Observe Contract input diversity and Runtime response. "
            "Do not optimize strength."
        ),
        "candidate": "adaptive_replay_contract_runtime_v0.py",
        "baseline": "decem_replay_distilled_157026_v0.py",
        "fresh_seeds": SEEDS,
        "subject_seats": SUBJECT_SEATS,
        "opponents": list(OPPONENTS),
        "summary": summary,
        "cases": cases,
        "boundary": {
            "no_new_guard": True,
            "no_recovery": True,
            "no_policy_change": True,
            "decem_replay_opponent_role": (
                "controlled alternate shared-Market world; not a strength benchmark"
            ),
            "interpretation_rule": (
                "If inputs do not diversify, adaptation remains unexposed. "
                "If inputs diversify, inspect Runtime response before any new intervention."
            ),
        },
    }

    Path("adaptive_replay_boundary_observation_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "BOUNDARY_OBSERVATION_SUMMARY "
        + json.dumps(summary, separators=(",", ":"))
    )


if __name__ == "__main__":
    main()
