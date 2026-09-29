#!/usr/bin/env python3
"""Recovery single-shot terminal probe.

Runs the same seed/seat/opponent as the validated 24-step World Gate.
Baseline is ordinary Strong. Variant uses RecoveryOneRuntime, which injects
exactly one recovery allocation when it is first actually scheduled, then
returns to ordinary Strong behavior on the resulting World state.

Adoption evidence is terminal self only. Margin is recorded diagnostically.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from kaggle_environments import make

import run_terminal_return_world_gate_v0 as gate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "recovery_single_shot_terminal_v0_result.json"


def _status(env: Any, seat: int) -> str | None:
    try:
        state = env.state[seat]
        if isinstance(state, dict):
            return state.get("status")
        return getattr(state, "status", None)
    except Exception:
        return None


def _final_obs(env: Any, seat: int) -> dict[str, Any]:
    return gate.plain(env._Environment__get_shared_state(seat)["observation"]) if hasattr(gate, "plain") else copy.deepcopy(env._Environment__get_shared_state(seat)["observation"])


def _run(label: str, variant: bool) -> dict[str, Any]:
    opponent = gate._load_opponent()
    gate._reset(opponent)
    gate.strong_module.reset_agent()

    holder: dict[str, Any] = {"runtime": None}
    intervention_steps: list[int] = []
    scheduled_steps: list[int] = []

    def self_agent(obs: Any, configuration: Any):
        if not variant:
            return gate.strong_module.agent(obs, configuration)

        cfg = gate.settings_from(configuration)
        rt = holder["runtime"]
        if rt is None or rt.cfg != cfg:
            rt = gate.RecoveryOneRuntime(cfg)
            holder["runtime"] = rt

        action = rt.act(obs)
        allocation = rt.last_allocation or {}
        if allocation.get("added_recovery") is not None:
            intervention_steps.append(int(allocation["step"]))
        if allocation.get("recovery_scheduled"):
            scheduled_steps.append(int(allocation["step"]))
        return action

    env = make("kaggriculture", configuration={"seed": gate.SEED}, debug=False)
    agents = [opponent.agent, opponent.agent]
    agents[gate.SEAT] = self_agent
    env.run(agents)

    obs = copy.deepcopy(env._Environment__get_shared_state(gate.SEAT)["observation"])
    p = int(obs["player"])
    self_cash = float(obs["farms"][p]["money"])
    opponent_cash = float(obs["farms"][1 - p]["money"])
    return {
        "label": label,
        "seed": gate.SEED,
        "seat": gate.SEAT,
        "self_cash": self_cash,
        "opponent_cash": opponent_cash,
        "margin": self_cash - opponent_cash,
        "status": _status(env, gate.SEAT),
        "intervention_steps": intervention_steps,
        "scheduled_steps": scheduled_steps,
        "intervention_count": len(intervention_steps),
        "scheduled_intervention_count": len(scheduled_steps),
    }


def main() -> None:
    baseline = _run("baseline_strong", variant=False)
    candidate = _run("recovery_single_shot", variant=True)

    result = {
        "schema": "recovery-single-shot-terminal-v0",
        "conditions": {
            "seed": gate.SEED,
            "seat": gate.SEAT,
            "opponent": "Seyamalam pinned v21",
            "official_world_commit": "d7729da06cc1382eb742d6980dc3180aa85caa28",
            "world_gate_already_validated": True,
            "single_shot": True,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "terminal_self_delta": candidate["self_cash"] - baseline["self_cash"],
            "terminal_margin_delta": candidate["margin"] - baseline["margin"],
            "valid_single_shot": (
                candidate["intervention_count"] == 1
                and candidate["scheduled_intervention_count"] == 1
            ),
            "adoption_rule": "terminal self only",
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
