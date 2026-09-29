#!/usr/bin/env python3
"""Strong selection-state parity gate.

Purpose:
Capture the *actual* selection mode around Strong choose() and verify that
replaying that captured mode through the existing _service_action() reproduces
the exact chosen Action before any recovery intervention is attempted.

This is an entrance/parity test only. No strength claim and no terminal test.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from plan_generator_entrance_v0 import bind_official_state

agent_module = importlib.import_module("strong_model_v0_reimplementation.agent")
planner = importlib.import_module("strong_model_v0_reimplementation.planner")
jobs_module = importlib.import_module("strong_model_v0_reimplementation.jobs")

SEED = 92804001
SEAT = 0
STEPS = 24
OPPONENT_PATH = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"
OUT = ROOT / "strong_selection_state_parity_gate_v0_result.json"


def _load_opponent():
    spec = importlib.util.spec_from_file_location("selection_parity_opponent", OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {OPPONENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, separators=(",", ":"), default=str) == json.dumps(
        b, sort_keys=True, separators=(",", ":"), default=str
    )


def _active_from_specs(raw: dict[str, Any], cfg: Any, active_specs: dict[str, dict[str, Any]]):
    fresh = jobs_module.fresh_jobs(raw, cfg.episodeSteps, cfg.turnsPerDay, cfg.boardSize)
    investments = planner.investment_jobs(raw, cfg, fresh)
    lookup = {j.key: j for j in investments + fresh + planner.operating_jobs(raw, cfg)}
    active = []
    for key, spec in active_specs.items():
        if not jobs_module.still_needed(spec, raw):
            continue
        j = lookup.get(key) or jobs_module.materialize_active(spec, raw)
        if j is not None:
            j.active = True
            active.append(j)
    return active


def _capture_and_reconstruct(raw: dict[str, Any], cfg: Any, active_specs: dict[str, dict[str, Any]]):
    chosen, representative, active, continuation = planner.choose(raw, cfg, active_specs)
    target = chosen.action

    candidates: list[dict[str, Any]] = []

    def add(mode: str, planned, action, forced=None, harvest_now=False, extra=None):
        candidates.append({
            "mode": mode,
            "planned": planned,
            "action": action,
            "forced": set(forced or []),
            "harvest_now": bool(harvest_now),
            "extra": extra,
        })

    # Reconstruct exactly the modes choose() can create.
    a, s, _, _ = planner._service_action(raw, cfg, active)
    add("active", active, a)

    if active:
        a0, s0, _, _ = planner._service_action(raw, cfg, [])
        add("empty", [], a0)

    if representative is not None:
        planned = active + [representative]
        ar, sr, _, _ = planner._service_action(raw, cfg, planned, {representative.key})
        add("representative", planned, ar, {representative.key})

    ah, sh, _, _ = planner._service_action(raw, cfg, active, harvest_now=True)
    add("harvest_now", active, ah, harvest_now=True)

    if (
        raw["hour"] < cfg.turnsPerDay - 1
        and (active or planner.operating_jobs(raw, cfg))
        and len(a["market"]) < cfg.maxMarketOrdersPerTurn
    ):
        hire = copy.deepcopy(a)
        hire["market"].append(["HIRE"])
        add("hire_postprocess", active, hire, extra={"postprocess": "append_hire"})

    for idx, order in enumerate(a["market"]):
        if order[0] == "SELL":
            no_sell = copy.deepcopy(a)
            removed = no_sell["market"].pop(idx)
            add(
                "remove_sell_postprocess",
                active,
                no_sell,
                extra={"postprocess": "remove_sell", "index": idx, "removed": removed},
            )

    matches = [row for row in candidates if _same(row["action"], target)]
    if not matches:
        return chosen, {
            "parity": False,
            "reason": "no_reconstruction_mode_matched_chosen_action",
            "chosen_action": target,
            "representative": None if representative is None else representative.spec(),
            "candidate_modes": [r["mode"] for r in candidates],
        }

    # Prefer the structurally most specific match.
    priority = {
        "representative": 0,
        "harvest_now": 1,
        "hire_postprocess": 2,
        "remove_sell_postprocess": 3,
        "empty": 4,
        "active": 5,
    }
    row = min(matches, key=lambda r: priority.get(r["mode"], 99))

    replay, scheduled, live, _ = planner._service_action(
        raw,
        cfg,
        row["planned"],
        forced=row["forced"] or None,
        harvest_now=row["harvest_now"],
    )
    if row["extra"]:
        if row["extra"]["postprocess"] == "append_hire":
            replay = copy.deepcopy(replay)
            replay["market"].append(["HIRE"])
        elif row["extra"]["postprocess"] == "remove_sell":
            replay = copy.deepcopy(replay)
            replay["market"].pop(int(row["extra"]["index"]))

    parity = _same(replay, target)
    state = {
        "parity": parity,
        "mode": row["mode"],
        "representative": None if representative is None else representative.spec(),
        "planned_keys": [j.key for j in row["planned"]],
        "forced": sorted(row["forced"]),
        "harvest_now": row["harvest_now"],
        "extra": row["extra"],
        "scheduled": list(scheduled),
        "chosen_action": target,
        "replayed_action": replay,
        "matching_modes": [r["mode"] for r in matches],
    }
    return chosen, state


def main() -> None:
    opponent = _load_opponent()
    reset = getattr(opponent, "reset_agent", None)
    if callable(reset):
        reset()
    agent_module.reset_agent()

    runtime = None
    rows = []

    def self_agent(obs: Any, configuration: Any):
        nonlocal runtime
        cfg = planner.settings_from(configuration)
        if runtime is None:
            runtime = agent_module.Runtime(cfg)

        snapshot = bind_official_state(obs)
        raw = snapshot.raw()
        raw["step"] = planner.step_of(raw, cfg)
        step = int(raw["step"])

        runtime.active = {
            key: spec
            for key, spec in runtime.active.items()
            if jobs_module.still_needed(spec, raw)
        }

        chosen, captured = _capture_and_reconstruct(raw, cfg, runtime.active)
        runtime.active = {spec["key"]: copy.deepcopy(spec) for spec in chosen.commitments}
        runtime.last_step = step

        rows.append({
            "step": step,
            **captured,
        })

        if step >= STEPS - 1:
            # We only need the first 24 decisions for parity.
            return copy.deepcopy(chosen.action)
        return copy.deepcopy(chosen.action)

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    agents = [opponent.agent, opponent.agent]
    agents[SEAT] = self_agent
    env.run(agents)

    first_fail = next((r["step"] for r in rows[:STEPS] if not r["parity"]), None)
    result = {
        "schema": "strong-selection-state-parity-gate-v0",
        "conditions": {
            "seed": SEED,
            "seat": SEAT,
            "steps_checked": min(STEPS, len(rows)),
            "opponent": "Seyamalam pinned v21",
            "official_world_commit": "d7729da06cc1382eb742d6980dc3180aa85caa28",
        },
        "summary": {
            "checked": min(STEPS, len(rows)),
            "matched": sum(1 for r in rows[:STEPS] if r["parity"]),
            "first_failure_step": first_fail,
            "pass": first_fail is None and len(rows) >= STEPS,
        },
        "rows": rows[:STEPS],
        "boundary": (
            "PASS only means the chosen Strong Action can be reconstructed from an "
            "explicit captured selection mode before intervention. It is not a strength result."
        ),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
