#!/usr/bin/env python3
"""Terminal Return Operating Model v0 — one-variant 24-step World gate.

This is intentionally NOT a terminal strength test.

Baseline:
    ordinary Strong Model v0 reimplementation for steps 0..23.

Variant:
    capture the exact selected Strong choice state (representative, forced,
    harvest-now, and native post-processing mode), add at most ONE currently
    executable recovery job, then generate the Action through `_service_action()`.

No Action is patched after generation.

The gate asks only whether this single allocation change produces an
observable World-state divergence within 24 actions under the same
seed / seat / opponent.

Output:
    terminal_return_world_gate_v0_result.json
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from plan_generator_entrance_v0 import bind_official_state
from importlib import import_module
strong_module = import_module("strong_model_v0_reimplementation.agent")
from strong_model_v0_reimplementation.jobs import still_needed
from strong_model_v0_reimplementation.planner import (
    _service_action,
    choose,
    investment_jobs,
    operating_jobs,
    settings_from,
    step_of,
)

SEED = 92804001
SEAT = 0
GATE_ACTIONS = 24
OPPONENT_PATH = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"
OUT = ROOT / "terminal_return_world_gate_v0_result.json"


def _load_opponent():
    spec = importlib.util.spec_from_file_location("terminal_return_gate_opponent", OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {OPPONENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset(obj: Any) -> None:
    fn = getattr(obj, "reset_agent", None)
    if callable(fn):
        fn()


def _pass_action(obs: Any) -> dict[str, Any]:
    farm = obs["farms"][int(obs["player"])]
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in (farm.get("hands", []) or [])],
        "market": [],
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _world_signature(obs: dict[str, Any]) -> dict[str, Any]:
    """Visible World + self private state. This is the gate comparison surface."""
    p = int(obs["player"])
    return {
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "farms": copy.deepcopy(obs["farms"]),
        "market": copy.deepcopy(obs["market"]),
        "town": copy.deepcopy(obs["town"]),
        "self_private": copy.deepcopy(obs["private"]),
        "self_seat": p,
    }


def _metrics(obs: dict[str, Any]) -> dict[str, Any]:
    p = int(obs["player"])
    farm = obs["farms"][p]
    plants = animals = yield_units = 0
    for row in farm.get("tiles", []) or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                plants += 1
            if tile.get("animal"):
                animals += 1
            yield_units += int(tile.get("yield_units", 0) or 0)

    carried_maps = [dict(v or {}) for v in (obs["private"].get("inventories", []) or [])]
    shed = dict(obs["private"].get("shed", {}) or {})
    carried_total = sum(int(q or 0) for inv in carried_maps for q in inv.values())
    shed_total = sum(int(q or 0) for q in shed.values())

    return {
        "self_cash": float(farm["money"]),
        "opponent_cash": float(obs["farms"][1 - p]["money"]),
        "hands": len(farm.get("hands", []) or []),
        "plants": plants,
        "animals": animals,
        "productive_assets": plants + animals,
        "tile_yield_units": yield_units,
        "carried_total_items": carried_total,
        "shed_total_items": shed_total,
        "carried": carried_maps,
        "shed": shed,
    }


def _diff_paths(a: Any, b: Any, path: str = "", limit: int = 80) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(x: Any, y: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if type(x) is not type(y):
            out.append({"path": p, "baseline": x, "variant": y})
            return
        if isinstance(x, dict):
            for key in sorted(set(x) | set(y), key=str):
                q = f"{p}.{key}" if p else str(key)
                if key not in x:
                    out.append({"path": q, "baseline": "<missing>", "variant": y[key]})
                elif key not in y:
                    out.append({"path": q, "baseline": x[key], "variant": "<missing>"})
                else:
                    walk(x[key], y[key], q)
                if len(out) >= limit:
                    return
            return
        if isinstance(x, list):
            if len(x) != len(y):
                out.append({"path": p + ".length", "baseline": len(x), "variant": len(y)})
            for i, (xx, yy) in enumerate(zip(x, y)):
                walk(xx, yy, f"{p}[{i}]")
                if len(out) >= limit:
                    return
            return
        if x != y:
            out.append({"path": p, "baseline": x, "variant": y})

    walk(a, b, path)
    return out


def _same(a: Any, b: Any) -> bool:
    return _canonical(a) == _canonical(b)


def _capture_selected_state(raw: dict[str, Any], cfg: Any, active_specs: dict[str, Any]):
    """Return Strong's selected branch, including the mode needed to replay it."""
    chosen, representative, active, _ = choose(raw, cfg, active_specs)
    target = chosen.action
    candidates: list[dict[str, Any]] = []

    def add(mode, planned, action, scheduled, forced=None, harvest_now=False, extra=None):
        candidates.append({
            "mode": mode,
            "planned": planned,
            "action": action,
            "scheduled": list(scheduled),
            "forced": set(forced or []),
            "harvest_now": bool(harvest_now),
            "extra": extra,
        })

    action, scheduled, _, _ = _service_action(raw, cfg, active)
    add("active", active, action, scheduled)
    if active:
        action_empty, scheduled_empty, _, _ = _service_action(raw, cfg, [])
        add("empty", [], action_empty, scheduled_empty)
    if representative is not None:
        planned = active + [representative]
        action_rep, scheduled_rep, _, _ = _service_action(
            raw, cfg, planned, {representative.key}
        )
        add("representative", planned, action_rep, scheduled_rep, {representative.key})

    action_harvest, scheduled_harvest, _, _ = _service_action(
        raw, cfg, active, harvest_now=True
    )
    add("harvest_now", active, action_harvest, scheduled_harvest, harvest_now=True)

    if (
        raw["hour"] < cfg.turnsPerDay - 1
        and (active or operating_jobs(raw, cfg))
        and len(action["market"]) < cfg.maxMarketOrdersPerTurn
    ):
        action_hire = copy.deepcopy(action)
        action_hire["market"].append(["HIRE"])
        add("hire_postprocess", active, action_hire, scheduled,
            extra={"postprocess": "append_hire"})

    for idx, order in enumerate(action["market"]):
        if order[0] == "SELL":
            action_no_sell = copy.deepcopy(action)
            removed = action_no_sell["market"].pop(idx)
            add("remove_sell_postprocess", active, action_no_sell, scheduled,
                extra={"postprocess": "remove_sell", "index": idx, "removed": removed})

    matches = [row for row in candidates if _same(row["action"], target)]
    if not matches:
        raise RuntimeError("selected Strong Action has no reconstructable selection state")
    priority = {
        "representative": 0,
        "harvest_now": 1,
        "hire_postprocess": 2,
        "remove_sell_postprocess": 3,
        "empty": 4,
        "active": 5,
    }
    selected = min(matches, key=lambda row: priority[row["mode"]])
    replay, replay_scheduled, _, _ = _service_action(
        raw,
        cfg,
        selected["planned"],
        forced=selected["forced"] or None,
        harvest_now=selected["harvest_now"],
    )
    replay = _apply_native_postprocess(replay, selected["extra"])
    if not _same(replay, target):
        raise RuntimeError("captured Strong selection state failed Action parity")
    selected["scheduled"] = replay_scheduled
    selected["chosen_action"] = copy.deepcopy(target)
    selected["matching_modes"] = [row["mode"] for row in matches]
    selected["base_commitments"] = copy.deepcopy(chosen.commitments)
    return chosen, selected


def _apply_native_postprocess(action: dict[str, Any], extra: dict[str, Any] | None):
    """Reapply a native Strong choice mode, not a probe-specific Action patch."""
    out = copy.deepcopy(action)
    if not extra:
        return out
    if extra["postprocess"] == "append_hire":
        out["market"].append(["HIRE"])
    elif extra["postprocess"] == "remove_sell":
        target = extra["removed"]
        matches = [i for i, order in enumerate(out["market"]) if order == target]
        if len(matches) != 1:
            raise RuntimeError("native SELL-removal selection could not be replayed after allocation")
        out["market"].pop(matches[0])
    return out


class RecoveryOneRuntime:
    """Single-shot allocation change: inject one scheduled recovery commitment once."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.active: dict[str, dict[str, Any]] = {}
        self.last_step = -1
        self.last_allocation: dict[str, Any] | None = None
        self.intervened = False

    def reset(self) -> None:
        self.active = {}
        self.last_step = -1
        self.last_allocation = None
        self.intervened = False

    def act(self, obs: Any) -> dict[str, Any]:
        snapshot = bind_official_state(obs)
        raw = snapshot.raw()
        raw["step"] = step_of(raw, self.cfg)
        step = int(raw["step"])
        if step <= self.last_step:
            self.reset()

        self.active = {
            key: spec
            for key, spec in self.active.items()
            if still_needed(spec, raw)
        }

        # Capture and replay the exact Strong selection branch before intervention.
        chosen, selection = _capture_selected_state(raw, self.cfg, self.active)
        planned = list(selection["planned"])
        base_specs = copy.deepcopy(chosen.commitments)
        base_keys = {str(spec["key"]) for spec in base_specs}

        # Add one currently executable recovery allocation to that same branch.
        recovery = [
            j
            for j in operating_jobs(raw, self.cfg, harvest_now=True)
            if j.category == "recovery" and j.key not in base_keys
        ]
        added = None
        forced = set(selection["forced"])
        if recovery and not self.intervened:
            # Single-shot probe: inject only until the first recovery job is actually scheduled.
            added = max(recovery, key=lambda j: (float(j.central_delta), j.key))
            planned.append(added)
            forced.add(added.key)

        action, scheduled, live, _ = _service_action(
            raw,
            self.cfg,
            planned,
            forced=forced or None,
            harvest_now=selection["harvest_now"],
        )
        action = _apply_native_postprocess(action, selection["extra"])

        recovery_scheduled = added is not None and added.key in scheduled
        if recovery_scheduled:
            self.intervened = True

        # Single-shot boundary: after this Action, return to ordinary Strong commitments.
        # Do not carry the injected recovery job into the next turn.
        next_specs = list(base_specs)
        self.active = {
            str(spec["key"]): copy.deepcopy(spec)
            for spec in next_specs
            if still_needed(spec, raw)
        }
        self.last_allocation = {
            "step": step,
            "selection_state_reconstructed": True,
            "selection_mode": selection["mode"],
            "matching_modes": selection["matching_modes"],
            "base_commitments": [s["key"] for s in base_specs],
            "added_recovery": None if added is None else added.spec(),
            "scheduled": list(scheduled),
            "recovery_scheduled": recovery_scheduled,
            "single_shot_already_fired": self.intervened,
            "native_postprocess": selection["extra"],
            "chosen_action_before_intervention": selection["chosen_action"],
        }
        self.last_step = step
        return copy.deepcopy(action)


def _run(label: str, variant: bool) -> dict[str, Any]:
    opponent = _load_opponent()
    _reset(opponent)
    strong_module.reset_agent()

    runtime_holder: dict[str, Any] = {"runtime": None}
    baseline_holder: dict[str, Any] = {"active": {}, "last_step": -1}
    trace: list[dict[str, Any]] = []

    def self_agent(obs: Any, configuration: Any):
        step = int(obs["step"])
        raw_obs = copy.deepcopy(obs)
        if step >= GATE_ACTIONS:
            action = _pass_action(obs)
            if step == GATE_ACTIONS:
                trace.append({
                    "step": step,
                    "phase": "gate_state",
                    "metrics": _metrics(raw_obs),
                    "world_signature": _world_signature(raw_obs),
                    "world_digest": _digest(_world_signature(raw_obs)),
                    "action": action,
                    "allocation": None,
                })
            return action

        if variant:
            rt = runtime_holder["runtime"]
            cfg = settings_from(configuration)
            if rt is None or rt.cfg != cfg:
                rt = RecoveryOneRuntime(cfg)
                runtime_holder["runtime"] = rt
            action = rt.act(obs)
            allocation = copy.deepcopy(rt.last_allocation)
        else:
            cfg = settings_from(configuration)
            baseline = baseline_holder
            if step <= baseline["last_step"]:
                baseline["active"] = {}
            raw = bind_official_state(obs).raw()
            raw["step"] = step_of(raw, cfg)
            baseline["active"] = {
                key: spec for key, spec in baseline["active"].items()
                if still_needed(spec, raw)
            }
            chosen, selection = _capture_selected_state(raw, cfg, baseline["active"])
            baseline["active"] = {
                str(spec["key"]): copy.deepcopy(spec) for spec in chosen.commitments
            }
            baseline["last_step"] = step
            action = copy.deepcopy(chosen.action)
            allocation = {
                "selection_state_reconstructed": True,
                "selection_mode": selection["mode"],
                "matching_modes": selection["matching_modes"],
                "base_commitments": [s["key"] for s in chosen.commitments],
            }

        trace.append({
            "step": step,
            "phase": "decision",
            "metrics": _metrics(raw_obs),
            "world_signature": _world_signature(raw_obs),
            "world_digest": _digest(_world_signature(raw_obs)),
            "action": copy.deepcopy(action),
            "allocation": allocation,
            "sell_orders": [
                copy.deepcopy(order) for order in action.get("market", [])
                if order and order[0] == "SELL"
            ],
        })
        return action

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    agents = [opponent.agent, opponent.agent]
    agents[SEAT] = self_agent
    env.run(agents)

    gate = next(row for row in trace if row["step"] == GATE_ACTIONS)
    return {
        "label": label,
        "seed": SEED,
        "seat": SEAT,
        "gate_actions": GATE_ACTIONS,
        "gate_metrics": gate["metrics"],
        "trace": trace,
    }


def _compare(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    b_by = {int(r["step"]): r for r in baseline["trace"]}
    v_by = {int(r["step"]): r for r in variant["trace"]}
    steps = sorted(set(b_by) & set(v_by))

    first_world = None
    first_action = None
    for step in steps:
        if first_world is None and b_by[step]["world_digest"] != v_by[step]["world_digest"]:
            first_world = step
        if (
            first_action is None
            and step < GATE_ACTIONS
            and _canonical(b_by[step]["action"]) != _canonical(v_by[step]["action"])
        ):
            first_action = step

    world_diffs = []
    if first_world is not None:
        world_diffs = _diff_paths(
            b_by[first_world]["world_signature"],
            v_by[first_world]["world_signature"],
        )

    b_gate = baseline["gate_metrics"]
    v_gate = variant["gate_metrics"]
    numeric_delta = {}
    for key in (
        "self_cash",
        "opponent_cash",
        "hands",
        "plants",
        "animals",
        "productive_assets",
        "tile_yield_units",
        "carried_total_items",
        "shed_total_items",
    ):
        numeric_delta[key] = v_gate[key] - b_gate[key]

    return {
        "recovery_candidate_steps": [
            int(r["step"]) for r in variant["trace"]
            if r.get("phase") == "decision" and r.get("allocation", {}).get("added_recovery")
        ],
        "recovery_scheduled_steps": [
            int(r["step"]) for r in variant["trace"]
            if r.get("phase") == "decision" and r.get("allocation", {}).get("recovery_scheduled")
        ],
        "intervention_count": sum(
            bool(r.get("allocation", {}).get("added_recovery"))
            for r in variant["trace"] if r.get("phase") == "decision"
        ),
        "scheduled_intervention_count": sum(
            bool(r.get("allocation", {}).get("recovery_scheduled"))
            for r in variant["trace"] if r.get("phase") == "decision"
        ),
        "first_action_divergence_step": first_action,
        "first_world_divergence_step": first_world,
        "first_world_divergence_paths": world_diffs,
        "gate_metric_delta_variant_minus_baseline": numeric_delta,
        "world_gate_pass": (
            first_world is not None and first_world <= GATE_ACTIONS
            and any(r.get("allocation", {}).get("recovery_scheduled")
                    for r in variant["trace"] if r.get("phase") == "decision")
        ),
        "interpretation_boundary": (
            "PASS means only that the one-allocation recovery variant changed the "
            "observed World within 24 actions. It is not a strength or terminal verdict."
        ),
    }


def main() -> None:
    baseline = _run("baseline_strong_first24", variant=False)
    variant = _run("recovery_one_allocation_first24", variant=True)
    comparison = _compare(baseline, variant)

    result = {
        "schema": "terminal-return-world-gate-v0",
        "purpose": (
            "Acceptance test for one design-aligned allocation variant. "
            "No terminal adoption decision is made here."
        ),
        "conditions": {
            "seed": SEED,
            "seat": SEAT,
            "opponent": "Seyamalam pinned v21",
            "official_world_commit": "d7729da06cc1382eb742d6980dc3180aa85caa28",
            "baseline": "Strong Model v0 reimplementation",
            "variant": (
                "ordinary Strong planned commitments + one single-shot recovery "
                "commitment before _service_action(); never carried into the next turn"
            ),
            "no_post_action_replacement": True,
        },
        "comparison": comparison,
        "baseline": baseline,
        "variant": variant,
        "boundary": {
            "not_terminal_strength_test": True,
            "not_adoption_verdict": True,
            "one_seed_one_seat": True,
            "world_state_gate_only": True,
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "world_gate_pass": comparison["world_gate_pass"],
        "intervention_count": comparison["intervention_count"],
        "scheduled_intervention_count": comparison["scheduled_intervention_count"],
        "first_action_divergence_step": comparison["first_action_divergence_step"],
        "first_world_divergence_step": comparison["first_world_divergence_step"],
        "gate_metric_delta": comparison["gate_metric_delta_variant_minus_baseline"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
