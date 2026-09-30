"""Adaptive Replay v0 — Contract Runtime form.

Goal of this file:
    Preserve the already-proven behavior of adaptive_replay_v0.py while
    moving the step24 patch into an explicit Effect Contract runtime.

This is not a stronger model.
It must emit the same actions and terminal results as adaptive_replay_v0.py
on the fixed10 equivalence gate.

Architecture:
    Frozen Replay Skeleton
      -> Active Effect Contract lookup
      -> Minimal Current-World check
      -> Proven minimal transform
      -> Action

Only one contract is active in v0: step24 HIRE realized-effect +1.
"""
from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "decem_replay_distilled_157026_v0.py"
CONTRACTS_PATH = ROOT / "adaptive_replay_effect_contracts_v0.json"
WORLD_ADAPTER = ROOT / "adaptive_replay_world_adapter_v0.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


replay = _load_module(REPLAY, "_adaptive_replay_contract_runtime_body")
world = _load_module(WORLD_ADAPTER, "_adaptive_replay_contract_world_adapter")
_contract_doc = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
_contracts = [
    c for c in _contract_doc.get("contracts", [])
    if c.get("status") == "active"
]

trigger_count = 0
last_guard_event = None
last_contract_id = None


def reset_agent():
    global trigger_count, last_guard_event, last_contract_id
    trigger_count = 0
    last_guard_event = None
    last_contract_id = None
    if hasattr(replay, "reset_agent"):
        replay.reset_agent()


def _cost_mult(configuration) -> float:
    if configuration is None:
        return 1.0
    if isinstance(configuration, dict):
        return float(configuration.get("farmHandCostMult", 1) or 1)
    return float(getattr(configuration, "farmHandCostMult", 1) or 1)


def _count_market_operation(market_orders, operation: str) -> int:
    return sum(
        1
        for order in market_orders
        if isinstance(order, (list, tuple))
        and len(order) >= 1
        and order[0] == operation
    )


def _cap_market_operation_count(market_orders, operation: str, count: int):
    kept = 0
    out = []
    for order in market_orders:
        is_target = (
            isinstance(order, (list, tuple))
            and len(order) >= 1
            and order[0] == operation
        )
        if not is_target:
            out.append(order)
            continue
        if kept < int(count):
            out.append(order)
            kept += 1
    return out


def _apply_contract(action, obs, configuration, contract):
    scope = contract.get("scope", {})
    step = int(obs.get("step", 0) or 0)
    if step != int(scope.get("step", -1)):
        return action, None

    operation = contract.get("operation")
    market_orders = list(action.get("market", []) or [])
    requested = _count_market_operation(market_orders, operation)

    source_requested = int(contract["source_requested"]["count"])
    protected_count = int(contract["protected_effect"]["count"])

    # Preserve old-v0 behavior: if replay no longer requests more than the
    # protected realized count, do nothing.
    if requested <= protected_count:
        return action, None

    # Contract applicability is descriptive; the replay source count remains
    # recorded but does not add a new rejection rule beyond old-v0 behavior.
    _ = source_requested

    check = contract.get("world_check", {})
    if check.get("adapter") != "hire_feasibility":
        return action, None

    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    money = float(farm.get("money", 0) or 0)
    hires_today = int(farm.get("hires_today", 0) or 0)

    feasible = world.count_feasible_hires(
        money=money,
        hires_today=hires_today,
        requested=requested,
        cost_mult=_cost_mult(configuration),
    )

    if feasible <= protected_count:
        return action, None

    transform = contract.get("transform", {})
    if transform.get("kind") != "cap_market_operation_count":
        return action, None

    transformed = deepcopy(action)
    transformed["market"] = _cap_market_operation_count(
        market_orders,
        transform.get("operation", operation),
        int(transform.get("count", protected_count)),
    )

    event = {
        "step": step,
        "money": money,
        "hires_today": hires_today,
        "requested_hires": requested,
        "current_feasible_hires": feasible,
        "source_realized_hires": protected_count,
    }
    return transformed, event


def agent(obs, configuration=None):
    global trigger_count, last_guard_event, last_contract_id

    action = deepcopy(replay.agent(obs, configuration))

    for contract in _contracts:
        action, event = _apply_contract(action, obs, configuration, contract)
        if event is not None:
            trigger_count += 1
            last_guard_event = event
            last_contract_id = contract.get("id")

    return action
