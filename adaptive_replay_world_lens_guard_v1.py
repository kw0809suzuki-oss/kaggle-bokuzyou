"""Adaptive Replay + World Lens Guard v1.

Final submission shape for the current model line.

Principle:
    Replay owns the operating skeleton.
    World Lens observes Current World.
    Guard acts only when a proven realized-effect boundary is violated.

This version deliberately does NOT:
- optimize BUY by current price,
- reroute SELL by current price,
- predict future market/shop paths,
- add fitted thresholds,
- repair every divergence from the source replay.

Active behavior remains the proven step24 HIRE Effect Contract only.

Architecture:
    Frozen Replay Skeleton
        -> World Lens snapshot
        -> Proven Effect Contract lookup
        -> Minimal Current-World feasibility check
        -> Minimal Guard transform
        -> Action

World Lens is observational unless a promoted contract explicitly consumes it.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT/"decem_replay_distilled_157026_v0.py"
CONTRACTS=ROOT/"adaptive_replay_effect_contracts_v0.json"
WORLD_ADAPTER=ROOT/"adaptive_replay_world_adapter_v0.py"

def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

replay=_load(REPLAY,"_world_lens_guard_replay")
world=_load(WORLD_ADAPTER,"_world_lens_guard_adapter")
_contract_doc=json.loads(CONTRACTS.read_text(encoding="utf-8"))
_contracts=[c for c in _contract_doc.get("contracts",[]) if c.get("status")=="active"]

trigger_count=0
last_guard_event=None
last_contract_id=None
last_world_lens=None

def reset_agent():
    global trigger_count,last_guard_event,last_contract_id,last_world_lens
    trigger_count=0
    last_guard_event=None
    last_contract_id=None
    last_world_lens=None
    if hasattr(replay,"reset_agent"):
        replay.reset_agent()

def _world_lens(obs):
    player=int(obs.get("player",0) or 0)
    farm=(obs.get("farms") or [])[player]
    market=obs.get("market") or {}
    town=obs.get("town") or {}
    private=obs.get("private") or {}
    return {
        "step":int(obs.get("step",0) or 0),
        "day":int(obs.get("day",0) or 0),
        "hour":int(obs.get("hour",0) or 0),
        "money":float(farm.get("money",0) or 0),
        "hires_today":int(farm.get("hires_today",0) or 0),
        "market_prices":deepcopy(market.get("prices") or {}),
        "market_inventory":deepcopy(market.get("inventory") or {}),
        "town":deepcopy(town),
        "shed":deepcopy(private.get("shed") or {}),
        "seeds":deepcopy(private.get("seeds") or {}),
    }

def _cost_mult(configuration):
    if configuration is None:
        return 1.0
    if isinstance(configuration,dict):
        return float(configuration.get("farmHandCostMult",1) or 1)
    return float(getattr(configuration,"farmHandCostMult",1) or 1)

def _count_market_op(orders,op):
    return sum(
        1 for x in orders
        if isinstance(x,(list,tuple)) and x and x[0]==op
    )

def _cap_market_op(orders,op,count):
    kept=0
    out=[]
    for x in orders:
        is_target=isinstance(x,(list,tuple)) and x and x[0]==op
        if not is_target:
            out.append(x)
        elif kept<int(count):
            out.append(x)
            kept+=1
    return out

def _apply_contract(action,obs,configuration,contract):
    step=int(obs.get("step",0) or 0)
    scope=contract.get("scope",{})
    if step!=int(scope.get("step",-1)):
        return action,None

    operation=contract.get("operation")
    orders=list(action.get("market",[]) or [])
    requested=_count_market_op(orders,operation)
    protected=int(contract["protected_effect"]["count"])

    if requested<=protected:
        return action,None

    check=contract.get("world_check",{})
    if check.get("adapter")!="hire_feasibility":
        return action,None

    player=int(obs.get("player",0) or 0)
    farm=obs["farms"][player]
    money=float(farm.get("money",0) or 0)
    hires_today=int(farm.get("hires_today",0) or 0)

    feasible=world.count_feasible_hires(
        money=money,
        hires_today=hires_today,
        requested=requested,
        cost_mult=_cost_mult(configuration),
    )
    if feasible<=protected:
        return action,None

    transform=contract.get("transform",{})
    if transform.get("kind")!="cap_market_operation_count":
        return action,None

    out=deepcopy(action)
    out["market"]=_cap_market_op(
        orders,
        transform.get("operation",operation),
        int(transform.get("count",protected)),
    )

    return out,{
        "step":step,
        "money":money,
        "hires_today":hires_today,
        "requested":requested,
        "current_feasible":feasible,
        "protected_effect":protected,
    }

def agent(obs,configuration=None):
    global trigger_count,last_guard_event,last_contract_id,last_world_lens

    last_world_lens=_world_lens(obs)
    action=deepcopy(replay.agent(obs,configuration))

    for contract in _contracts:
        action,event=_apply_contract(action,obs,configuration,contract)
        if event is not None:
            trigger_count+=1
            last_guard_event=event
            last_contract_id=contract.get("id")

    return action
