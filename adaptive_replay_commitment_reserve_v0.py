"""Adaptive Replay Commitment Reserve v0.

Exploratory second model. The proven Adaptive Replay Runtime remains untouched.

Idea:
    Replay is the default policy, but before the known step106 COW purchase,
    preserve enough projected Cash to satisfy the next three frozen Replay
    STRAWBERRY seed purchases (steps107,108,110; 100 each).

This is NOT a promoted Effect Contract. It is a separate experimental policy.

At step106:
  - start from Adaptive Replay Contract Runtime v0 action;
  - reconstruct exact same-step FERTILIZER SELL revenue from current inventory;
  - project Cash after BUY_ANIMAL COW (cost 400);
  - if projected Cash < 300 commitment reserve, remove only BUY_ANIMAL COW.

No other action is changed.
"""
from __future__ import annotations
import importlib.util
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"

def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

base=_load(BASE,"_adaptive_replay_commitment_reserve_base")

reserve_trigger_count=0
last_reserve_event=None

def reset_agent():
    global reserve_trigger_count,last_reserve_event
    reserve_trigger_count=0
    last_reserve_event=None
    if hasattr(base,"reset_agent"): base.reset_agent()

def _fertilizer_price(inventory:int)->int:
    # Pinned official engine d7729d...:
    # base=100, I0=10000, T=200, linear target=0.40 both sides.
    base=100.0
    delta=abs(int(inventory)-10000)
    move=0.2*delta
    price=base+move if inventory<10000 else base-move
    return max(1,int(round(price)))

def _sell_revenue(obs, action)->int:
    inv=int(obs["market"]["inventory"]["FERTILIZER"])
    shed=int(obs["private"]["shed"].get("FERTILIZER",0) or 0)
    revenue=0
    for order in action.get("market",[]) or []:
        if not (isinstance(order,(list,tuple)) and len(order)>=3):
            continue
        if order[0]!="SELL" or order[1]!="FERTILIZER":
            continue
        qty=min(int(order[2]),shed)
        for i in range(qty):
            revenue += _fertilizer_price(inv+i)
        shed -= qty
        inv += qty
    return revenue

def _remove_one_cow_buy(action):
    out=deepcopy(action)
    removed=False
    market=[]
    for order in out.get("market",[]) or []:
        is_cow=(isinstance(order,(list,tuple)) and len(order)>=3
                and order[0]=="BUY_ANIMAL" and order[1]=="COW"
                and int(order[2])>=1)
        if is_cow and not removed:
            # v0 source order requests exactly one COW.
            removed=True
            continue
        market.append(order)
    out["market"]=market
    return out,removed

def agent(obs, configuration=None):
    global reserve_trigger_count,last_reserve_event
    action=deepcopy(base.agent(obs,configuration))
    step=int(obs.get("step",0) or 0)
    if step!=106:
        return action

    has_cow=any(
        isinstance(o,(list,tuple)) and len(o)>=3
        and o[0]=="BUY_ANIMAL" and o[1]=="COW" and int(o[2])>=1
        for o in (action.get("market",[]) or [])
    )
    if not has_cow:
        return action

    player=int(obs.get("player",0) or 0)
    cash=float(obs["farms"][player].get("money",0) or 0)
    sell_revenue=float(_sell_revenue(obs,action))
    cow_cost=400.0
    commitment_reserve=300.0
    projected_after_cow=cash+sell_revenue-cow_cost

    if projected_after_cow >= commitment_reserve:
        return action

    transformed,removed=_remove_one_cow_buy(action)
    if not removed:
        return action

    reserve_trigger_count += 1
    last_reserve_event={
        "step":step,
        "pre_cash":cash,
        "same_step_fertilizer_sell_revenue":sell_revenue,
        "cow_cost":cow_cost,
        "projected_after_cow":projected_after_cow,
        "commitment_reserve":commitment_reserve,
        "future_replay_commitments":[
            {"step":107,"operation":"BUY_SEED","item":"STRAWBERRY","cost":100},
            {"step":108,"operation":"BUY_SEED","item":"STRAWBERRY","cost":100},
            {"step":110,"operation":"BUY_SEED","item":"STRAWBERRY","cost":100},
        ],
    }
    return transformed
