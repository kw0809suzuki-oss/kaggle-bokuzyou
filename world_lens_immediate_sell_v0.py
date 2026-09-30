"""World Lens — Immediate Sell v0.

Purpose:
Add one narrow Current-World lens at SELL contacts without changing Replay's
overall operating skeleton.

Rule:
- Replay decides WHEN and HOW MANY sell-only units to sell.
- WHEAT and FERTILIZER are never rerouted because they have farm-side uses.
- For sell-only products already present in the shed
  (CARROT, TOMATO, STRAWBERRY, MELON, EGG, MILK, WOOL),
  redirect the requested sell-only quantity toward the currently highest
  displayed cash price, consuming only actual shed stock.
- No future-price forecast.
- No fitted threshold.
- No extra sell quantity beyond Replay's requested sell-only quantity.

This is an immediate-realization lens, not a production-choice policy.
"""

from __future__ import annotations

from copy import deepcopy

import adaptive_replay_contract_runtime_v0 as body

SELL_ONLY = ("CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL")

trigger_count = 0
rerouted_units = 0
last_event = None


def reset_agent():
    global trigger_count, rerouted_units, last_event
    trigger_count = 0
    rerouted_units = 0
    last_event = None
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def _is_sell(order):
    return isinstance(order,(list,tuple)) and len(order)>=3 and str(order[0])=="SELL"


def _sell_only_requested(market_orders):
    total=0
    for o in market_orders:
        if _is_sell(o) and str(o[1]) in SELL_ONLY:
            try:
                total += max(0,int(o[2]))
            except Exception:
                pass
    return total


def _allocation(obs, requested):
    private=obs.get("private") or {}
    shed=private.get("shed") or {}
    prices=((obs.get("market") or {}).get("prices") or {})

    candidates=[]
    for item in SELL_ONLY:
        n=int(shed.get(item,0) or 0)
        if n<=0 or item not in prices:
            continue
        candidates.append((float(prices[item]), item, n))

    candidates.sort(key=lambda x:(-x[0],x[1]))
    remain=int(requested)
    alloc=[]
    for price,item,stock in candidates:
        if remain<=0:
            break
        take=min(remain,stock)
        if take>0:
            alloc.append(["SELL",item,take])
            remain-=take
    return alloc


def agent(obs, configuration=None):
    global trigger_count, rerouted_units, last_event

    action=deepcopy(body.agent(obs,configuration))
    market=list(action.get("market",[]) or [])
    requested=_sell_only_requested(market)
    if requested<=0:
        return action

    replacement=_allocation(obs,requested)
    if not replacement:
        return action

    # Preserve all non-target market operations and the relative insertion point
    # of the first sell-only order. Replace the whole sell-only group once.
    out=[]
    inserted=False
    original_sell_only=[]
    for o in market:
        if _is_sell(o) and str(o[1]) in SELL_ONLY:
            original_sell_only.append(deepcopy(o))
            if not inserted:
                out.extend(deepcopy(replacement))
                inserted=True
            continue
        out.append(deepcopy(o))

    if original_sell_only==replacement:
        return action

    action["market"]=out
    trigger_count += 1
    rerouted_units += sum(int(o[2]) for o in replacement)
    last_event={
        "step":int(obs.get("step",0) or 0),
        "original_sell_only":original_sell_only,
        "replacement_sell_only":deepcopy(replacement),
        "prices":{k:float(v) for k,v in (((obs.get("market") or {}).get("prices") or {}).items()) if k in SELL_ONLY},
        "shed":{k:int(v or 0) for k,v in ((obs.get("private") or {}).get("shed") or {}).items() if k in SELL_ONLY},
    }
    return action
