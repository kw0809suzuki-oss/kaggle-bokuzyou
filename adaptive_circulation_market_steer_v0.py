"""Market Steer v0.

At the observed shop->market boundary, keep Replay structure but steer the
existing BUY_SEED operation toward the crop with the highest current
market-price / base-price ratio.

Minimal intervention:
- one-shot only
- step 289 only
- preserve BUY_SEED quantity
- do not increase seed unit cost versus the original crop
- all non-market actions unchanged
- no fitted thresholds
"""

from __future__ import annotations
from copy import deepcopy

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS, MARKET_PARAMS
import adaptive_circulation_runtime_v0 as body

trigger_count = 0
last_event = None

def reset_agent():
    global trigger_count, last_event
    trigger_count = 0
    last_event = None
    if hasattr(body, "reset_agent"):
        body.reset_agent()

def _best_crop(obs, original_crop):
    prices=((obs.get("market") or {}).get("prices") or {})
    if original_crop not in CROPS:
        return original_crop, {}
    original_cost=float(CROPS[original_crop]["seed"])
    candidates=[]
    ratios={}
    for crop in CROPS:
        if crop not in prices:
            continue
        if float(CROPS[crop]["seed"]) > original_cost:
            continue
        base=float(MARKET_PARAMS[crop]["base"])
        ratio=float(prices[crop])/base if base else 0.0
        ratios[crop]=ratio
        candidates.append((ratio,crop))
    if not candidates:
        return original_crop, ratios
    candidates.sort(key=lambda x:(-x[0],x[1]))
    return candidates[0][1], ratios

def agent(obs, configuration=None):
    global trigger_count, last_event
    action=deepcopy(body.agent(obs, configuration))
    if trigger_count>0 or int(obs.get("step",0) or 0)!=289:
        return action

    market=list(action.get("market",[]) or [])
    for i,m in enumerate(market):
        if not (isinstance(m,(list,tuple)) and len(m)>=3 and str(m[0])=="BUY_SEED"):
            continue
        original=str(m[1]); qty=int(m[2])
        chosen,ratios=_best_crop(obs,original)
        if chosen==original:
            trigger_count=1
            last_event={
                "step":289,"original":list(m),"replacement":list(m),
                "chosen_crop":chosen,"ratios":ratios,"changed":False,
            }
            return action
        out=deepcopy(action)
        out_market=list(out.get("market",[]) or [])
        out_market[i]=["BUY_SEED",chosen,qty]
        out["market"]=out_market
        trigger_count=1
        last_event={
            "step":289,"original":list(m),"replacement":["BUY_SEED",chosen,qty],
            "chosen_crop":chosen,"ratios":ratios,"changed":True,
        }
        return out

    return action
