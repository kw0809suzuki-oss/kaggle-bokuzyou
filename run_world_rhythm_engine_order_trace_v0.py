#!/usr/bin/env python3
"""World Rhythm — Engine Order Trace Probe v0.

Purpose:
  Explain the already-confirmed local order effect at step217 without looking at terminal.

Known order-only comparison:
  AB = HIRE -> BUY_PRODUCT WHEAT 19
  BA = BUY_PRODUCT WHEAT 19 -> HIRE

This probe instruments the pinned Official Kaggriculture engine itself for
three representative seat0 worlds:
  low cash  : seed 94002001
  mid cash  : seed 94002003
  high cash : seed 94002008

We record the exact call sequence for:
  _do_hire
  _commit_unit
  _refresh_prices

This distinguishes:
  - pure Cash feasibility effects
  - shared-market inventory / repricing effects

No terminal. No policy change. No Phase semantics.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

TARGET_STEP=217
SEEDS=[94002001,94002003,94002008]
AB=[["HIRE"],["BUY_PRODUCT","WHEAT",19]]
BA=[["BUY_PRODUCT","WHEAT",19],["HIRE"]]


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod


def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]


def snap(obs,seat):
    farm=obs["farms"][seat]
    priv=obs["private"]
    market=obs["market"]
    return {
        "cash":float(farm.get("money",0) or 0),
        "hands":len(farm.get("hands",[]) or []),
        "hires_today":int(farm.get("hires_today",0) or 0),
        "shed_wheat":int((priv.get("shed",{}) or {}).get("WHEAT",0) or 0),
        "market_wheat_inventory":int((market.get("inventory",{}) or {}).get("WHEAT",0) or 0),
        "market_wheat_price":int((market.get("prices",{}) or {}).get("WHEAT",0) or 0),
    }


def run(seed,mode):
    seat=0
    model=load(BASE,f"engine_trace_model_{mode}_{seed}_{os.getpid()}")
    opp=load(OPP,f"engine_trace_opp_{mode}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0=plain(shared(env,0));o1=plain(shared(env,1))
        step=int(o0.get("step",0) or 0)
        sa=plain(model.agent(o0));oa=plain(opp.agent(o1))

        if step != TARGET_STEP:
            env.step([sa,oa])
            continue

        if sa.get("market",[]) != AB:
            raise RuntimeError(f"unexpected subject market seed={seed}: {sa.get('market')}")
        if mode=="ba":
            sa=deepcopy(sa);sa["market"]=deepcopy(BA)

        pre=snap(o0,seat)
        events=[]

        orig_hire=kg._do_hire
        orig_commit=kg._commit_unit
        orig_refresh=kg._refresh_prices

        def wrap_hire(farm,private,board_size,mult=kg.FARM_HAND_COST_MULT):
            before={
                "money":farm["money"],
                "hires_today":farm["hires_today"],
                "hands":len(farm["hands"]),
            }
            result=orig_hire(farm,private,board_size,mult)
            after={
                "money":farm["money"],
                "hires_today":farm["hires_today"],
                "hands":len(farm["hands"]),
            }
            events.append({"kind":"HIRE","before":before,"after":after})
            return result

        def wrap_commit(op,item,price,farm,private,market,shed_capacity=100):
            before={
                "op":op,
                "item":item,
                "price":price,
                "money":farm["money"],
                "shed_wheat":private["shed"].get("WHEAT",0),
                "market_wheat_inventory":market["inventory"].get("WHEAT"),
                "market_wheat_price":market["prices"].get("WHEAT"),
            }
            ok=orig_commit(op,item,price,farm,private,market,shed_capacity)
            after={
                "ok":ok,
                "money":farm["money"],
                "shed_wheat":private["shed"].get("WHEAT",0),
                "market_wheat_inventory":market["inventory"].get("WHEAT"),
                "market_wheat_price":market["prices"].get("WHEAT"),
            }
            events.append({"kind":"COMMIT_UNIT","before":before,"after":after})
            return ok

        def wrap_refresh(market):
            before={
                "market_wheat_inventory":market["inventory"].get("WHEAT"),
                "market_wheat_price":market["prices"].get("WHEAT"),
            }
            result=orig_refresh(market)
            after={
                "market_wheat_inventory":market["inventory"].get("WHEAT"),
                "market_wheat_price":market["prices"].get("WHEAT"),
            }
            events.append({"kind":"REFRESH_PRICES","before":before,"after":after})
            return result

        kg._do_hire=wrap_hire
        kg._commit_unit=wrap_commit
        kg._refresh_prices=wrap_refresh
        try:
            env.step([sa,oa])
        finally:
            kg._do_hire=orig_hire
            kg._commit_unit=orig_commit
            kg._refresh_prices=orig_refresh

        post=plain(shared(env,seat))
        return {
            "seed":seed,
            "mode":mode,
            "subject_market":deepcopy(sa["market"]),
            "opponent_market":deepcopy(oa.get("market",[]) or []),
            "pre":pre,
            "events":events,
            "post":snap(post,seat),
        }

    raise RuntimeError("target step not reached")


def main():
    rows=[]
    for seed in SEEDS:
        ab=run(seed,"ab")
        ba=run(seed,"ba")
        row={"seed":seed,"ab":ab,"ba":ba}
        rows.append(row)
        print("ENGINE_ORDER_TRACE_CASE "+json.dumps(row,separators=(",",":")))

    out={
        "schema":"world-rhythm-engine-order-trace-v0",
        "question":"Inside step217, which engine-side mutation first makes HIRE->BUY_PRODUCT differ from BUY_PRODUCT->HIRE?",
        "cases":rows,
        "boundary":{
            "representative_worlds_only":True,
            "seeds":SEEDS,
            "seat":0,
            "single_transition_only":True,
            "terminal_not_used":True,
            "official_engine_instrumented":True,
            "no_policy_change":True,
            "no_phase_semantics":True
        }
    }
    Path("world_rhythm_engine_order_trace_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )


if __name__=="__main__": main()
