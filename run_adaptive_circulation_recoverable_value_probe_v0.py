#!/usr/bin/env python3
"""Observe price x recoverable WHEAT value at step250.

Predeclared surfaces:
- field mature WHEAT yield currently harvestable
- carried WHEAT across farmer + hands
- shed WHEAT
- each multiplied by current visible WHEAT price
- simple sums and ratios only

No classifier fitting. No compound rescue rule.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
HARVEST_MINUS_PASS = {
  92802001:-7657,92802002:-104,92802003:-9578,92802004:-50,92802005:-794,
  92802006:895,92802007:190,92802008:-53,92802009:20312,92802010:21568,
}

def plain(x):
    if isinstance(x, dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])

def harvestable_wheat(farm,day):
    total=0
    plants=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if not isinstance(tile,dict) or tile.get("kind")!="PLANT" or tile.get("crop")!="WHEAT":
                continue
            y=float(tile.get("yield_units",0) or 0)
            planted=int(tile.get("planted_day",day) or day)
            if y>0 and int(day)-planted>=int(CROPS["WHEAT"]["first_yield_day"]):
                total+=y; plants+=1
    return plants,total

def carried_wheat(private):
    total=0
    for inv in private.get("inventories",[]) or []:
        if isinstance(inv,dict): total += float(inv.get("WHEAT",0) or 0)
    return total

def capture(seed):
    model=load(MODEL,f"rv_model_{seed}")
    opp=load(OPPONENT,f"rv_opp_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    while not env.done:
        o0=shared(env,0)
        if int(o0.get("step",0) or 0)==250:
            farm=o0["farms"][0]
            private=o0.get("private") or {}
            price=float((o0.get("market") or {}).get("prices",{}).get("WHEAT",0) or 0)
            plants,field_units=harvestable_wheat(farm,int(o0.get("day",0) or 0))
            carry=carried_wheat(private)
            shed=float((private.get("shed") or {}).get("WHEAT",0) or 0)
            money=float(farm.get("money",0) or 0)
            f={
              "price":price,
              "harvestable_plants":plants,
              "field_units":field_units,
              "carry_units":carry,
              "shed_units":shed,
              "field_value":field_units*price,
              "carry_value":carry*price,
              "shed_value":shed*price,
              "ready_units":carry+shed,
              "ready_value":(carry+shed)*price,
              "field_plus_ready_units":field_units+carry+shed,
              "field_plus_ready_value":(field_units+carry+shed)*price,
              "money":money,
              "ready_value_over_money":((carry+shed)*price/money) if money>0 else None,
              "field_plus_ready_over_money":(((field_units+carry+shed)*price)/money) if money>0 else None,
            }
            return f
        o1=shared(env,1)
        env.step([plain(model.agent(o0,env.configuration)), plain(opp.agent(o1))])
    raise RuntimeError(seed)

def perfect_threshold(rows,key):
    vals=[]
    for r in rows:
        v=r["features"].get(key)
        if v is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):
            return []
        vals.append((float(v),r["label"],r["seed"]))
    uniq=sorted(set(v for v,_,_ in vals))
    ans=[]
    for a,b in zip(uniq,uniq[1:]):
        cut=(a+b)/2
        for orient in ("GT_HARVEST","LE_HARVEST"):
            ok=True
            for v,lab,_ in vals:
                pred="HARVEST_GT_PASS" if ((v>cut) if orient=="GT_HARVEST" else (v<=cut)) else "PASS_GT_HARVEST"
                if pred!=lab: ok=False; break
            if ok: ans.append({"feature":key,"threshold":cut,"orientation":orient})
    return ans

def main():
    rows=[]
    for seed in SEEDS:
        f=capture(seed)
        lab="HARVEST_GT_PASS" if HARVEST_MINUS_PASS[seed]>0 else "PASS_GT_HARVEST"
        row={"seed":seed,"label":lab,"delta_harvest_minus_pass":HARVEST_MINUS_PASS[seed],"features":f}
        rows.append(row)
        print("CASE "+json.dumps(row,separators=(",",":")))

    keys=sorted(rows[0]["features"])
    pos=[r for r in rows if r["label"]=="HARVEST_GT_PASS"]
    neg=[r for r in rows if r["label"]=="PASS_GT_HARVEST"]
    disjoint=[]; thresholds=[]
    for k in keys:
        pv={r["features"][k] for r in pos}; nv={r["features"][k] for r in neg}
        if None not in pv and None not in nv and pv.isdisjoint(nv):
            disjoint.append({"feature":k,"harvest_values":[r["features"][k] for r in pos],"pass_values":[r["features"][k] for r in neg]})
        thresholds.extend(perfect_threshold(rows,k))

    summary={"n":10,"features_checked":keys,"disjoint_feature_count":len(disjoint),"disjoint_features":disjoint,
             "perfect_threshold_count":len(thresholds),"perfect_thresholds":thresholds}
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_circulation_recoverable_value_probe_v0_result.json").write_text(
      json.dumps({"schema":"adaptive-circulation-recoverable-value-probe-v0","summary":summary,"cases":rows,
      "boundary":["No compound classifier is fitted.","A fixed10 separator is observational, not causal.","No rescue condition if simple surfaces fail."]},indent=2)+"\n",
      encoding="utf-8"
    )

if __name__=="__main__": main()
