#!/usr/bin/env python3
"""Observe WHEAT return-stage composition at step250.

Stages:
- FIELD: harvestable WHEAT still on plants
- CARRY: WHEAT already carried by farmer/hands
- SHED: WHEAT already in shed

Features are simple shares / stage gaps only.
No fitted compound classifier.
"""

from __future__ import annotations

import importlib.util, json, math, sys
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_circulation_runtime_v0.py"
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
HMP={92802001:-7657,92802002:-104,92802003:-9578,92802004:-50,92802005:-794,92802006:895,92802007:190,92802008:-53,92802009:20312,92802010:21568}

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
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

def field_wheat(farm,day):
    total=0.0
    for row in farm.get("tiles",[]) or []:
        for t in row:
            if not isinstance(t,dict) or t.get("kind")!="PLANT" or t.get("crop")!="WHEAT": continue
            y=float(t.get("yield_units",0) or 0)
            planted=int(t.get("planted_day",day) or day)
            if y>0 and int(day)-planted>=int(CROPS["WHEAT"]["first_yield_day"]):
                total+=y
    return total

def carry_wheat(private):
    return sum(float((inv or {}).get("WHEAT",0) or 0) for inv in (private.get("inventories",[]) or []) if isinstance(inv,dict))

def capture(seed):
    model=load(MODEL,f"mix_model_{seed}")
    opp=load(OPPONENT,f"mix_opp_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    while not env.done:
        o0=shared(env,0)
        if int(o0.get("step",0) or 0)==250:
            farm=o0["farms"][0]; private=o0.get("private") or {}
            field=field_wheat(farm,int(o0.get("day",0) or 0))
            carry=carry_wheat(private)
            shed=float((private.get("shed") or {}).get("WHEAT",0) or 0)
            total=field+carry+shed
            def share(v): return v/total if total>0 else 0.0
            return {
              "field_units":field,"carry_units":carry,"shed_units":shed,"total_units":total,
              "field_share":share(field),"carry_share":share(carry),"shed_share":share(shed),
              "ready_share":share(carry+shed),
              "shed_minus_field":shed-field,
              "carry_minus_field":carry-field,
              "shed_minus_carry":shed-carry,
              "ready_minus_field":carry+shed-field,
            }
        o1=shared(env,1)
        env.step([plain(model.agent(o0,env.configuration)),plain(opp.agent(o1))])
    raise RuntimeError(seed)

def perfect_threshold(rows,key):
    vals=[]
    for r in rows:
        v=r["features"].get(key)
        if v is None or isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)): return []
        vals.append((float(v),r["label"],r["seed"]))
    uniq=sorted(set(v for v,_,_ in vals)); ans=[]
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
        f=capture(seed); lab="HARVEST_GT_PASS" if HMP[seed]>0 else "PASS_GT_HARVEST"
        row={"seed":seed,"label":lab,"delta_harvest_minus_pass":HMP[seed],"features":f}
        rows.append(row); print("CASE "+json.dumps(row,separators=(",",":")))
    keys=sorted(rows[0]["features"]); pos=[r for r in rows if r["label"]=="HARVEST_GT_PASS"]; neg=[r for r in rows if r["label"]=="PASS_GT_HARVEST"]
    disjoint=[]; thresholds=[]
    for k in keys:
        pv={r["features"][k] for r in pos}; nv={r["features"][k] for r in neg}
        if pv.isdisjoint(nv):
            disjoint.append({"feature":k,"harvest_values":[r["features"][k] for r in pos],"pass_values":[r["features"][k] for r in neg]})
        thresholds.extend(perfect_threshold(rows,k))
    summary={"n":10,"features_checked":keys,"disjoint_feature_count":len(disjoint),"disjoint_features":disjoint,"perfect_threshold_count":len(thresholds),"perfect_thresholds":thresholds}
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_circulation_return_stage_mix_v0_result.json").write_text(json.dumps({
      "schema":"adaptive-circulation-return-stage-mix-v0","summary":summary,"cases":rows,
      "boundary":["No compound classifier fitted.","A fixed10 separator is observational, not causal.","No rescue if simple stage composition fails."]
    },indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
