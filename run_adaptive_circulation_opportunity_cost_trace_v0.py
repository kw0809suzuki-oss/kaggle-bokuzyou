#!/usr/bin/env python3
"""Short opportunity-cost trace for seeds 92802003 and 92802010.

Compare WATER / PASS / HARVEST from the same step250 state.
Stop after first meaningful divergence chain can be observed.
"""

from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
WATER=ROOT/"adaptive_circulation_runtime_v0.py"
PASS=ROOT/"adaptive_circulation_step250_pass_v0.py"
HARVEST=ROOT/"adaptive_circulation_opportunity_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802003,92802010]
END_STEP=280

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

def compact(obs,action):
    f=obs["farms"][0]; p=obs.get("private") or {}; m=obs.get("market") or {}
    invs=p.get("inventories",[]) or []
    carry={}
    for inv in invs:
        if isinstance(inv,dict):
            for k,v in inv.items(): carry[k]=carry.get(k,0)+int(v or 0)
    plants={}
    yield_units={}
    for row in f.get("tiles",[]) or []:
        for t in row:
            if isinstance(t,dict) and t.get("kind")=="PLANT":
                c=str(t.get("crop")); plants[c]=plants.get(c,0)+1
                yield_units[c]=yield_units.get(c,0)+float(t.get("yield_units",0) or 0)
    return {
      "step":int(obs.get("step",0) or 0),
      "day":int(obs.get("day",0) or 0),
      "hour":int(obs.get("hour",0) or 0),
      "action":action,
      "money":float(f.get("money",0) or 0),
      "seeds":plain(p.get("seeds") or {}),
      "shed":plain(p.get("shed") or {}),
      "carry":carry,
      "plants":plants,
      "yield_units":yield_units,
      "market_prices":plain(m.get("prices") or {}),
      "market_inventory":plain(m.get("inventory") or {}),
      "town":plain(obs.get("town") or {}),
    }

def run(seed,path,tag):
    model=load(path,f"{tag}_{seed}")
    opp=load(OPP,f"opp_{tag}_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    trace=[]
    while not env.done:
        o0=shared(env,0); o1=shared(env,1)
        a0=plain(model.agent(o0,env.configuration)); a1=plain(opp.agent(o1))
        step=int(o0.get("step",0) or 0)
        if 248 <= step <= END_STEP:
            trace.append(compact(o0,a0))
        if step>=END_STEP: break
        env.step([a0,a1])
    return trace

def diff(a,b):
    keys=["action","money","seeds","shed","carry","plants","yield_units","market_prices","market_inventory","town"]
    out={}
    for k in keys:
        if a.get(k)!=b.get(k): out[k]={"a":a.get(k),"b":b.get(k)}
    return out

def first_diff(ta,tb, *, start_step=251, include_action=True):
    by={r["step"]:r for r in tb}
    for a in ta:
        if a["step"] < start_step:
            continue
        b=by.get(a["step"])
        if b is None:
            continue
        d=diff(a,b)
        if not include_action:
            d.pop("action", None)
        if d:
            return {"step":a["step"],"diff":d}
    return None

def first_resource_diff(ta,tb):
    return first_diff(ta,tb,start_step=251,include_action=False)

def main():
    out={"schema":"adaptive-circulation-opportunity-cost-trace-v0","end_step":END_STEP,"seeds":{}}
    for seed in SEEDS:
        w=run(seed,WATER,"water"); p=run(seed,PASS,"pass"); h=run(seed,HARVEST,"harvest")
        rec={
          "water_vs_pass_first_post_action_diff":first_diff(w,p,start_step=251,include_action=True),
          "water_vs_harvest_first_post_action_diff":first_diff(w,h,start_step=251,include_action=True),
          "pass_vs_harvest_first_post_action_diff":first_diff(p,h,start_step=251,include_action=True),
          "water_vs_pass_first_resource_diff":first_resource_diff(w,p),
          "water_vs_harvest_first_resource_diff":first_resource_diff(w,h),
          "pass_vs_harvest_first_resource_diff":first_resource_diff(p,h),
          "water":w,"pass":p,"harvest":h,
        }
        out["seeds"][str(seed)]=rec
        print("SEED "+json.dumps({
          "seed":seed,
          "water_vs_pass_first_post_action_diff":rec["water_vs_pass_first_post_action_diff"],
          "water_vs_harvest_first_post_action_diff":rec["water_vs_harvest_first_post_action_diff"],
          "pass_vs_harvest_first_post_action_diff":rec["pass_vs_harvest_first_post_action_diff"],
          "water_vs_pass_first_resource_diff":rec["water_vs_pass_first_resource_diff"],
          "water_vs_harvest_first_resource_diff":rec["water_vs_harvest_first_resource_diff"],
          "pass_vs_harvest_first_resource_diff":rec["pass_vs_harvest_first_resource_diff"],
        },ensure_ascii=False,separators=(",",":")))
    Path("adaptive_circulation_opportunity_cost_trace_v0_result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
