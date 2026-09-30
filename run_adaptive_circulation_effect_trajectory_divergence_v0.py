#!/usr/bin/env python3
"""Find first step where HARVEST-vs-PASS effect trajectory differs between
seed92802003 (negative) and seed92802010 (positive).

Compare DELTAS, not raw state levels.
"""

from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
PASS=ROOT/"adaptive_circulation_step250_pass_v0.py"
HARVEST=ROOT/"adaptive_circulation_opportunity_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802003,92802010]
END_STEP=340

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def compact(obs,action):
    f=obs["farms"][0]; p=obs.get("private") or {}; m=obs.get("market") or {}
    carry={}
    for inv in p.get("inventories",[]) or []:
        if isinstance(inv,dict):
            for k,v in inv.items(): carry[k]=carry.get(k,0)+int(v or 0)
    plants={}; yields={}
    for row in f.get("tiles",[]) or []:
        for t in row:
            if isinstance(t,dict) and t.get("kind")=="PLANT":
                c=str(t.get("crop")); plants[c]=plants.get(c,0)+1
                yields[c]=yields.get(c,0)+float(t.get("yield_units",0) or 0)
    return {"step":int(obs.get("step",0) or 0),"action":action,"money":float(f.get("money",0) or 0),
            "seeds":plain(p.get("seeds") or {}),"shed":plain(p.get("shed") or {}),"carry":carry,
            "plants":plants,"yield_units":yields,"market_prices":plain(m.get("prices") or {}),
            "market_inventory":plain(m.get("inventory") or {}),"town":plain(obs.get("town") or {})}

def run(seed,path,tag):
    model=load(path,f"{tag}_{seed}"); opp=load(OPP,f"opp_{tag}_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    out={}
    while not env.done:
        o0=shared(env,0); o1=shared(env,1); step=int(o0.get("step",0) or 0)
        a0=plain(model.agent(o0,env.configuration)); a1=plain(opp.agent(o1))
        if 250 <= step <= END_STEP: out[step]=compact(o0,a0)
        if step>=END_STEP: break
        env.step([a0,a1])
    return out

def dict_delta(a,b):
    keys=set(a)|set(b); out={}
    for k in sorted(keys):
        av=a.get(k,0) or 0; bv=b.get(k,0) or 0
        if isinstance(av,(int,float)) and isinstance(bv,(int,float)):
            d=bv-av
            if d!=0: out[k]=d
        elif av!=bv:
            out[k]={"pass":av,"harvest":bv}
    return out

def action_signature(pa,ha):
    def units(bundle):
        xs=[bundle.get("farmer",["PASS"])] + list(bundle.get("hands",[]) or [])
        return xs
    pu,hu=units(pa),units(ha)
    dif=[]
    n=max(len(pu),len(hu))
    for i in range(n):
        a=pu[i] if i<len(pu) else ["PASS"]; b=hu[i] if i<len(hu) else ["PASS"]
        if a!=b: dif.append({"actor":i,"pass":a,"harvest":b})
    pm=pa.get("market",[]) or []; hm=ha.get("market",[]) or []
    if pm!=hm: dif.append({"market":{"pass":pm,"harvest":hm}})
    return dif

def effect(passrow,harvestrow):
    return {
      "action_diff":action_signature(passrow["action"],harvestrow["action"]),
      "money_delta":harvestrow["money"]-passrow["money"],
      "seeds_delta":dict_delta(passrow["seeds"],harvestrow["seeds"]),
      "shed_delta":dict_delta(passrow["shed"],harvestrow["shed"]),
      "carry_delta":dict_delta(passrow["carry"],harvestrow["carry"]),
      "plants_delta":dict_delta(passrow["plants"],harvestrow["plants"]),
      "yield_delta":dict_delta(passrow["yield_units"],harvestrow["yield_units"]),
      "market_price_delta":dict_delta(passrow["market_prices"],harvestrow["market_prices"]),
      "market_inventory_delta":dict_delta(passrow["market_inventory"],harvestrow["market_inventory"]),
      "town_equal": passrow["town"]==harvestrow["town"],
    }

def main():
    effects={}
    for seed in SEEDS:
        p=run(seed,PASS,"pass"); h=run(seed,HARVEST,"harvest")
        effects[seed]={s:effect(p[s],h[s]) for s in sorted(set(p)&set(h)) if s>=251}
    first=None
    common=sorted(set(effects[SEEDS[0]])&set(effects[SEEDS[1]]))
    for s in common:
        if effects[SEEDS[0]][s] != effects[SEEDS[1]][s]:
            first=s; break
    result={"schema":"adaptive-circulation-effect-trajectory-divergence-v0","first_effect_divergence_step":first}
    if first is not None:
        result["seed92802003"]=effects[92802003][first]
        result["seed92802010"]=effects[92802010][first]
        # add prior step to prove they were still same immediately before
        if first-1 in effects[92802003] and first-1 in effects[92802010]:
            result["prior_step"]=first-1
            result["prior_equal"]=effects[92802003][first-1]==effects[92802010][first-1]
            result["prior_effect"]=effects[92802003][first-1]
    print("RESULT "+json.dumps(result,ensure_ascii=False,separators=(",",":")))
    Path("adaptive_circulation_effect_trajectory_divergence_v0_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
