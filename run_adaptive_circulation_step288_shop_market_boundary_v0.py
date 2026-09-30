#!/usr/bin/env python3
"""Inspect the step288->289 town/shop to market boundary for seeds 92802003/10,
PASS vs HARVEST only.
"""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
PASS=ROOT/"adaptive_circulation_step250_pass_v0.py"
HARVEST=ROOT/"adaptive_circulation_opportunity_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802003,92802010]

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

def run(seed,path,tag):
    model=load(path,f"{tag}_{seed}"); opp=load(OPP,f"opp_{tag}_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    out={}
    while not env.done:
        o0=shared(env,0); o1=shared(env,1); s=int(o0.get("step",0) or 0)
        a0=plain(model.agent(o0,env.configuration)); a1=plain(opp.agent(o1))
        if s in (287,288,289,290):
            out[s]={
              "town":plain(o0.get("town") or {}),
              "market_inventory":plain((o0.get("market") or {}).get("inventory") or {}),
              "market_prices":plain((o0.get("market") or {}).get("prices") or {}),
              "action":a0,
            }
        if s>=290: break
        env.step([a0,a1])
    return out

def delta(a,b):
    out={}
    for k in sorted(set(a)|set(b)):
        d=(b.get(k,0) or 0)-(a.get(k,0) or 0)
        if d: out[k]=d
    return out

def main():
    result={}
    for seed in SEEDS:
        p=run(seed,PASS,"pass"); h=run(seed,HARVEST,"harvest")
        rec={"pass":p,"harvest":h}
        rec["market_effect_288_to_289"]={
          "pass":delta(p[288]["market_inventory"],p[289]["market_inventory"]),
          "harvest":delta(h[288]["market_inventory"],h[289]["market_inventory"]),
        }
        rec["harvest_minus_pass_at_289"]=delta(p[289]["market_inventory"],h[289]["market_inventory"])
        result[str(seed)]=rec
        print("SEED "+json.dumps({
          "seed":seed,
          "pass_town_288":p[288]["town"],
          "harvest_town_288":h[288]["town"],
          "pass_market_288_289":rec["market_effect_288_to_289"]["pass"],
          "harvest_market_288_289":rec["market_effect_288_to_289"]["harvest"],
          "harvest_minus_pass_at_289":rec["harvest_minus_pass_at_289"],
        },ensure_ascii=False,separators=(",",":")))
    Path("adaptive_circulation_step288_shop_market_boundary_v0_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
