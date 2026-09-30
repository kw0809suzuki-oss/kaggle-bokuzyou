#!/usr/bin/env python3
"""One-transition confirmation at step250 for extreme cases.

Compare baseline WATER vs opportunity HARVEST from the exact same pre-state.
No terminal inference; confirm immediate realized effect only.
"""

from __future__ import annotations

import importlib.util, json, sys
from copy import deepcopy
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_circulation_runtime_v0.py"
CAND=ROOT/"adaptive_circulation_opportunity_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802003,92802009,92802010]
TARGET=250

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

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def actor_pos(farm,idx):
    if idx==0: return list(farm["farmer"])
    return list(farm["hands"][idx-1])

def tile(obs,idx):
    p=int(obs["player"]); farm=obs["farms"][p]; x,y=actor_pos(farm,idx)
    return plain(farm["tiles"][y][x])

def inv(obs,idx):
    invs=obs["private"].get("inventories",[]) or []
    if idx>=len(invs): return {}
    return {k:v for k,v in invs[idx].items() if v}

def run_to_target(seed, model_path, tag):
    model=load(model_path,f"{tag}_m_{seed}")
    opp=load(OPP,f"{tag}_o_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    while not env.done:
        o0=shared(env,0); o1=shared(env,1); step=int(o0["step"])
        a0=plain(model.agent(deepcopy(o0),env.configuration)); a1=plain(opp.agent(deepcopy(o1)))
        if step==TARGET:
            pre={
                "cash":float(o0["farms"][0]["money"]),
                "actor3_pos":actor_pos(o0["farms"][0],3),
                "actor3_tile":tile(o0,3),
                "actor3_inv":inv(o0,3),
                "action":a0,
            }
            env.step([a0,a1])
            post=shared(env,0)
            return {
                "pre":pre,
                "post":{
                    "cash":float(post["farms"][0]["money"]),
                    "actor3_pos":actor_pos(post["farms"][0],3),
                    "actor3_tile":tile(post,3),
                    "actor3_inv":inv(post,3),
                    "shed":{k:v for k,v in post["private"]["shed"].items() if v},
                }
            }
        env.step([a0,a1])
    raise RuntimeError(seed)

def main():
    rows=[]
    for seed in SEEDS:
        b=run_to_target(seed,BASE,"b")
        c=run_to_target(seed,CAND,"c")
        row={"seed":seed,"baseline":b,"candidate":c}
        rows.append(row)
        print("ONE_STEP "+json.dumps(row,ensure_ascii=False,separators=(",",":")))
    out={"schema":"adaptive-circulation-step250-one-transition-v0","cases":rows,
         "boundary":"Immediate realized effect only; no causal claim about terminal."}
    Path("adaptive_circulation_step250_one_transition_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
