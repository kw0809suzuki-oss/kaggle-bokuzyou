#!/usr/bin/env python3
"""World Rhythm — WHEAT Value Lineage Probe v0.

Question:
  After the first Cash-caused WHEAT purchase-effect divergence, what happens to
  that missing WHEAT unit in the later World?

We do not infer value from the purchase itself.
We only trace where the differential migrates among:
  WHEAT seed count -> productive WHEAT plants -> shed WHEAT -> Cash.

Base:
  Adaptive Replay Runtime v0.1
Early:
  Base + HIRE +1 at step168

For each world:
  1. Find the first equal-core-action meaningful effect divergence from the
     prior Cash-to-Effect definition.
  2. From that boundary onward, record only the WHEAT lineage state
     (seed count, WHEAT plant count, shed WHEAT, Cash).
  3. Report the first later step where the original WHEAT differential changes
     form.

No intervention after step168. No Phase semantics. No new policy.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[94002001,94002002,94002003,94002004,94002005,94002006,94002007,94002008,94002009,94002010]
HIRE_STEP=168
START_COMPARE=169

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]

def wheat_plants(farm):
    n=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if isinstance(tile,dict) and tile.get("kind")=="PLANT" and tile.get("crop")=="WHEAT":
                n+=1
    return n

def wheat_state(obs,seat):
    farm=obs["farms"][seat]
    priv=obs["private"]
    return {
        "cash":float(farm.get("money",0) or 0),
        "seed_wheat":int((priv.get("seeds",{}) or {}).get("WHEAT",0) or 0),
        "plant_wheat":wheat_plants(farm),
        "shed_wheat":int((priv.get("shed",{}) or {}).get("WHEAT",0) or 0),
    }

def core_action(action):
    hands=list(action.get("hands",[]) or [])
    return {
        "farmer":deepcopy(action.get("farmer")),
        "hands_first8":deepcopy(hands[:8]),
        "market":deepcopy(action.get("market",[]) or []),
    }

def meaningful_delta(pre,post):
    return {
        "seed_wheat":post["seed_wheat"]-pre["seed_wheat"],
        "plant_wheat":post["plant_wheat"]-pre["plant_wheat"],
        "shed_wheat":post["shed_wheat"]-pre["shed_wheat"],
    }

def run(seed,seat,extra,tag):
    model=load(BASE,f"wheat_lineage_model_{tag}_{seed}_{seat}_{os.getpid()}")
    opp=load(OPP,f"wheat_lineage_opp_{tag}_{seed}_{seat}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    trace=[]
    while not env.done:
        o0=plain(shared(env,0));o1=plain(shared(env,1))
        so=o0 if seat==0 else o1
        oo=o1 if seat==0 else o0
        step=int(so.get("step",0) or 0)
        sa=plain(model.agent(so));oa=plain(opp.agent(oo))
        if extra and step==HIRE_STEP:
            sa=deepcopy(sa);sa.setdefault("market",[]).append(["HIRE"])
        pre=wheat_state(so,seat)
        if seat==0: env.step([sa,oa])
        else: env.step([oa,sa])
        po=plain(shared(env,seat));post=wheat_state(po,seat)
        trace.append({
            "step":step,
            "core_action":core_action(sa),
            "opponent_action":deepcopy(oa),
            "pre":pre,"post":post,
            "wheat_effect":meaningful_delta(pre,post),
        })
    final=plain(env.state[0].observation)
    return {"terminal_self":float(final["farms"][seat]["money"]),"trace":trace}

def first_boundary(base,early):
    b={x["step"]:x for x in base["trace"]};e={x["step"]:x for x in early["trace"]}
    for step in range(START_COMPARE,min(max(b),max(e))+1):
        if b[step]["core_action"]!=e[step]["core_action"]: continue
        if b[step]["wheat_effect"]!=e[step]["wheat_effect"]:
            return step
    return None

def diff_state(b,e):
    return {
        "cash":e["cash"]-b["cash"],
        "seed_wheat":e["seed_wheat"]-b["seed_wheat"],
        "plant_wheat":e["plant_wheat"]-b["plant_wheat"],
        "shed_wheat":e["shed_wheat"]-b["shed_wheat"],
    }

def first_lineage_change(base,early,boundary):
    if boundary is None: return None
    b={x["step"]:x for x in base["trace"]};e={x["step"]:x for x in early["trace"]}
    initial=diff_state(b[boundary]["post"],e[boundary]["post"])
    for step in range(boundary+1,min(max(b),max(e))+1):
        d=diff_state(b[step]["post"],e[step]["post"])
        if any(d[k]!=initial[k] for k in ("seed_wheat","plant_wheat","shed_wheat")):
            return {
                "step":step,
                "initial_lineage_diff":initial,
                "new_lineage_diff":d,
                "baseline_core_action":b[step]["core_action"],
                "early_core_action":e[step]["core_action"],
                "core_action_equal":b[step]["core_action"]==e[step]["core_action"],
                "baseline_wheat_effect":b[step]["wheat_effect"],
                "early_wheat_effect":e[step]["wheat_effect"],
            }
    return None

def main():
    cases=[]
    for seed in SEEDS:
        for seat in (0,1):
            b=run(seed,seat,False,"base")
            e=run(seed,seat,True,"early")
            boundary=first_boundary(b,e)
            byb={x["step"]:x for x in b["trace"]};bye={x["step"]:x for x in e["trace"]}
            initial_diff=diff_state(byb[boundary]["post"],bye[boundary]["post"]) if boundary is not None else None
            lineage=first_lineage_change(b,e,boundary)
            case={
                "seed":seed,"seat":seat,
                "baseline_terminal_self":b["terminal_self"],
                "early_terminal_self":e["terminal_self"],
                "delta_terminal":e["terminal_self"]-b["terminal_self"],
                "terminal_sign_group":"positive" if e["terminal_self"]>b["terminal_self"] else "negative",
                "first_wheat_effect_boundary_step":boundary,
                "boundary_core_action":byb[boundary]["core_action"] if boundary is not None else None,
                "boundary_initial_lineage_diff":initial_diff,
                "first_lineage_change":lineage,
            }
            cases.append(case)
            print("WHEAT_LINEAGE_CASE "+json.dumps(case,separators=(",",":")))
    pos=[c for c in cases if c["delta_terminal"]>0]
    neg=[c for c in cases if c["delta_terminal"]<0]
    summary={
        "worlds":len(cases),
        "positive_worlds":len(pos),
        "negative_worlds":len(neg),
        "positive_boundary_steps":[c["first_wheat_effect_boundary_step"] for c in pos],
        "negative_boundary_steps":[c["first_wheat_effect_boundary_step"] for c in neg],
        "positive_lineage_change_steps":[c["first_lineage_change"]["step"] if c["first_lineage_change"] else None for c in pos],
        "negative_lineage_change_steps":[c["first_lineage_change"]["step"] if c["first_lineage_change"] else None for c in neg],
    }
    out={
        "schema":"world-rhythm-wheat-value-lineage-v0",
        "question":"After the first Cash-caused WHEAT purchase-effect divergence, where does the missing WHEAT differential migrate next?",
        "summary":summary,
        "cases":cases,
        "boundary":{
            "observation_only":True,
            "no_phase_semantics":True,
            "no_new_policy":True,
            "no_new_contract":True,
            "no_value_claim_from_purchase_alone":True
        }
    }
    Path("world_rhythm_wheat_value_lineage_v0_result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("WHEAT_LINEAGE_SUMMARY "+json.dumps(summary,separators=(",",":")))

if __name__=="__main__":main()
