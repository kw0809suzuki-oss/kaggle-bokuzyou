#!/usr/bin/env python3
"""Observe subject cash-changing transitions from step97 through step110.

Purpose: locate one minimal spending transition that could form a genuinely
different exploratory model. Observation only; no intervention.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS={"positive":93803004,"negative":93803005}
START,END=97,110

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m)
    if hasattr(m,"reset_agent"): m.reset_agent()
    return m

def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]

def run(seed,label):
    model=load(MODEL,f"cashpath_model_{label}_{seed}_{os.getpid()}")
    opp=load(OPP,f"cashpath_opp_{label}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    rows=[]
    while not env.done:
        o0=plain(shared(env,0)); o1=plain(shared(env,1)); step=int(o0.get("step",0) or 0)
        a0=plain(model.agent(o0)); a1=plain(opp.agent(o1))
        pre=float(o0["farms"][0]["money"])
        env.step([a0,a1])
        if START <= step <= END:
            post=float(plain(shared(env,0))["farms"][0]["money"])
            rows.append({
                "step":step,"pre_cash":pre,"post_cash":post,"cash_delta":post-pre,
                "subject_market":a0.get("market",[]) or [],
                "subject_action":a0,
                "opponent_market":a1.get("market",[]) or []
            })
    return rows

def main():
    runs={k:run(v,k) for k,v in SEEDS.items()}
    p={r["step"]:r for r in runs["positive"]}; n={r["step"]:r for r in runs["negative"]}
    comparison=[]
    for step in range(START,END+1):
        pr,nr=p[step],n[step]
        comparison.append({
            "step":step,
            "action_equal":pr["subject_action"]==nr["subject_action"],
            "positive_pre_cash":pr["pre_cash"],"negative_pre_cash":nr["pre_cash"],
            "pre_cash_gap":pr["pre_cash"]-nr["pre_cash"],
            "positive_delta":pr["cash_delta"],"negative_delta":nr["cash_delta"],
            "positive_market":pr["subject_market"],"negative_market":nr["subject_market"],
        })
    print("CASH_PATH_97_110 "+json.dumps(comparison,separators=(",",":")))
    Path("adaptive_replay_cash_path_97_110_v0.json").write_text(json.dumps({
        "schema":"adaptive-replay-cash-path-97-110-v0",
        "runs":runs,"comparison":comparison,
        "boundary":{"observation_only":True,"no_intervention":True}
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
