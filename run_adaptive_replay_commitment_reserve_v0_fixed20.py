#!/usr/bin/env python3
"""Paired gate: current Adaptive Replay vs Commitment Reserve v0.

10 fresh seeds x seat0/1 = 20 paired worlds.
Primary metric: terminal self. Also record trigger count.
"""
from __future__ import annotations
import importlib.util,json,os,statistics,sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"adaptive_replay_commitment_reserve_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[93901001,93901002,93901003,93901004,93901005,93901006,93901007,93901008,93901009,93901010]

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"):return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m)
    if hasattr(m,"reset_agent"):m.reset_agent()
    return m

def shared(env,seat):return env._Environment__get_shared_state(seat)["observation"]

def run(seed,seat,path,tag):
    model=load(path,f"{tag}_{seed}_{seat}_{os.getpid()}")
    opp=load(OPP,f"opp_{tag}_{seed}_{seat}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    while not env.done:
        o0=plain(shared(env,0));o1=plain(shared(env,1))
        if seat==0:a0,a1=plain(model.agent(o0)),plain(opp.agent(o1))
        else:a0,a1=plain(opp.agent(o0)),plain(model.agent(o1))
        env.step([a0,a1])
    final=plain(env.state[0].observation)
    return {
      "terminal_self":float(final["farms"][seat]["money"]),
      "terminal_opponent":float(final["farms"][1-seat]["money"]),
      "reserve_trigger_count":int(getattr(model,"reserve_trigger_count",0)),
      "last_reserve_event":plain(getattr(model,"last_reserve_event",None))
    }

def summarize(vals):
    return {"mean":statistics.mean(vals),"median":statistics.median(vals),"min":min(vals),"max":max(vals)}

def main():
    cases=[]
    for seed in SEEDS:
      for seat in (0,1):
        b=run(seed,seat,BASE,"base");c=run(seed,seat,CAND,"cand")
        case={"seed":seed,"seat":seat,"baseline":b,"candidate":c,
              "delta_terminal_self":c["terminal_self"]-b["terminal_self"]}
        cases.append(case);print("COMMITMENT_RESERVE_CASE "+json.dumps(case,separators=(",",":")))
    ds=[x["delta_terminal_self"] for x in cases]
    bs=[x["baseline"]["terminal_self"] for x in cases]
    cs=[x["candidate"]["terminal_self"] for x in cases]
    summary={
      "n":len(cases),
      "triggered":sum(x["candidate"]["reserve_trigger_count"]>0 for x in cases),
      "improved":sum(d>0 for d in ds),"worsened":sum(d<0 for d in ds),"same":sum(d==0 for d in ds),
      "mean_delta_terminal_self":statistics.mean(ds),
      "median_delta_terminal_self":statistics.median(ds),
      "min_delta":min(ds),"max_delta":max(ds),
      "baseline_terminal":summarize(bs),"candidate_terminal":summarize(cs),
    }
    print("COMMITMENT_RESERVE_SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_replay_commitment_reserve_v0_fixed20_result.json").write_text(
      json.dumps({"schema":"adaptive-replay-commitment-reserve-v0-fixed20","summary":summary,"cases":cases,
      "boundary":{"separate_experimental_model":True,"not_an_effect_contract":True,"no_promotion_without_terminal_evidence":True}},
      ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
