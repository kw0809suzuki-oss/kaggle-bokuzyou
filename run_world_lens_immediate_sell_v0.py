#!/usr/bin/env python3
"""Fixed10 Adaptive Replay baseline vs World Lens Immediate Sell v0."""

from __future__ import annotations
import importlib.util,json,statistics,sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"world_lens_immediate_sell_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"):mod.reset_agent()
    return mod

def shared(env,seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])

def run(seed,path,tag):
    m=load(path,f"{tag}_{seed}");o=load(OPP,f"opp_{tag}_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    while not env.done:
        a=shared(env,0);b=shared(env,1)
        env.step([plain(m.agent(a,env.configuration)),plain(o.agent(b))])
    fin=plain(env.state[0].observation)
    s=float(fin["farms"][0]["money"]);q=float(fin["farms"][1]["money"])
    return {
      "self":s,"opp":q,"margin":s-q,
      "trigger_count":int(getattr(m,"trigger_count",0)),
      "rerouted_units":int(getattr(m,"rerouted_units",0)),
      "last_event":plain(getattr(m,"last_event",None)),
    }

def main():
    rows=[]
    for seed in SEEDS:
        b=run(seed,BASE,"base");c=run(seed,CAND,"cand")
        row={"seed":seed,"baseline":b,"candidate":c,"delta_self":c["self"]-b["self"],"delta_margin":c["margin"]-b["margin"]}
        rows.append(row)
        print("CASE "+json.dumps(row,ensure_ascii=False,separators=(",",":")))
    ds=[r["delta_self"] for r in rows]; dm=[r["delta_margin"] for r in rows]
    summary={
      "n":10,
      "improved":sum(x>0 for x in ds),"worsened":sum(x<0 for x in ds),"same":sum(x==0 for x in ds),
      "mean_delta_self":statistics.mean(ds),"median_delta_self":statistics.median(ds),
      "min_delta_self":min(ds),"max_delta_self":max(ds),
      "mean_delta_margin":statistics.mean(dm),
      "baseline_mean":statistics.mean(r["baseline"]["self"] for r in rows),
      "candidate_mean":statistics.mean(r["candidate"]["self"] for r in rows),
      "mean_triggers":statistics.mean(r["candidate"]["trigger_count"] for r in rows),
      "mean_rerouted_units":statistics.mean(r["candidate"]["rerouted_units"] for r in rows),
    }
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("world_lens_immediate_sell_v0_result.json").write_text(json.dumps({
      "schema":"world-lens-immediate-sell-v0",
      "summary":summary,
      "cases":rows,
      "boundary":[
        "Replay decides sell timing and requested sell-only quantity.",
        "WHEAT and FERTILIZER are untouched.",
        "No future-price forecast or fitted threshold.",
        "Terminal self decides adoption; negative result rejects rule as written."
      ]
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
