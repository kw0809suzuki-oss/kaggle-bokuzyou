#!/usr/bin/env python3
import copy, importlib.util, json, os, statistics, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"adaptive_replay_pruning_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SCREEN_SEED=92802001
FIXED10=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
ELIGIBLE={"HIRE","BUY_ANIMAL","BUY_SEED","BUY_LAND","BUY_PRODUCT"}

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

def shared(env,seat): return env._Environment__get_shared_state(seat)["observation"]

def run(seed, mode="base", candidate=None):
    path=BASE if mode=="base" else CAND
    mod=load(path,f"{mode}_{seed}_{os.getpid()}_{candidate and candidate['step']}_{candidate and candidate['market_index']}")
    if mode!="base":
        mod.configure(candidate["step"],candidate["market_index"],candidate["order"]); mod.reset_agent()
    opp=load(OPP,f"opp_{mode}_{seed}_{os.getpid()}_{candidate and candidate['step']}_{candidate and candidate['market_index']}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(2)
    trace=[]
    while not env.done:
        o0=plain(shared(env,0)); o1=plain(shared(env,1))
        a0=plain(mod.agent(o0)); a1=plain(opp.agent(o1))
        trace.append({"step":int(o0["step"]),"action":copy.deepcopy(a0)})
        env.step([a0,a1])
    f=plain(env.state[0].observation)
    return {
      "terminal_self":float(f["farms"][0]["money"]),
      "terminal_opponent":float(f["farms"][1]["money"]),
      "margin":float(f["farms"][0]["money"]-f["farms"][1]["money"]),
      "activation_count":int(getattr(mod,"activation_count",0)),
      "last_event":plain(getattr(mod,"last_event",None)),
      "trace":trace,
    }

def candidates_from(trace):
    out=[]
    for row in trace:
        for i,order in enumerate(row["action"].get("market",[]) or []):
            if isinstance(order,list) and order and order[0] in ELIGIBLE:
                out.append({"step":row["step"],"market_index":i,"order":order})
    return out

def main():
    base_screen=run(SCREEN_SEED)
    candidates=candidates_from(base_screen["trace"])
    screen=[]
    for n,c in enumerate(candidates,1):
        x=run(SCREEN_SEED,"cand",c)
        row={**c,
          "activation_count":x["activation_count"],
          "baseline_terminal_self":base_screen["terminal_self"],
          "candidate_terminal_self":x["terminal_self"],
          "delta_terminal_self":x["terminal_self"]-base_screen["terminal_self"],
          "delta_margin":x["margin"]-base_screen["margin"],
        }
        screen.append(row)
        print("PRUNE_SCREEN",n,len(candidates),json.dumps(row,separators=(",",":")))
    survivors=[r for r in screen if r["activation_count"]==1 and r["delta_terminal_self"]>0]
    survivors=sorted(survivors,key=lambda r:r["delta_terminal_self"],reverse=True)[:3]

    fixed=[]
    baseline_by_seed={s:run(s) for s in FIXED10}
    for rank,c in enumerate(survivors,1):
        rows=[]
        for seed in FIXED10:
            b=baseline_by_seed[seed]; x=run(seed,"cand",c)
            rows.append({
              "seed":seed,"activation_count":x["activation_count"],
              "baseline_terminal_self":b["terminal_self"],
              "candidate_terminal_self":x["terminal_self"],
              "delta_terminal_self":x["terminal_self"]-b["terminal_self"],
              "delta_margin":x["margin"]-b["margin"],
            })
        ds=[r["delta_terminal_self"] for r in rows]
        summary={
          "rank":rank,"candidate":{k:c[k] for k in ("step","market_index","order")},
          "improved":sum(d>0 for d in ds),"worsened":sum(d<0 for d in ds),"same":sum(d==0 for d in ds),
          "activated_cases":sum(r["activation_count"]==1 for r in rows),
          "mean_delta_terminal_self":statistics.mean(ds),
          "median_delta_terminal_self":statistics.median(ds),
          "min_delta_terminal_self":min(ds),"max_delta_terminal_self":max(ds),
          "rows":rows,
        }
        fixed.append(summary)
        print("PRUNE_FIXED10",json.dumps({k:v for k,v in summary.items() if k!="rows"},separators=(",",":")))

    out={
      "schema":"adaptive-replay-pruning-v0",
      "baseline":"adaptive_replay_contract_runtime_v0",
      "screen_seed":SCREEN_SEED,
      "eligible_operations":sorted(ELIGIBLE),
      "screen_candidate_count":len(candidates),
      "screen_positive_count":len([r for r in screen if r["activation_count"]==1 and r["delta_terminal_self"]>0]),
      "screen_top3":survivors,
      "fixed10":fixed,
      "screen":screen,
      "boundary":"one spending market order omitted; no worker change, reorder, recovery, or rescue tuning",
    }
    Path("adaptive_replay_pruning_v0_result.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PRUNE_SUMMARY",json.dumps({
      "screen_candidate_count":out["screen_candidate_count"],
      "screen_positive_count":out["screen_positive_count"],
      "top3":[{k:r[k] for k in ("step","market_index","order","delta_terminal_self")} for r in survivors],
      "fixed10":[{k:s[k] for k in ("rank","candidate","improved","worsened","same","activated_cases","mean_delta_terminal_self")} for s in fixed],
    },separators=(",",":")))

if __name__=="__main__": main()
