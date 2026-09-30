#!/usr/bin/env python3
"""Fixed10 equivalence: Contract Runtime v0 vs World Lens Guard v1."""

from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"adaptive_replay_world_lens_guard_v1.py"
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

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def run(seed,path,tag):
    m=load(path,f"{tag}_{seed}");o=load(OPP,f"opp_{tag}_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    actions=[]
    while not env.done:
        a=shared(env,0);b=shared(env,1)
        act=plain(m.agent(a,env.configuration));opp=plain(o.agent(b))
        actions.append(act)
        env.step([act,opp])
    fin=plain(env.state[0].observation)
    return {
      "actions":actions,
      "self":float(fin["farms"][0]["money"]),
      "opp":float(fin["farms"][1]["money"]),
      "trigger_count":int(getattr(m,"trigger_count",0)),
    }

def main():
    rows=[]
    for seed in SEEDS:
        b=run(seed,BASE,"base");c=run(seed,CAND,"cand")
        row={
          "seed":seed,
          "action_match":b["actions"]==c["actions"],
          "terminal_match":b["self"]==c["self"] and b["opp"]==c["opp"],
          "base_self":b["self"],"candidate_self":c["self"],
          "base_triggers":b["trigger_count"],"candidate_triggers":c["trigger_count"],
        }
        rows.append(row)
        print("CASE "+json.dumps(row,separators=(",",":")))
    summary={
      "n":10,
      "action_match":sum(r["action_match"] for r in rows),
      "terminal_match":sum(r["terminal_match"] for r in rows),
      "all_exact":all(r["action_match"] and r["terminal_match"] for r in rows),
    }
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_replay_world_lens_guard_v1_equivalence.json").write_text(
      json.dumps({"schema":"adaptive-replay-world-lens-guard-v1-equivalence","summary":summary,"cases":rows},indent=2)+"\n",
      encoding="utf-8"
    )

if __name__=="__main__":main()
