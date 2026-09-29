#!/usr/bin/env python3
import importlib.util, json, os, sys, requests
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"decem_replay_state_effect_guard_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
EPISODE_ID=115303987
SOURCE_SEAT=0

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

def source_obs(steps,idx,seat=0):
    s=steps[idx][seat]
    o=s.get("observation")
    if isinstance(o,str): o=json.loads(o)
    return o

def compact_market(o):
    m=o["market"]
    return {"inventory":plain(m["inventory"]),"prices":plain(m["prices"])}

def main():
    seed=int(os.environ["SEED"])
    rr=requests.get(f"https://www.kaggle.com/competitions/episodes/{EPISODE_ID}/replay.json",timeout=30)
    replay=rr.json()
    if isinstance(replay,dict) and "replay" in replay and isinstance(replay["replay"],str):
        replay=json.loads(replay["replay"])
    steps=replay["steps"]

    model=load(MODEL,f"m_market_{seed}"); opp=load(OPP,f"o_market_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    first=None
    prev_self=None; prev_opp=None
    while not env.done:
        pre=plain(shared(env,0)); step=int(pre["step"])
        so=source_obs(steps,step,SOURCE_SEAT)
        actual_m=compact_market(pre); source_m=compact_market(so)
        fields=[]
        if actual_m["inventory"]!=source_m["inventory"]: fields.append("inventory")
        if actual_m["prices"]!=source_m["prices"]: fields.append("prices")
        if fields:
            source_prev_self=None; source_prev_opp=None
            if step>0:
                source_prev_self=plain(steps[step][0].get("action"))
                source_prev_opp=plain(steps[step][1].get("action"))
            first={
                "observed_step":step,
                "day":pre["day"],
                "hour":pre["hour"],
                "diff_fields":fields,
                "actual_market":actual_m,
                "source_market":source_m,
                "previous_current_self_action":prev_self,
                "previous_current_opponent_action":prev_opp,
                "previous_source_self_action":source_prev_self,
                "previous_source_opponent_action":source_prev_opp,
            }
            break
        a0=plain(model.agent(pre)); a1=plain(opp.agent(shared(env,1)))
        prev_self=a0; prev_opp=a1
        env.step([a0,a1])

    out={"schema":"decem-first-market-divergence-v0","seed":seed,"source_episode":EPISODE_ID,"first_market_divergence":first}
    Path(f"decem_first_market_divergence_seed{seed}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("FIRST_MARKET_DIVERGENCE "+json.dumps(out,separators=(",",":")))

if __name__=="__main__":
    main()
