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

def parse(x):
    if isinstance(x,str):
        try:return json.loads(x)
        except Exception:return x
    return x

def src_obs(steps,idx):
    return plain(parse(steps[idx][SOURCE_SEAT].get("observation")))

def src_action(steps,idx):
    return plain(parse(steps[idx][SOURCE_SEAT].get("action")))

def self_struct(obs):
    f=obs["farms"][0]; p=obs["private"]
    return {
        "farmer":f["farmer"],
        "hands":f["hands"],
        "unlocked":f["unlocked_quadrants"],
        "hires":f["hires_today"],
        "tiles":f["tiles"],
        "shed":p["shed"],
        "seeds":p["seeds"],
        "inventories":p["inventories"],
    }

def diff_paths(a,b,path=""):
    out=[]
    if type(a) != type(b):
        return [path or "$"]
    if isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            p=f"{path}.{k}" if path else k
            if k not in a or k not in b: out.append(p)
            else: out.extend(diff_paths(a[k],b[k],p))
        return out
    if isinstance(a,list):
        if len(a)!=len(b):
            out.append((path or "$")+".length")
        for i,(x,y) in enumerate(zip(a,b)):
            out.extend(diff_paths(x,y,f"{path}[{i}]"))
        return out
    if a!=b: out.append(path or "$")
    return out

def main():
    seed=int(os.environ["SEED"])
    rr=requests.get(f"https://www.kaggle.com/competitions/episodes/{EPISODE_ID}/replay.json",timeout=30)
    rr.raise_for_status()
    replay=rr.json()
    if isinstance(replay,dict) and isinstance(replay.get("replay"),str):
        replay=json.loads(replay["replay"])
    steps=replay["steps"]

    model=load(MODEL,f"m_prop_{seed}"); opp=load(OPP,f"o_prop_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)

    base_residual=None
    first_extension=None
    checkpoints=[]
    while not env.done:
        pre=plain(shared(env,0)); step=int(pre["step"])
        a0=plain(model.agent(pre)); a1=plain(opp.agent(shared(env,1)))
        env.step([a0,a1])
        if env.done: break
        observed=step+1
        actual=plain(shared(env,0))
        source=src_obs(steps,observed)
        paths=diff_paths(self_struct(actual),self_struct(source))

        if observed==35:
            base_residual=paths[:]
        if observed>=35 and observed<=45:
            checkpoints.append({
                "observed_step":observed,
                "issued_step":step,
                "diff_paths":paths,
                "current_action":a0,
                "source_action":src_action(steps,observed),
                "current_money":actual["farms"][0]["money"],
                "source_money":source["farms"][0]["money"],
            })
        if observed>35 and base_residual is not None:
            extra=[p for p in paths if p not in base_residual]
            if extra and first_extension is None:
                first_extension={
                    "issued_step":step,
                    "observed_step":observed,
                    "base_residual_paths":base_residual,
                    "all_diff_paths":paths,
                    "new_diff_paths":extra,
                    "current_action":a0,
                    "source_action":src_action(steps,observed),
                    "current_money":actual["farms"][0]["money"],
                    "source_money":source["farms"][0]["money"],
                    "current_struct":self_struct(actual),
                    "source_struct":self_struct(source),
                }
                break

    out={
        "schema":"decem-step34-residual-propagation-v0",
        "seed":seed,
        "base_residual_at_step35":base_residual,
        "first_structural_extension":first_extension,
        "checkpoints":checkpoints,
    }
    Path(f"decem_step34_residual_propagation_seed{seed}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STEP34_RESIDUAL_PROPAGATION "+json.dumps(out,separators=(",",":")))

if __name__=="__main__": main()
