#!/usr/bin/env python3
import importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "decem_replay_distilled_157026_v0.py"
CAND = ROOT / "decem_replay_state_effect_guard_v0.py"
OPP = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

def plain(x):
    if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    if isinstance(x, (str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k): plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]

def run(seed, model_path, name):
    model=load(model_path,f"{name}_{seed}_{os.getpid()}")
    opp=load(OPP,f"opp_{name}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    pre24=None
    while not env.done:
        obs0=plain(shared(env,0))
        if int(obs0["step"])==24:
            pre24={"money":obs0["farms"][0]["money"],"hands":len(obs0["farms"][0]["hands"]),"hires_today":obs0["farms"][0]["hires_today"]}
        a0=plain(model.agent(obs0))
        a1=plain(opp.agent(shared(env,1)))
        env.step([a0,a1])
    final=plain(env.state[0].observation)
    return {
        "terminal_self":float(final["farms"][0]["money"]),
        "pre24":pre24,
        "trigger_count":int(getattr(model,"trigger_count",0)),
    }

def main():
    seed=int(os.environ["SEED"])
    baseline=run(seed,BASE,"baseline")
    candidate=run(seed,CAND,"candidate")
    out={
        "schema":"decem-state-effect-guard-v0-ab",
        "seed":seed,
        "baseline":baseline,
        "candidate":candidate,
        "delta_terminal_self":candidate["terminal_self"]-baseline["terminal_self"],
    }
    fn=f"decem_state_effect_guard_seed{seed}.json"
    Path(fn).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STATE_EFFECT_GUARD "+json.dumps(out,separators=(",",":")))

if __name__=="__main__":
    main()
