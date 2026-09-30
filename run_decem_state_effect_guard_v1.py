#!/usr/bin/env python3
import importlib.util,json,os,sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
V0=ROOT/"decem_replay_state_effect_guard_v0.py"
V1=ROOT/"decem_replay_state_effect_guard_v1.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"):return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"):mod.reset_agent()
    return mod

def shared(env,seat):return env._Environment__get_shared_state(seat)["observation"]

def run(seed,path,name):
    m=load(path,f"{name}_{seed}_{os.getpid()}");o=load(OPP,f"opp_{name}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    pre85=None
    while not env.done:
        obs=plain(shared(env,0))
        if int(obs["step"])==85:
            pre85={"money":obs["farms"][0]["money"],"strawberry_seeds":obs["private"]["seeds"]["STRAWBERRY"]}
        a0=plain(m.agent(obs));a1=plain(o.agent(shared(env,1)));env.step([a0,a1])
    f=plain(env.state[0].observation)
    return {"terminal_self":float(f["farms"][0]["money"]),"pre85":pre85,
            "trigger_step24":int(getattr(m,"trigger_step24",getattr(m,"trigger_count",0))),
            "trigger_step85":int(getattr(m,"trigger_step85",0))}

def main():
    seed=int(os.environ["SEED"])
    v0=run(seed,V0,"v0");v1=run(seed,V1,"v1")
    out={"schema":"decem-state-effect-guard-v1-ab","seed":seed,"v0":v0,"v1":v1,"delta_terminal_self":v1["terminal_self"]-v0["terminal_self"]}
    Path(f"decem_state_effect_guard_v1_seed{seed}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STATE_EFFECT_GUARD_V1 "+json.dumps(out,separators=(",",":")))
if __name__=="__main__":main()
