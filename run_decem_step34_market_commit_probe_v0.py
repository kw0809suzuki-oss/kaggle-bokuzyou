#!/usr/bin/env python3
import importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make
import kaggle_environments.envs.kaggriculture.kaggriculture as kg

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"decem_replay_state_effect_guard_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

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

def main():
    seed=int(os.environ["SEED"])
    model=load(MODEL,f"m34_{seed}"); opp=load(OPP,f"o34_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)

    while not env.done:
        pre=plain(shared(env,0)); step=int(pre["step"])
        a0=plain(model.agent(pre)); a1=plain(opp.agent(shared(env,1)))
        if step != 34:
            env.step([a0,a1]); continue

        pre_market=plain(pre["market"])
        pre_private=plain(pre["private"])
        trace=[]
        original=kg._commit_unit

        def wrapped(op,item,price,farm,private,market,shed_capacity=100):
            before={
                "op":op,"item":item,"price":price,
                "money":float(farm["money"]),
                "shed_item":int(private["shed"].get(item,0)),
                "market_inventory":int(market["inventory"].get(item,0)),
            }
            ok=original(op,item,price,farm,private,market,shed_capacity)
            after={
                "money":float(farm["money"]),
                "shed_item":int(private["shed"].get(item,0)),
                "market_inventory":int(market["inventory"].get(item,0)),
            }
            trace.append({"before":before,"ok":bool(ok),"after":after})
            return ok

        kg._commit_unit=wrapped
        try:
            env.step([a0,a1])
        finally:
            kg._commit_unit=original

        post=plain(shared(env,0))
        out={
            "schema":"decem-step34-market-commit-probe-v0",
            "seed":seed,
            "pre":{
                "money":pre["farms"][0]["money"],
                "shed":pre_private["shed"],
                "market_inventory":pre_market["inventory"],
                "market_prices":pre_market["prices"],
            },
            "self_action":a0,
            "opponent_action":a1,
            "commit_trace":trace,
            "post":{
                "money":post["farms"][0]["money"],
                "shed":post["private"]["shed"],
                "market_inventory":post["market"]["inventory"],
                "market_prices":post["market"]["prices"],
            },
        }
        fn=f"decem_step34_market_commit_seed{seed}.json"
        Path(fn).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("STEP34_MARKET_COMMIT "+json.dumps(out,separators=(",",":")))
        return

if __name__=="__main__":
    main()
