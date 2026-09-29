#!/usr/bin/env python3
# Trigger state-effect divergence probe.
import importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"decem_replay_state_effect_guard_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

spec=importlib.util.spec_from_file_location("_src_probe", ROOT/"run_decem_first_source_divergence_v0.py")
src=importlib.util.module_from_spec(spec); sys.modules["_src_probe"]=src; spec.loader.exec_module(src)
EXPECTED=src.EXPECTED

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def shared(env,seat): return env._Environment__get_shared_state(seat)["observation"]

def sig(obs):
    f=obs["farms"][0]; p=obs["private"]; tiles=[]
    for y,row in enumerate(f["tiles"]):
        for x,t in enumerate(row):
            if t and t!="LOCKED":
                tiles.append([x,y,t.get("kind"),t.get("crop"),t.get("animal"),t.get("stage"),t.get("yield_units"),t.get("watered")])
    return {"step":obs["step"],"day":obs["day"],"hour":obs["hour"],"money":f["money"],"farmer":f["farmer"],"hands":f["hands"],"unlocked":f["unlocked_quadrants"],"hires":f["hires_today"],"tiles":tiles,"shed":p["shed"],"seeds":p["seeds"],"inventories":p["inventories"]}

def diffs(a,b):
    return [k for k in ("farmer","hands","unlocked","hires","tiles","shed","seeds","inventories","money") if a.get(k)!=b.get(k)]

def main():
    seed=int(os.environ["SEED"])
    model=load(MODEL,f"m_{seed}"); opp=load(OPP,f"o_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)

    first_any=None; first_struct=None; checkpoints=[]
    while not env.done:
        pre=plain(shared(env,0)); step=int(pre["step"]); action=plain(model.agent(pre)); oa=plain(opp.agent(shared(env,1)))
        env.step([action,oa])
        if env.done: break
        post=plain(shared(env,0)); actual=sig(post); expected=EXPECTED[step+1]; ds=diffs(actual,expected)
        structural=[d for d in ds if d!="money"]
        if ds and first_any is None:
            first_any={"issued_step":step,"observed_step":step+1,"diff_fields":ds,"action":action,"actual":actual,"expected":expected}
        if structural and first_struct is None:
            first_struct={"issued_step":step,"observed_step":step+1,"diff_fields":structural,"action":action,"actual":actual,"expected":expected}
        if step in (23,24,25,26,27,28,29,30) or (structural and len(checkpoints)<12):
            checkpoints.append({"issued_step":step,"observed_step":step+1,"diff_fields":ds,"structural_fields":structural,"action":action,"actual_money":actual["money"],"expected_money":expected["money"]})
    final=plain(env.state[0].observation)
    out={"schema":"decem-state-effect-guard-first-divergence-v0","seed":seed,"terminal_self":float(final["farms"][0]["money"]),"trigger_count":int(getattr(model,"trigger_count",0)),"first_any_divergence":first_any,"first_structural_divergence":first_struct,"checkpoints":checkpoints}
    Path(f"decem_state_effect_guard_divergence_seed{seed}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STATE_EFFECT_DIVERGENCE "+json.dumps(out,separators=(",",":")))

if __name__=="__main__": main()
