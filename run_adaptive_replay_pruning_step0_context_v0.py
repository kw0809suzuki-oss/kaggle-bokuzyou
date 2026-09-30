#!/usr/bin/env python3
import copy, hashlib, importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"adaptive_replay_pruning_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
TERMINAL_SIGN={
92802001:"improved",92802002:"improved",92802003:"improved",92802004:"improved",92802005:"worsened",
92802006:"improved",92802007:"worsened",92802008:"worsened",92802009:"improved",92802010:"worsened",
}
TARGET={"step":0,"market_index":2,"order":["BUY_PRODUCT","WHEAT",5]}

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

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def h(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def compact(obs):
    farm=obs["farms"][0]
    opp=obs["farms"][1]
    market=obs.get("market",{})
    town=obs.get("town",{})
    def inv(v):
        if isinstance(v,dict): return {k:v[k] for k in sorted(v)}
        return v
    return {
      "step":obs.get("step"),
      "self":{
        "money":farm.get("money"),
        "hires_today":farm.get("hires_today"),
        "hands":len(farm.get("hands",[]) or []),
        "shed":inv(farm.get("shed",{})),
        "inventory":inv(farm.get("inventory",{})),
        "land_count":len(farm.get("land",[]) or []) if isinstance(farm.get("land"),list) else farm.get("land"),
      },
      "opponent":{
        "money":opp.get("money"),
        "hires_today":opp.get("hires_today"),
        "hands":len(opp.get("hands",[]) or []),
        "shed":inv(opp.get("shed",{})),
      },
      "market":market,
      "town":town,
    }

def run_pair(seed):
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(2)
    base=load(BASE,f"base_{seed}_{os.getpid()}")
    cand=load(CAND,f"cand_{seed}_{os.getpid()}")
    cand.configure(TARGET["step"],TARGET["market_index"],TARGET["order"]); cand.reset_agent()
    opp=load(OPP,f"opp_{seed}_{os.getpid()}")

    pre=shared(env,0)
    base_action=plain(base.agent(copy.deepcopy(pre)))
    cand_action=plain(cand.agent(copy.deepcopy(pre)))
    opp_action=plain(opp.agent(shared(env,1)))

    # Baseline transition
    env_b=make("kaggriculture",configuration={"seed":seed},debug=False); env_b.reset(2)
    base2=load(BASE,f"base2_{seed}_{os.getpid()}"); opp_b=load(OPP,f"oppb_{seed}_{os.getpid()}")
    ob0=shared(env_b,0); ob1=shared(env_b,1)
    ab=plain(base2.agent(ob0)); aob=plain(opp_b.agent(ob1))
    env_b.step([ab,aob])
    post_b=shared(env_b,0)

    # Candidate transition
    env_c=make("kaggriculture",configuration={"seed":seed},debug=False); env_c.reset(2)
    cand2=load(CAND,f"cand2_{seed}_{os.getpid()}"); cand2.configure(TARGET["step"],TARGET["market_index"],TARGET["order"]); cand2.reset_agent()
    opp_c=load(OPP,f"oppc_{seed}_{os.getpid()}")
    oc0=shared(env_c,0); oc1=shared(env_c,1)
    ac=plain(cand2.agent(oc0)); aoc=plain(opp_c.agent(oc1))
    env_c.step([ac,aoc])
    post_c=shared(env_c,0)

    return {
      "seed":seed,
      "terminal_sign":TERMINAL_SIGN[seed],
      "pre_hash":h(pre),
      "pre_compact":compact(pre),
      "baseline_action":base_action,
      "candidate_action":cand_action,
      "opponent_action":opp_action,
      "post_baseline_hash":h(post_b),
      "post_candidate_hash":h(post_c),
      "post_equal":post_b==post_c,
      "post_baseline_compact":compact(post_b),
      "post_candidate_compact":compact(post_c),
    }

def main():
    rows=[run_pair(s) for s in SEEDS]
    pre_groups={}
    for r in rows: pre_groups.setdefault(r["pre_hash"],[]).append(r["seed"])
    out={
      "schema":"adaptive-replay-pruning-step0-context-v0",
      "target":TARGET,
      "rows":rows,
      "pre_unique_hashes":len(pre_groups),
      "pre_groups":pre_groups,
      "boundary":"step0 pre visible state and one official transition only; no terminal explanation",
    }
    Path("adaptive_replay_pruning_step0_context_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("STEP0_CONTEXT_SUMMARY",json.dumps({
      "pre_unique_hashes":out["pre_unique_hashes"],
      "pre_groups":pre_groups,
      "post_equal_cases":sum(r["post_equal"] for r in rows),
      "post_diverged_cases":sum(not r["post_equal"] for r in rows),
    },separators=(",",":")))
    for r in rows:
      print("STEP0_CONTEXT_CASE",json.dumps({
        "seed":r["seed"],"sign":r["terminal_sign"],"pre_hash":r["pre_hash"],
        "post_equal":r["post_equal"],
        "pre_compact":r["pre_compact"],
        "post_baseline_compact":r["post_baseline_compact"],
        "post_candidate_compact":r["post_candidate_compact"],
      },separators=(",",":")))

if __name__=="__main__": main()
