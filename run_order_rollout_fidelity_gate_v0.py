#!/usr/bin/env python3
"""Order Rollout Fidelity Gate v0.

Parent goal:
  terminal self を高く残す。

This is NOT a strength test.
It validates the minimal comparison harness C would need.

At step217:
  1. Replay Official World to the exact state.
  2. Capture subject/opponent actions.
  3. Fork the live Official env with deepcopy.
  4. Advance one fork with Original:
       HIRE -> BUY_PRODUCT WHEAT 19
     and one fork with Swap:
       BUY_PRODUCT WHEAT 19 -> HIRE
  5. Independently replay Official World from the start for each branch.
  6. Compare:
       Cash / hands / hires_today / shed WHEAT /
       market WHEAT inventory / market WHEAT price

Gate passes only if forked rollout == independent Official replay
for both Original and Swap.

Representative worlds:
  low cash  seed 94002001 seat0
  mid cash  seed 94002003 seat0
  high cash seed 94002008 seat0

No terminal. No policy promotion. No repair.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

TARGET_STEP=217
SEEDS=[94002001,94002003,94002008]
AB=[["HIRE"],["BUY_PRODUCT","WHEAT",19]]
BA=[["BUY_PRODUCT","WHEAT",19],["HIRE"]]


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
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


def snap(obs,seat=0):
    farm=obs["farms"][seat]
    priv=obs["private"]
    market=obs["market"]
    return {
        "cash":float(farm.get("money",0) or 0),
        "hands":len(farm.get("hands",[]) or []),
        "hires_today":int(farm.get("hires_today",0) or 0),
        "shed_wheat":int((priv.get("shed",{}) or {}).get("WHEAT",0) or 0),
        "market_wheat_inventory":int((market.get("inventory",{}) or {}).get("WHEAT",0) or 0),
        "market_wheat_price":int((market.get("prices",{}) or {}).get("WHEAT",0) or 0),
    }


def prepare_live(seed):
    model=load(BASE,f"fidelity_live_model_{seed}_{os.getpid()}")
    opp=load(OPP,f"fidelity_live_opp_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0=plain(shared(env,0))
        o1=plain(shared(env,1))
        step=int(o0.get("step",0) or 0)
        sa=plain(model.agent(o0))
        oa=plain(opp.agent(o1))

        if step==TARGET_STEP:
            if sa.get("market",[]) != AB:
                raise RuntimeError(f"unexpected step217 market seed={seed}: {sa.get('market')}")
            return env, o0, sa, oa

        env.step([sa,oa])

    raise RuntimeError("target step not reached")


def fork_rollout(env, subject_action, opponent_action):
    fork=copy.deepcopy(env)
    fork.step([deepcopy(subject_action),deepcopy(opponent_action)])
    return snap(plain(shared(fork,0)),0)


def independent_official(seed, mode):
    model=load(BASE,f"fidelity_ref_model_{mode}_{seed}_{os.getpid()}")
    opp=load(OPP,f"fidelity_ref_opp_{mode}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    while not env.done:
        o0=plain(shared(env,0))
        o1=plain(shared(env,1))
        step=int(o0.get("step",0) or 0)
        sa=plain(model.agent(o0))
        oa=plain(opp.agent(o1))

        if step==TARGET_STEP:
            if sa.get("market",[]) != AB:
                raise RuntimeError(f"unexpected ref step217 market seed={seed}: {sa.get('market')}")
            if mode=="swap":
                sa=deepcopy(sa)
                sa["market"]=deepcopy(BA)
            env.step([sa,oa])
            return snap(plain(shared(env,0)),0)

        env.step([sa,oa])

    raise RuntimeError("target step not reached")


def main():
    cases=[]
    for seed in SEEDS:
        env,pre_obs,sa,oa=prepare_live(seed)

        original_action=deepcopy(sa)
        swap_action=deepcopy(sa)
        swap_action["market"]=deepcopy(BA)

        pre=snap(pre_obs,0)

        fork_original=fork_rollout(env,original_action,oa)
        fork_swap=fork_rollout(env,swap_action,oa)

        ref_original=independent_official(seed,"original")
        ref_swap=independent_official(seed,"swap")

        case={
            "seed":seed,
            "pre":pre,
            "fork_original":fork_original,
            "official_original":ref_original,
            "original_exact_match":fork_original==ref_original,
            "fork_swap":fork_swap,
            "official_swap":ref_swap,
            "swap_exact_match":fork_swap==ref_swap,
        }
        cases.append(case)
        print("ORDER_ROLLOUT_FIDELITY_CASE "+json.dumps(case,separators=(",",":")))

    summary={
        "cases":len(cases),
        "original_exact_matches":sum(c["original_exact_match"] for c in cases),
        "swap_exact_matches":sum(c["swap_exact_match"] for c in cases),
        "all_exact":all(c["original_exact_match"] and c["swap_exact_match"] for c in cases),
    }

    out={
        "schema":"order-rollout-fidelity-gate-v0",
        "question":"Can a forked Official-env comparator reproduce the confirmed step217 Original/Swap effects exactly?",
        "summary":summary,
        "cases":cases,
        "boundary":{
            "representative_worlds_only":True,
            "seeds":SEEDS,
            "seat":0,
            "single_transition_only":True,
            "official_env_forked_with_deepcopy":True,
            "independent_official_replay_reference":True,
            "terminal_not_used":True,
            "no_policy_promotion":True,
            "no_repair":True
        }
    }

    Path("order_rollout_fidelity_gate_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("ORDER_ROLLOUT_FIDELITY_SUMMARY "+json.dumps(summary,separators=(",",":")))

    if not summary["all_exact"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
