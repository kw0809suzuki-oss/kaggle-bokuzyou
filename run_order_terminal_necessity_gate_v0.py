#!/usr/bin/env python3
"""Order Terminal Necessity Gate v0.

Question:
  Is a state-dependent C selector actually necessary?

Compare, on the same 20 Worlds used by Order Swap Probe v0:
  Original: step217 market order = HIRE -> BUY_PRODUCT WHEAT 19
  Swap:     step217 market order = BUY_PRODUCT WHEAT 19 -> HIRE

Only the order at step217 is changed.
No quantity change. No repair. No other policy change.
After step217, both branches continue with the same frozen Adaptive Replay
runtime and the same opponent policy until terminal.

Primary observation:
  terminal self by World.
Decision boundary:
  - If one order is never worse across these Worlds, C is not yet necessary;
    a fixed order repair may be enough.
  - If the better order varies by World, state-dependent order selection has
    direct empirical motivation.

This probe does NOT implement C.
"""
from __future__ import annotations

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
SEEDS=[
    94002001,94002002,94002003,94002004,94002005,
    94002006,94002007,94002008,94002009,94002010,
]
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


def terminal_self(env,seat):
    obs=plain(shared(env,seat))
    farm=obs["farms"][seat]
    return float(farm.get("money",0) or 0)


def run_world(seed,seat,mode):
    model=load(BASE,f"order_terminal_model_{mode}_{seed}_{seat}_{os.getpid()}")
    opp=load(OPP,f"order_terminal_opp_{mode}_{seed}_{seat}_{os.getpid()}")

    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    intervention_count=0

    while not env.done:
        o0=plain(shared(env,0)); o1=plain(shared(env,1))
        so=o0 if seat==0 else o1
        oo=o1 if seat==0 else o0

        step=int(so.get("step",0) or 0)
        sa=plain(model.agent(so))
        oa=plain(opp.agent(oo))

        if step==TARGET_STEP:
            if sa.get("market",[]) != AB:
                raise RuntimeError(
                    f"unexpected step217 market seed={seed} seat={seat}: {sa.get('market')}"
                )
            if mode=="swap":
                sa=deepcopy(sa)
                sa["market"]=deepcopy(BA)
                intervention_count += 1

        if seat==0:
            env.step([sa,oa])
        else:
            env.step([oa,sa])

    return {
        "terminal_self":terminal_self(env,seat),
        "intervention_count":intervention_count,
    }


def main():
    cases=[]
    for seed in SEEDS:
        for seat in (0,1):
            original=run_world(seed,seat,"original")
            swap=run_world(seed,seat,"swap")
            delta=swap["terminal_self"]-original["terminal_self"]

            if delta>0:
                better="swap"
            elif delta<0:
                better="original"
            else:
                better="tie"

            case={
                "seed":seed,
                "seat":seat,
                "original_terminal_self":original["terminal_self"],
                "swap_terminal_self":swap["terminal_self"],
                "delta_swap_minus_original":delta,
                "better_order":better,
                "swap_intervention_count":swap["intervention_count"],
            }
            cases.append(case)
            print("ORDER_TERMINAL_CASE "+json.dumps(case,separators=(",",":")))

    ds=[c["delta_swap_minus_original"] for c in cases]
    summary={
        "worlds":len(cases),
        "swap_better":sum(d>0 for d in ds),
        "original_better":sum(d<0 for d in ds),
        "ties":sum(d==0 for d in ds),
        "mean_delta_swap_minus_original":sum(ds)/len(ds),
        "min_delta":min(ds),
        "max_delta":max(ds),
        "world_dependent_better_order":any(d>0 for d in ds) and any(d<0 for d in ds),
        "swap_triggered_exactly_once_all":all(c["swap_intervention_count"]==1 for c in cases),
    }

    out={
        "schema":"order-terminal-necessity-gate-v0",
        "question":"Does the terminal-better order vary by World for the confirmed step217 HIRE/WHEAT order swap?",
        "summary":summary,
        "cases":cases,
        "boundary":{
            "same_20_worlds_as_order_swap_probe":True,
            "single_order_intervention_step":TARGET_STEP,
            "same_operation_set":True,
            "same_quantities":True,
            "no_repair":True,
            "common_continuation_policy":True,
            "c_not_implemented":True
        }
    }

    Path("order_terminal_necessity_gate_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("ORDER_TERMINAL_SUMMARY "+json.dumps(summary,separators=(",",":")))


if __name__=="__main__":
    main()
