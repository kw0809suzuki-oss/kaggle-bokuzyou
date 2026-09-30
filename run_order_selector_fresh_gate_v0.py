#!/usr/bin/env python3
"""Order Selector Fresh Gate v0.

Purpose:
  Fresh validation of the frozen two-stage C candidate rule.

Frozen rule (DO NOT TUNE ON THIS RUN):
  1) if market.inventory.EGG > 9961.5 -> Swap
  2) else if seat == 0 and market.inventory.WHEAT > 9899.5 -> Swap
  3) else -> Original

Fresh population:
  seeds 94003001..94003010 x seat0/1 = 20 Worlds

For each World:
  - capture step217 pre-State before intervention
  - run Original to terminal
  - run Swap to terminal
  - choose the rule-predicted side from the captured pre-State
  - compare the rule-selected terminal with Frozen A (Original)

Primary outputs:
  - Order prediction accuracy against the per-World better order
  - terminal self delta of frozen rule vs Frozen A

No threshold changes. No repair. No extra policy.
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
SEEDS=list(range(94003001,94003011))
AB=[["HIRE"],["BUY_PRODUCT","WHEAT",19]]
BA=[["BUY_PRODUCT","WHEAT",19],["HIRE"]]

EGG_THRESHOLD=9961.5
WHEAT_THRESHOLD=9899.5


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


def frozen_rule(pre_obs, seat):
    market=pre_obs["market"]
    egg=float((market.get("inventory",{}) or {}).get("EGG",0) or 0)
    wheat=float((market.get("inventory",{}) or {}).get("WHEAT",0) or 0)

    if egg > EGG_THRESHOLD:
        return "swap"
    if seat == 0 and wheat > WHEAT_THRESHOLD:
        return "swap"
    return "original"


def run_world(seed,seat,mode,capture=False):
    model=load(BASE,f"fresh_selector_model_{mode}_{seed}_{seat}_{os.getpid()}")
    opp=load(OPP,f"fresh_selector_opp_{mode}_{seed}_{seat}_{os.getpid()}")

    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    captured=None
    intervention_count=0

    while not env.done:
        o0=plain(shared(env,0))
        o1=plain(shared(env,1))
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
            if capture and captured is None:
                market=so["market"]
                captured={
                    "egg_inventory":float((market.get("inventory",{}) or {}).get("EGG",0) or 0),
                    "wheat_inventory":float((market.get("inventory",{}) or {}).get("WHEAT",0) or 0),
                    "self_cash":float(so["farms"][seat].get("money",0) or 0),
                    "seat":seat,
                    "rule_choice":frozen_rule(so,seat),
                }

            if mode=="swap":
                sa=deepcopy(sa)
                sa["market"]=deepcopy(BA)
                intervention_count += 1

        if seat==0:
            env.step([sa,oa])
        else:
            env.step([oa,sa])

    obs=plain(shared(env,seat))
    terminal=float(obs["farms"][seat].get("money",0) or 0)
    return terminal,captured,intervention_count


def main():
    cases=[]

    for seed in SEEDS:
        for seat in (0,1):
            original,pre,_=run_world(seed,seat,"original",capture=True)
            swap,_,swap_count=run_world(seed,seat,"swap",capture=False)

            if pre is None:
                raise RuntimeError("missing pre-State capture")
            if swap_count != 1:
                raise RuntimeError(f"swap intervention count != 1 seed={seed} seat={seat}: {swap_count}")

            oracle="swap" if swap>original else "original" if original>swap else "tie"
            choice=pre["rule_choice"]
            selected=swap if choice=="swap" else original
            delta_selected_vs_a=selected-original
            correct=(oracle=="tie" or choice==oracle)

            case={
                "seed":seed,
                "seat":seat,
                "egg_inventory":pre["egg_inventory"],
                "wheat_inventory":pre["wheat_inventory"],
                "self_cash":pre["self_cash"],
                "rule_choice":choice,
                "oracle_better_order":oracle,
                "prediction_correct":correct,
                "original_terminal_self":original,
                "swap_terminal_self":swap,
                "selected_terminal_self":selected,
                "delta_selected_vs_frozen_a":delta_selected_vs_a,
                "delta_swap_minus_original":swap-original,
            }
            cases.append(case)
            print("ORDER_SELECTOR_FRESH_CASE "+json.dumps(case,separators=(",",":")))

    ds=[c["delta_selected_vs_frozen_a"] for c in cases]
    pred=[c["prediction_correct"] for c in cases]

    summary={
        "worlds":len(cases),
        "prediction_correct":sum(pred),
        "prediction_accuracy":sum(pred)/len(pred),
        "swap_choices":sum(c["rule_choice"]=="swap" for c in cases),
        "original_choices":sum(c["rule_choice"]=="original" for c in cases),
        "oracle_swap_better":sum(c["oracle_better_order"]=="swap" for c in cases),
        "oracle_original_better":sum(c["oracle_better_order"]=="original" for c in cases),
        "oracle_ties":sum(c["oracle_better_order"]=="tie" for c in cases),
        "terminal_improved":sum(d>0 for d in ds),
        "terminal_worsened":sum(d<0 for d in ds),
        "terminal_same":sum(d==0 for d in ds),
        "mean_delta_selected_vs_frozen_a":sum(ds)/len(ds),
        "min_delta_selected_vs_frozen_a":min(ds),
        "max_delta_selected_vs_frozen_a":max(ds),
        "baseline_mean_terminal":sum(c["original_terminal_self"] for c in cases)/len(cases),
        "selected_mean_terminal":sum(c["selected_terminal_self"] for c in cases)/len(cases),
    }

    out={
        "schema":"order-selector-fresh-gate-v0",
        "frozen_rule":{
            "egg_threshold":EGG_THRESHOLD,
            "wheat_threshold":WHEAT_THRESHOLD,
            "logic":"if EGG > 9961.5 -> Swap; elif seat==0 and WHEAT inventory > 9899.5 -> Swap; else Original",
            "source_run":36673204618,
            "source_commit":"9d360962981f3e1b4a5cd0e0e34b60f31e18a144",
        },
        "fresh_population":{
            "seeds":SEEDS,
            "seats":[0,1],
            "worlds":len(cases),
        },
        "summary":summary,
        "cases":cases,
        "boundary":{
            "fresh_worlds":True,
            "thresholds_frozen_before_run":True,
            "no_posthoc_tuning":True,
            "same_action_set":True,
            "same_quantities":True,
            "single_order_intervention_step":TARGET_STEP,
            "frozen_a_is_original_order":True,
            "candidate_terminal_is_exact_selected_branch":True,
            "no_policy_promotion":True,
        }
    }

    Path("order_selector_fresh_gate_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("ORDER_SELECTOR_FRESH_SUMMARY "+json.dumps(summary,separators=(",",":")))


if __name__=="__main__":
    main()
