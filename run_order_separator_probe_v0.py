#!/usr/bin/env python3
"""Order Separator Probe v0.

Question:
  Can the terminal-better order at step217 be separated from information that
  is observable BEFORE the order intervention?

Population:
  Same 20 Worlds as Order Terminal Necessity Gate v0.

Labels:
  swap     if Swap terminal self > Original terminal self
  original if Original terminal self > Swap terminal self

Procedure:
  1. Capture step217 pre-State in one fixed schema before any intervention.
  2. Run Original and Swap to terminal to obtain the label.
  3. Join the label only after the pre-State snapshot is captured.
  4. Search only simple numeric separators:
       - one feature + one threshold
       - two threshold literals combined by AND / OR
     This is exploratory on the same 20 Worlds, not a promoted policy.

No learned model. No terminal-derived feature. No repair.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import math
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


def count_plants(farm,crop):
    n=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if isinstance(tile,dict) and tile.get("kind")=="PLANT" and tile.get("crop")==crop:
                n+=1
    return n


def count_animals(farm,animal):
    n=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if isinstance(tile,dict) and tile.get("animal")==animal:
                n+=1
    return n


def flatten_numeric(prefix,obj,out):
    if isinstance(obj,bool):
        out[prefix]=int(obj)
    elif isinstance(obj,(int,float)) and math.isfinite(float(obj)):
        out[prefix]=float(obj)
    elif isinstance(obj,dict):
        for k,v in sorted(obj.items(),key=lambda kv:str(kv[0])):
            p=f"{prefix}.{k}" if prefix else str(k)
            flatten_numeric(p,v,out)


def pre_features(obs,seat,action):
    self_farm=obs["farms"][seat]
    opp_farm=obs["farms"][1-seat]
    priv=obs["private"]
    market=obs["market"]
    town=obs.get("town",{}) or {}

    f={
        "seat":float(seat),
        "self.cash":float(self_farm.get("money",0) or 0),
        "self.hands":float(len(self_farm.get("hands",[]) or [])),
        "self.hires_today":float(self_farm.get("hires_today",0) or 0),
        "self.land_count":float(len(self_farm.get("unlocked_quadrants",[]) or [])),
        "opp.cash":float(opp_farm.get("money",0) or 0),
        "opp.hands":float(len(opp_farm.get("hands",[]) or [])),
        "opp.hires_today":float(opp_farm.get("hires_today",0) or 0),
        "opp.land_count":float(len(opp_farm.get("unlocked_quadrants",[]) or [])),
        "town.unlocked_shop_count":float(len(town.get("unlocked_shops",[]) or [])),
        "action.market_count":float(len(action.get("market",[]) or [])),
        "action.hand_count":float(len(action.get("hands",[]) or [])),
    }

    for item,val in sorted((priv.get("shed",{}) or {}).items()):
        f[f"self.shed.{item}"]=float(val or 0)
    for item,val in sorted((priv.get("seeds",{}) or {}).items()):
        f[f"self.seeds.{item}"]=float(val or 0)
    for item,val in sorted((market.get("inventory",{}) or {}).items()):
        f[f"market.inventory.{item}"]=float(val or 0)
    for item,val in sorted((market.get("prices",{}) or {}).items()):
        f[f"market.price.{item}"]=float(val or 0)

    for crop in ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON"):
        f[f"self.plants.{crop}"]=float(count_plants(self_farm,crop))
        f[f"opp.plants.{crop}"]=float(count_plants(opp_farm,crop))
    for animal in ("COW","SHEEP","CHICKEN"):
        f[f"self.animals.{animal}"]=float(count_animals(self_farm,animal))
        f[f"opp.animals.{animal}"]=float(count_animals(opp_farm,animal))

    return f


def run_world(seed,seat,mode,capture=False):
    model=load(BASE,f"separator_model_{mode}_{seed}_{seat}_{os.getpid()}")
    opp=load(OPP,f"separator_opp_{mode}_{seed}_{seat}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    captured=None
    while not env.done:
        o0=plain(shared(env,0)); o1=plain(shared(env,1))
        so=o0 if seat==0 else o1
        oo=o1 if seat==0 else o0
        step=int(so.get("step",0) or 0)
        sa=plain(model.agent(so))
        oa=plain(opp.agent(oo))

        if step==TARGET_STEP:
            if sa.get("market",[]) != AB:
                raise RuntimeError(f"unexpected step217 market seed={seed} seat={seat}: {sa.get('market')}")
            if capture and captured is None:
                captured={
                    "features":pre_features(so,seat,sa),
                    "pre_state":so,
                    "original_action":deepcopy(sa),
                    "opponent_action":deepcopy(oa),
                }
            if mode=="swap":
                sa=deepcopy(sa)
                sa["market"]=deepcopy(BA)

        if seat==0:
            env.step([sa,oa])
        else:
            env.step([oa,sa])

    terminal_obs=plain(shared(env,seat))
    terminal=float(terminal_obs["farms"][seat].get("money",0) or 0)
    return terminal,captured


def thresholds(values):
    u=sorted(set(values))
    return [(a+b)/2.0 for a,b in zip(u,u[1:])]


def pred_literal(x,op,t):
    return x<=t if op=="<=" else x>t


def accuracy(preds,labels):
    return sum(int(p==y) for p,y in zip(preds,labels))/len(labels)


def one_feature_rules(rows):
    labels=[r["label"]=="swap" for r in rows]
    feature_names=sorted(set.intersection(*(set(r["features"].keys()) for r in rows)))
    rules=[]
    for name in feature_names:
        vals=[r["features"][name] for r in rows]
        for t in thresholds(vals):
            for op in ("<=",">"):
                lit=[pred_literal(v,op,t) for v in vals]
                for swap_when_true in (True,False):
                    preds=[p if swap_when_true else not p for p in lit]
                    acc=accuracy(preds,labels)
                    rules.append({
                        "feature":name,"op":op,"threshold":t,
                        "swap_when_true":swap_when_true,
                        "correct":int(round(acc*len(rows))),
                        "total":len(rows),"accuracy":acc,
                    })
    rules.sort(key=lambda r:(-r["correct"],r["feature"],r["threshold"]))
    return rules


def two_feature_rules(rows,top_literals=24):
    labels=[r["label"]=="swap" for r in rows]
    # Build unique threshold literals from strong one-dimensional splits.
    base=one_feature_rules(rows)
    literal_specs=[]
    seen=set()
    for r in base:
        spec=(r["feature"],r["op"],r["threshold"])
        if spec not in seen:
            seen.add(spec)
            literal_specs.append(spec)
        if len(literal_specs)>=top_literals:
            break

    out=[]
    for a,b in itertools.combinations(literal_specs,2):
        if a[0]==b[0]:
            continue
        la=[pred_literal(r["features"][a[0]],a[1],a[2]) for r in rows]
        lb=[pred_literal(r["features"][b[0]],b[1],b[2]) for r in rows]
        for combine in ("AND","OR"):
            raw=[(x and y) if combine=="AND" else (x or y) for x,y in zip(la,lb)]
            for swap_when_true in (True,False):
                preds=[p if swap_when_true else not p for p in raw]
                acc=accuracy(preds,labels)
                out.append({
                    "left":{"feature":a[0],"op":a[1],"threshold":a[2]},
                    "right":{"feature":b[0],"op":b[1],"threshold":b[2]},
                    "combine":combine,
                    "swap_when_true":swap_when_true,
                    "correct":int(round(acc*len(rows))),
                    "total":len(rows),"accuracy":acc,
                })
    out.sort(key=lambda r:(-r["correct"],r["combine"],r["left"]["feature"],r["right"]["feature"]))
    return out


def main():
    rows=[]
    for seed in SEEDS:
        for seat in (0,1):
            original,captured=run_world(seed,seat,"original",capture=True)
            swap,_=run_world(seed,seat,"swap",capture=False)
            if captured is None:
                raise RuntimeError("missing pre-state capture")
            delta=swap-original
            label="swap" if delta>0 else "original" if delta<0 else "tie"
            row={
                "seed":seed,
                "seat":seat,
                "features":captured["features"],
                "label":label,
                "original_terminal_self":original,
                "swap_terminal_self":swap,
                "delta_swap_minus_original":delta,
            }
            rows.append(row)
            print("ORDER_SEPARATOR_ROW "+json.dumps(row,separators=(",",":")))

    if any(r["label"]=="tie" for r in rows):
        raise RuntimeError("v0 expected no ties based on prior necessity gate")

    one=one_feature_rules(rows)
    two=two_feature_rules(rows)

    # Check whether exact selected numeric feature snapshots ever conflict in label.
    groups={}
    for r in rows:
        key=json.dumps(r["features"],sort_keys=True,separators=(",",":"))
        groups.setdefault(key,set()).add(r["label"])
    conflicting_exact_groups=sum(len(v)>1 for v in groups.values())

    summary={
        "worlds":len(rows),
        "swap_labels":sum(r["label"]=="swap" for r in rows),
        "original_labels":sum(r["label"]=="original" for r in rows),
        "best_one_feature":one[0] if one else None,
        "best_two_feature":two[0] if two else None,
        "perfect_one_feature_exists":bool(one and one[0]["correct"]==len(rows)),
        "perfect_two_feature_exists":bool(two and two[0]["correct"]==len(rows)),
        "conflicting_identical_numeric_snapshots":conflicting_exact_groups,
    }

    out={
        "schema":"order-separator-probe-v0",
        "question":"Can step217 pre-State observables separate Original-better from Swap-better Worlds?",
        "summary":summary,
        "rows":rows,
        "top_one_feature_rules":one[:20],
        "top_two_feature_rules":two[:20],
        "boundary":{
            "same_20_worlds":True,
            "pre_intervention_features_only":True,
            "label_joined_after_capture":True,
            "exploratory_same_sample":True,
            "no_learned_model":True,
            "no_policy_promotion":True,
            "no_terminal_derived_feature":True,
        }
    }

    Path("order_separator_probe_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("ORDER_SEPARATOR_SUMMARY "+json.dumps(summary,separators=(",",":")))


if __name__=="__main__":
    main()
