#!/usr/bin/env python3
"""Adaptive Replay step72->73 Engine Transition Probe v0.

Question:
    Does the seed04/seed05 town.unlocked_shops[0] difference observed at
    step72 mechanically account for the market differences first observed
    at step73 under the pinned official engine?

Scope:
    Seyamalam v21, subject seat0
    positive seed 93803004
    negative seed 93803005
    bodies: frozen DECEM Replay and Contract Runtime v0

Observe only. No new Guard, Recovery, or policy logic.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

ROOT=Path(__file__).resolve().parent
MODELS={
    "baseline":ROOT/"decem_replay_distilled_157026_v0.py",
    "candidate":ROOT/"adaptive_replay_contract_runtime_v0.py",
}
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS={"positive":93803004,"negative":93803005}


def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x
    if hasattr(x,"items"):
        return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):
        return [plain(v) for v in x]
    return x


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"):
        mod.reset_agent()
    return mod


def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]


def run_to_73(seed,model_path,tag):
    model=load(model_path,f"{tag}_model_{seed}_{os.getpid()}")
    opp=load(OPPONENT,f"{tag}_opp_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    out={}
    while not env.done:
        obs0=plain(shared(env,0))
        obs1=plain(shared(env,1))
        step=int(obs0.get("step",0) or 0)
        a0=plain(model.agent(obs0))
        a1=plain(opp.agent(obs1))

        if step==72:
            out["pre72"]={
                "town":plain(obs0["town"]),
                "market":plain(obs0["market"]),
                "farms":plain(obs0["farms"]),
                "private":plain(obs0["private"]),
            }
            out["subject_action72"]=a0
            out["opponent_action72"]=a1

        env.step([a0,a1])

        if step==72:
            obs73=plain(shared(env,0))
            out["post73"]={
                "town":plain(obs73["town"]),
                "market":plain(obs73["market"]),
                "farms":plain(obs73["farms"]),
                "private":plain(obs73["private"]),
            }
            break
    return out


def dict_delta(a,b):
    keys=sorted(set(a)|set(b))
    return {k:b.get(k,0)-a.get(k,0) for k in keys if b.get(k)!=a.get(k)}


def shop_consumption(shop_name):
    products=kg.SHOPS[shop_name]
    multiplier=2 if len(products)==1 else 1
    out={}
    for item in products:
        out[item]=out.get(item,0)+multiplier
    return out


def expected_seed_difference_from_shop(pos_shop,neg_shop):
    # post_pos - post_neg caused by inventory subtraction:
    # (-consume_pos) - (-consume_neg) = consume_neg - consume_pos
    cp=shop_consumption(pos_shop)
    cn=shop_consumption(neg_shop)
    items=sorted(set(cp)|set(cn))
    return {item:cn.get(item,0)-cp.get(item,0) for item in items if cn.get(item,0)!=cp.get(item,0)}


def main():
    cases={}
    for model_name,path in MODELS.items():
        cases[model_name]={}
        for label,seed in SEEDS.items():
            cases[model_name][label]=run_to_73(seed,path,f"{model_name}_{label}")

    results={}
    for model_name, pair in cases.items():
        pos=pair["positive"]
        neg=pair["negative"]

        pre_market_equal=pos["pre72"]["market"]==neg["pre72"]["market"]
        subject_action_equal=pos["subject_action72"]==neg["subject_action72"]
        opponent_action_equal=pos["opponent_action72"]==neg["opponent_action72"]

        pos_shops=pos["pre72"]["town"]["unlocked_shops"]
        neg_shops=neg["pre72"]["town"]["unlocked_shops"]
        pos_shop=pos_shops[0] if pos_shops else None
        neg_shop=neg_shops[0] if neg_shops else None

        pre_inv_pos=pos["pre72"]["market"]["inventory"]
        pre_inv_neg=neg["pre72"]["market"]["inventory"]
        post_inv_pos=pos["post73"]["market"]["inventory"]
        post_inv_neg=neg["post73"]["market"]["inventory"]
        post_price_pos=pos["post73"]["market"]["prices"]
        post_price_neg=neg["post73"]["market"]["prices"]

        observed_post_inventory_diff={
            item:post_inv_pos[item]-post_inv_neg[item]
            for item in sorted(post_inv_pos)
            if post_inv_pos[item]!=post_inv_neg[item]
        }
        observed_post_price_diff={
            item:post_price_pos[item]-post_price_neg[item]
            for item in sorted(post_price_pos)
            if post_price_pos[item]!=post_price_neg[item]
        }

        expected={}
        if pos_shop is not None and neg_shop is not None:
            expected=expected_seed_difference_from_shop(pos_shop,neg_shop)

        results[model_name]={
            "pre72_market_equal":pre_market_equal,
            "step72_subject_action_equal":subject_action_equal,
            "step72_opponent_action_equal":opponent_action_equal,
            "positive_shop":pos_shop,
            "negative_shop":neg_shop,
            "positive_shop_products":None if pos_shop is None else kg.SHOPS[pos_shop],
            "negative_shop_products":None if neg_shop is None else kg.SHOPS[neg_shop],
            "positive_shop_consumption":None if pos_shop is None else shop_consumption(pos_shop),
            "negative_shop_consumption":None if neg_shop is None else shop_consumption(neg_shop),
            "expected_post_inventory_seed_diff_from_shop_only":expected,
            "observed_post_inventory_seed_diff":observed_post_inventory_diff,
            "observed_post_price_seed_diff":observed_post_price_diff,
            "expected_inventory_diff_matches_observed":expected==observed_post_inventory_diff,
            "step72_pre_inventory_seed_diff":dict_delta(pre_inv_neg,pre_inv_pos),
            "positive_transition":{
                "pre_inventory":pre_inv_pos,
                "post_inventory":post_inv_pos,
                "post_prices":post_price_pos,
                "subject_action":pos["subject_action72"],
                "opponent_action":pos["opponent_action72"],
            },
            "negative_transition":{
                "pre_inventory":pre_inv_neg,
                "post_inventory":post_inv_neg,
                "post_prices":post_price_neg,
                "subject_action":neg["subject_action72"],
                "opponent_action":neg["opponent_action72"],
            },
        }

    summary={
        "engine_order":[
            "_process_market(state, env)",
            "_town_consume(env, state, step)",
            "_refresh_prices(market) inside _town_consume"
        ],
        "step72_shop_consume_fires":72 % max(1,int(getattr(make("kaggriculture").configuration,"townShopSellInterval",4))) == 0,
        "step72_center_consume_fires":72 % max(1,int(getattr(make("kaggriculture").configuration,"townCenterSellInterval",24))) == 0,
        "results":results,
    }

    print("ENGINE_72_73_SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_replay_72_73_engine_transition_v0.json").write_text(
        json.dumps({
            "schema":"adaptive-replay-72-73-engine-transition-v0",
            "engine_commit":"d7729da06cc1382eb742d6980dc3180aa85caa28",
            "question":"Does the step72 town shop difference mechanically account for the step73 market difference?",
            "summary":summary,
            "boundary":{
                "no_new_guard":True,
                "no_causal_claim_beyond_engine_transition":True,
                "interpretation_limit":"Matching shop-consumption arithmetic establishes the engine mapping for this transition; it does not establish why one downstream terminal outcome is better."
            }
        },ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8"
    )


if __name__=="__main__":
    main()
