#!/usr/bin/env python3
import argparse, copy, json, statistics, subprocess, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import seyamalam_v21 as opponent
import whole_flow_control_agent as whole_flow
import strong_origin

SEEDS=list(range(7001,7011))
EXPANSION_OPS={"BUY_LAND","BUY_SEED","BUY_ANIMAL"}

_policy_stats={}
_trajectory=[]
_seen_snapshots=set()

def reset_local():
    global _policy_stats,_trajectory,_seen_snapshots
    _policy_stats={
        "changed_turns":0,
        "removed_orders":0,
        "return_weight_calls":0,
        "return_weighted_scores":0,
    }
    _trajectory=[]
    _seen_snapshots=set()

def is_expansion(order):
    if not isinstance(order,(list,tuple)) or not order:
        return False
    if order[0] in EXPANSION_OPS:
        return True
    return order[0]=="BUY_PRODUCT" and len(order)>1 and order[1]=="COW"

def farm_snapshot(farm):
    plants=animals=empty=0
    for row in farm.get("tiles",[]):
        for tile in row:
            if tile is None:
                empty+=1
            elif isinstance(tile,dict):
                if tile.get("kind")=="PLANT":
                    plants+=1
                if tile.get("animal"):
                    animals+=1
    return {
        "money":float(farm.get("money",0) or 0),
        "land":len(farm.get("unlocked_quadrants",[]) or []),
        "hands":len(farm.get("hands",[]) or []),
        "empty_tiles":empty,
        "plants":plants,
        "animals":animals,
    }

def record_state(obs):
    day=int(obs.get("day",0) or 0)
    hour=int(obs.get("hour",0) or 0)
    if hour not in (0,6,12,18,23):
        return
    key=(day,hour)
    if key in _seen_snapshots:
        return
    _seen_snapshots.add(key)
    player=int(obs["player"])
    me=obs["farms"][player]
    opp=obs["farms"][1-player]
    private=obs.get("private",{}) or {}
    seeds=private.get("seeds",{}) or {}
    shed=private.get("shed",{}) or {}
    _trajectory.append({
        "day":day,"hour":hour,
        "self":farm_snapshot(me),
        "opponent":farm_snapshot(opp),
        "self_seeds":{c:int(seeds.get(c,0) or 0) for c in ("WHEAT","MELON","STRAWBERRY")},
        "self_shed":{k:int(shed.get(k,0) or 0) for k in ("WHEAT","MELON","STRAWBERRY","MILK","WOOL","EGG","FERTILIZER")},
    })

def baseline_agent(obs):
    record_state(obs)
    return whole_flow.agent(obs)

def p1_agent(obs):
    record_state(obs)
    action=whole_flow.agent(obs)
    if int(obs.get("day",0) or 0)<12:
        return action
    market=list(action.get("market",[]) or [])
    kept=[o for o in market if not is_expansion(o)]
    removed=len(market)-len(kept)
    if removed:
        _policy_stats["changed_turns"]+=1
        _policy_stats["removed_orders"]+=removed
        revised=copy.deepcopy(action)
        revised["market"]=kept
        return revised
    return action

def p2_agent(obs):
    record_state(obs)
    action=whole_flow.agent(obs)
    day=int(obs.get("day",0) or 0)
    if day>=14:
        return action
    me=obs["farms"][int(obs["player"])]
    private=obs.get("private",{}) or {}
    empty=0
    for row in me.get("tiles",[]):
        for tile in row:
            if tile is None:
                empty+=1
    seeds=sum((private.get("seeds",{}) or {}).get(c,0) for c in ("WHEAT","MELON","STRAWBERRY"))
    # Decision surface only:
    # when already-held productive capacity can absorb at least two plantings,
    # prefer using it over issuing new expansion orders.
    if min(empty,seeds)<2:
        return action
    market=list(action.get("market",[]) or [])
    kept=[o for o in market if not is_expansion(o)]
    removed=len(market)-len(kept)
    if removed:
        _policy_stats["changed_turns"]+=1
        _policy_stats["removed_orders"]+=removed
        revised=copy.deepcopy(action)
        revised["market"]=kept
        return revised
    return action

def p3_agent(obs):
    record_state(obs)
    original=strong_origin.counter_crop_weights

    def time_weighted(field,base_price,town_demand):
        scores=dict(original(field,base_price,town_demand))
        remaining=max(1.0,float(field.remaining_days))
        for crop,score in list(scores.items()):
            return_time=float(strong_origin.FIRST_YIELD.get(crop,remaining))
            # Preserve native ranking logic and only add a bounded time-to-return weight.
            fit=max(0.0,min(1.0,(remaining-return_time)/remaining))
            factor=0.75+0.25*fit
            scores[crop]=score*factor
            _policy_stats["return_weighted_scores"]+=1
        _policy_stats["return_weight_calls"]+=1
        return scores

    strong_origin.counter_crop_weights=time_weighted
    try:
        return whole_flow.agent(obs)
    finally:
        strong_origin.counter_crop_weights=original

MODELS={
    "baseline":baseline_agent,
    "p1_early_liquidation":p1_agent,
    "p2_production_utilization":p2_agent,
    "p3_fast_return":p3_agent,
}

def run_one(model,seed):
    reset_local()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([MODELS[model],opponent.agent])
    rewards=[float(s.reward) for s in env.state]
    print(json.dumps({
        "model":model,
        "seed":seed,
        "self":rewards[0],
        "opponent":rewards[1],
        "margin":rewards[0]-rewards[1],
        "steps":len(env.steps),
        "policy_stats":dict(_policy_stats),
        "trajectory":list(_trajectory),
    },separators=(",",":")))

def child(model,seed):
    cp=subprocess.run(
        [sys.executable,__file__,"--model",model,"--seed",str(seed)],
        check=True,capture_output=True,text=True
    )
    return json.loads(cp.stdout.strip().splitlines()[-1])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=list(MODELS))
    p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.model:
        run_one(a.model,a.seed)
        return

    rows=[]
    for model in MODELS:
        for seed in SEEDS:
            row=child(model,seed)
            rows.append(row)
            print(json.dumps({
                "model":model,
                "seed":seed,
                "self":row["self"],
                "opponent":row["opponent"],
                "margin":row["margin"],
                "policy_stats":row["policy_stats"],
            },separators=(",",":")))

    base_by_seed={r["seed"]:r for r in rows if r["model"]=="baseline"}
    summaries={}
    for model in MODELS:
        xs=[r for r in rows if r["model"]==model]
        ds=[r["self"]-base_by_seed[r["seed"]]["self"] for r in xs] if model!="baseline" else [0.0]*len(xs)
        summaries[model]={
            "absolute_mean_self":statistics.mean(r["self"] for r in xs),
            "median_self":statistics.median(r["self"] for r in xs),
            "min_self":min(r["self"] for r in xs),
            "max_self":max(r["self"] for r in xs),
            "mean_margin":statistics.mean(r["margin"] for r in xs),
            "delta_self_vs_baseline":statistics.mean(ds),
            "improved":sum(d>0 for d in ds),
            "worsened":sum(d<0 for d in ds),
            "equal":sum(d==0 for d in ds),
            "policy_changed_turns":sum(r["policy_stats"]["changed_turns"] for r in xs),
            "policy_removed_orders":sum(r["policy_stats"]["removed_orders"] for r in xs),
            "return_weight_calls":sum(r["policy_stats"]["return_weight_calls"] for r in xs),
            "return_weighted_scores":sum(r["policy_stats"]["return_weighted_scores"] for r in xs),
        }

    out={
        "schema":"trajectory-policy-bundles-v1",
        "purpose":"direction screening only",
        "world":"official kaggle_environments kaggriculture",
        "opponent":"Seyamalam pinned 8b8c421e...",
        "baseline":"whole_flow_control_agent",
        "self_source_ref":"observer/wr02-value-carry-reanchor-v0-20260925",
        "seeds":SEEDS,
        "policies":{
            "p1_early_liquidation":"change only Build->Liquidation closure timing: D14 to D12",
            "p2_production_utilization":"change only priority when new expansion competes with already-held empty productive capacity plus on-hand seed",
            "p3_fast_return":"change only crop candidate score by a bounded remaining-time vs first-return-time weight; capacity and closure logic unchanged",
        },
        "summaries":summaries,
        "rows":rows,
        "boundary":[
            "Labels are indexes, not strategy conclusions.",
            "This is a Direction Screening run, not a ranking of policy ideas.",
            "Each policy changes one decision surface only.",
            "Each model/seed runs in a fresh subprocess.",
            "Primary judge is terminal absolute mean self and delta vs baseline.",
            "Trajectory snapshots are stored but should only be analyzed for bundles that move terminal materially.",
            "A losing bundle closes only the tested transformation, not its broader domain."
        ]
    }
    Path("trajectory_policy_bundles_v1.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(summaries,separators=(",",":")))

if __name__=="__main__":
    main()
