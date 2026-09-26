#!/usr/bin/env python3
import argparse, copy, importlib, json, statistics, subprocess, sys
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

def is_expansion(order):
    if not isinstance(order,(list,tuple)) or not order: return False
    if order[0] in EXPANSION_OPS: return True
    return order[0]=="BUY_PRODUCT" and len(order)>1 and order[1]=="COW"

_stats={"changed_turns":0,"removed_orders":0,"fast_return_calls":0,"fast_return_shifted_targets":0}

def reset_stats():
    for k in _stats: _stats[k]=0

def p1_agent(obs):
    action=whole_flow.agent(obs)
    if int(obs.get("day",0) or 0) < 12: return action
    market=list(action.get("market",[]) or [])
    kept=[o for o in market if not is_expansion(o)]
    removed=len(market)-len(kept)
    if removed:
        _stats["changed_turns"]+=1
        _stats["removed_orders"]+=removed
        action=copy.deepcopy(action); action["market"]=kept
    return action

def p2_agent(obs):
    action=whole_flow.agent(obs)
    day=int(obs.get("day",0) or 0)
    if day>=14: return action
    me=obs["farms"][int(obs["player"])]
    private=obs.get("private",{}) or {}
    empty=0
    for row in me.get("tiles",[]):
        for tile in row:
            if tile is None: empty+=1
    seeds=sum((private.get("seeds",{}) or {}).get(c,0) for c in ("WHEAT","MELON","STRAWBERRY"))
    convertible=min(empty,seeds)
    if convertible<2: return action
    market=list(action.get("market",[]) or [])
    kept=[o for o in market if not is_expansion(o)]
    removed=len(market)-len(kept)
    if removed:
        _stats["changed_turns"]+=1
        _stats["removed_orders"]+=removed
        action=copy.deepcopy(action); action["market"]=kept
    return action

def p3_agent(obs):
    orig=strong_origin.origin_targets
    def faster_targets(name,day,capacity):
        out=dict(orig(name,day,capacity))
        if day>=14: return out
        slow=max(0,out.get("MELON",0))+max(0,out.get("STRAWBERRY",0))
        shift=int(slow*0.25)
        if shift<=0 and slow>0: shift=1
        if shift<=0: return out
        remaining=shift
        for crop in ("STRAWBERRY","MELON"):
            take=min(remaining,max(0,out.get(crop,0)))
            out[crop]=out.get(crop,0)-take
            out["WHEAT"]=out.get("WHEAT",0)+take
            remaining-=take
            if remaining<=0: break
        moved=shift-remaining
        if moved:
            _stats["fast_return_calls"]+=1
            _stats["fast_return_shifted_targets"]+=moved
        return out
    strong_origin.origin_targets=faster_targets
    try:
        return whole_flow.agent(obs)
    finally:
        strong_origin.origin_targets=orig

MODELS={
    "baseline": whole_flow.agent,
    "p1_early_liquidation": p1_agent,
    "p2_production_utilization": p2_agent,
    "p3_fast_return": p3_agent,
}

def run_one(model,seed):
    reset_stats()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([MODELS[model],opponent.agent])
    r=[float(s.reward) for s in env.state]
    print(json.dumps({
        "model":model,"seed":seed,"self":r[0],"opponent":r[1],"margin":r[0]-r[1],
        "steps":len(env.steps),"policy_stats":dict(_stats)
    },separators=(",",":")))

def child(model,seed):
    cp=subprocess.run([sys.executable,__file__,"--model",model,"--seed",str(seed)],
                      check=True,capture_output=True,text=True)
    return json.loads(cp.stdout.strip().splitlines()[-1])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=list(MODELS))
    p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.model:
        run_one(a.model,a.seed); return

    rows=[]
    for model in MODELS:
        for seed in SEEDS:
            row=child(model,seed)
            rows.append(row)
            print(json.dumps(row,separators=(",",":")))

    summaries={}
    base=[r for r in rows if r["model"]=="baseline"]
    base_by_seed={r["seed"]:r for r in base}
    for model in MODELS:
        xs=[r for r in rows if r["model"]==model]
        deltas=[r["self"]-base_by_seed[r["seed"]]["self"] for r in xs] if model!="baseline" else [0.0]*len(xs)
        summaries[model]={
            "absolute_mean_self":statistics.mean(r["self"] for r in xs),
            "median_self":statistics.median(r["self"] for r in xs),
            "min_self":min(r["self"] for r in xs),
            "max_self":max(r["self"] for r in xs),
            "mean_margin":statistics.mean(r["margin"] for r in xs),
            "delta_self_vs_baseline":statistics.mean(deltas),
            "improved":sum(d>0 for d in deltas),
            "worsened":sum(d<0 for d in deltas),
            "equal":sum(d==0 for d in deltas),
            "policy_changed_turns":sum(r["policy_stats"]["changed_turns"] for r in xs),
            "policy_removed_orders":sum(r["policy_stats"]["removed_orders"] for r in xs),
            "fast_return_calls":sum(r["policy_stats"]["fast_return_calls"] for r in xs),
            "fast_return_shifted_targets":sum(r["policy_stats"]["fast_return_shifted_targets"] for r in xs),
        }

    out={
        "schema":"trajectory-policy-bundles-v0",
        "world":"official kaggle_environments kaggriculture",
        "opponent":"Seyamalam pinned 8b8c421e...",
        "baseline":"whole_flow_control_agent",
        "self_source_ref":"observer/wr02-value-carry-reanchor-v0-20260925",
        "seeds":SEEDS,
        "policies":{
            "p1_early_liquidation":"advance existing D14 expansion closure to D12",
            "p2_production_utilization":"before D14, when >=2 existing empty productive tiles and >=2 on-hand crop seeds, suppress new LAND/SEED/ANIMAL/COW expansion orders",
            "p3_fast_return":"before D14, move 25% of slow-crop target count toward WHEAT while preserving total crop target count",
        },
        "summaries":summaries,
        "rows":rows,
        "boundary":[
            "This is a fresh10 screening of thin trajectory policies, not component attribution.",
            "Each model/seed runs in a fresh subprocess.",
            "Primary judge is terminal absolute mean self and delta vs baseline.",
            "Policy telemetry only verifies that a policy actually changed execution."
        ]
    }
    Path("trajectory_policy_bundles_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(summaries,separators=(",",":")))

if __name__=="__main__":
    main()
