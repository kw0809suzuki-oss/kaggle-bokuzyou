#!/usr/bin/env python3
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from kaggle_environments import make
from kaggle_environments.utils import structify

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as teacher

SEEDS=[7001,7004,7007]
SAMPLE_HOURS={0,4,8,12,16,20}
MOVE_HEADS={"NORTH","SOUTH","EAST","WEST"}
TARGET_OPS={"PLANT","PICKUP","SELL","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL"}
MARKET_OPS=["HIRE","BUY_LAND","BUY_SEED","BUY_ANIMAL","BUY_PRODUCT","SELL"]

def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x
    if hasattr(x,"items"):
        return {str(k):plain(v) for k,v in x.items()}
    return repr(x)

def canon(x):
    return json.dumps(plain(x),sort_keys=True,separators=(",",":"),ensure_ascii=False)

def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32-1))

def load_capture(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def current_action(obs):
    return plain(current.agent(structify(copy.deepcopy(obs))))

def teacher_action(obs):
    return plain(teacher.agent(structify(copy.deepcopy(obs))))

def capture(seed):
    seed_rng(seed)
    current.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    records=[]
    step=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        obs0=plain(copy.deepcopy(s0["observation"]))
        obs1=plain(copy.deepcopy(s1["observation"]))
        a0=current_action(obs0)
        a1=teacher_action(obs1)
        records.append({
            "step":step,
            "day":int(obs0.get("day",0) or 0),
            "hour":int(obs0.get("hour",0) or 0),
            "current_state":obs0,
            "teacher_state":obs1,
            "current_observed_action":a0,
            "teacher_observed_action":a1,
        })
        env.step([a0,a1])
        step+=1

    out={
        "seed":seed,
        "terminal_current":float(env.state[0].reward),
        "terminal_teacher":float(env.state[1].reward),
        "records":records,
    }
    print(json.dumps(out,ensure_ascii=False,separators=(",",":")))

def replay(policy, state_key, observed_key, capture_path, seed):
    seed_rng(seed)
    cap=load_capture(capture_path)
    if policy=="current":
        current.reset_telemetry()
        fn=current_action
    else:
        fn=teacher_action

    rows=[]
    exact=0
    for r in cap["records"]:
        predicted=fn(r[state_key])
        observed=r[observed_key]
        ok=canon(predicted)==canon(observed)
        exact+=int(ok)
        rows.append({
            "step":r["step"],"day":r["day"],"hour":r["hour"],
            "observed":observed,"predicted":predicted,"exact":ok,
        })
    print(json.dumps({
        "policy":policy,
        "state_key":state_key,
        "observed_key":observed_key,
        "seed":seed,
        "total":len(rows),
        "exact":exact,
        "exact_rate":exact/max(1,len(rows)),
        "rows":rows,
    },ensure_ascii=False,separators=(",",":")))

def unit_slots(action):
    if not isinstance(action,dict):
        return []
    return [action.get("farmer",["PASS"])] + list(action.get("hands",[]) or [])

def unit_head(a):
    return a[0] if isinstance(a,list) and a else "PASS"

def unit_family(a):
    h=unit_head(a)
    return "MOVE" if h in MOVE_HEADS else h

def unit_target(a):
    if not isinstance(a,list) or len(a)<2:
        return None
    return a[1]

def market_orders(action):
    if not isinstance(action,dict):
        return []
    return list(action.get("market",[]) or [])

def order_op(o):
    return o[0] if isinstance(o,list) and o else None

def order_target(o):
    if not isinstance(o,list) or len(o)<2:
        return None
    return o[1]

def order_qty(o):
    if not isinstance(o,list) or len(o)<3:
        return 1
    try:
        return int(o[2])
    except Exception:
        return None

def count_intersection(a,b):
    ca,cb=Counter(a),Counter(b)
    return sum((ca & cb).values())

def safe_rate(n,d):
    return None if d==0 else n/d

def phase(day):
    if day<10: return "D00-D09"
    if day<20: return "D10-D19"
    return "D20-D29"

def compare_row(teacher_action_obj,current_action_obj):
    ts=unit_slots(teacher_action_obj)
    cs=unit_slots(current_action_obj)
    m=max(len(ts),len(cs))
    slot=[]
    for i in range(m):
        t=ts[i] if i<len(ts) else ["MISSING"]
        c=cs[i] if i<len(cs) else ["MISSING"]
        slot.append({
            "slot":i,
            "teacher":t,
            "current":c,
            "exact":canon(t)==canon(c),
            "family_match":unit_family(t)==unit_family(c),
            "head_match":unit_head(t)==unit_head(c),
            "target_match":unit_target(t)==unit_target(c),
        })

    to=market_orders(teacher_action_obj)
    co=market_orders(current_action_obj)
    t_op=[order_op(o) for o in to]
    c_op=[order_op(o) for o in co]
    t_target=[(order_op(o),order_target(o)) for o in to]
    c_target=[(order_op(o),order_target(o)) for o in co]
    t_exact=[canon(o) for o in to]
    c_exact=[canon(o) for o in co]

    return {
        "full_exact":canon(teacher_action_obj)==canon(current_action_obj),
        "unit_slots":slot,
        "market_exact":canon(to)==canon(co),
        "market_op_match_count":count_intersection(t_op,c_op),
        "market_target_match_count":count_intersection(t_target,c_target),
        "market_exact_order_match_count":count_intersection(t_exact,c_exact),
        "teacher_market_count":len(to),
        "current_market_count":len(co),
    }

def summarize_cross(cap,cross,own_current,own_teacher):
    rows=[]
    sampled=[]
    for r,x in zip(cap["records"],cross["rows"]):
        comp=compare_row(r["teacher_observed_action"],x["predicted"])
        row={
            "step":r["step"],"day":r["day"],"hour":r["hour"],"phase":phase(r["day"]),
            "teacher_action":r["teacher_observed_action"],
            "current_on_teacher_state":x["predicted"],
            "comparison":comp,
        }
        rows.append(row)
        if r["hour"] in SAMPLE_HOURS:
            sampled.append(row)

    summary={
        "seed":cap["seed"],
        "terminal_current":cap["terminal_current"],
        "terminal_teacher":cap["terminal_teacher"],
        "guard":{
            "current_self_replay_exact_rate":own_current["exact_rate"],
            "teacher_self_replay_exact_rate":own_teacher["exact_rate"],
            "current_self_replay_exact":own_current["exact"]==own_current["total"],
            "teacher_self_replay_exact":own_teacher["exact"]==own_teacher["total"],
        },
        "sampled_states":len(sampled),
        "all_turns":len(rows),
        "sample_full_action_exact":0,
        "sample_unit_slot_exact":0,
        "sample_unit_slot_total":0,
        "sample_unit_family_match":0,
        "sample_move_teacher":0,
        "sample_move_direction_match":0,
        "sample_plant_teacher":0,
        "sample_plant_crop_match":0,
        "market_teacher_orders":0,
        "market_current_orders":0,
        "market_op_matches":0,
        "market_target_matches":0,
        "market_exact_order_matches":0,
        "event_metrics":{},
        "phase_metrics":{},
    }

    phase_acc=defaultdict(lambda:Counter())
    event_acc={op:Counter() for op in MARKET_OPS}

    for row in rows:
        comp=row["comparison"]
        ph=row["phase"]
        phase_acc[ph]["turns"]+=1
        phase_acc[ph]["full_exact"]+=int(comp["full_exact"])
        phase_acc[ph]["market_exact"]+=int(comp["market_exact"])

        teacher_orders=market_orders(row["teacher_action"])
        current_orders=market_orders(row["current_on_teacher_state"])
        summary["market_teacher_orders"]+=len(teacher_orders)
        summary["market_current_orders"]+=len(current_orders)
        summary["market_op_matches"]+=comp["market_op_match_count"]
        summary["market_target_matches"]+=comp["market_target_match_count"]
        summary["market_exact_order_matches"]+=comp["market_exact_order_match_count"]

        for op in MARKET_OPS:
            t=[o for o in teacher_orders if order_op(o)==op]
            c=[o for o in current_orders if order_op(o)==op]
            tm=[(order_op(o),order_target(o)) for o in t]
            cm=[(order_op(o),order_target(o)) for o in c]
            event_acc[op]["teacher_turn"]+=int(bool(t))
            event_acc[op]["current_turn"]+=int(bool(c))
            event_acc[op]["both_turn"]+=int(bool(t) and bool(c))
            event_acc[op]["teacher_orders"]+=len(t)
            event_acc[op]["current_orders"]+=len(c)
            event_acc[op]["target_matches"]+=count_intersection(tm,cm)

    for row in sampled:
        comp=row["comparison"]
        summary["sample_full_action_exact"]+=int(comp["full_exact"])
        for s in comp["unit_slots"]:
            summary["sample_unit_slot_total"]+=1
            summary["sample_unit_slot_exact"]+=int(s["exact"])
            summary["sample_unit_family_match"]+=int(s["family_match"])
            th=unit_head(s["teacher"])
            if th in MOVE_HEADS:
                summary["sample_move_teacher"]+=1
                summary["sample_move_direction_match"]+=int(th==unit_head(s["current"]))
            if th=="PLANT":
                summary["sample_plant_teacher"]+=1
                summary["sample_plant_crop_match"]+=int(
                    unit_head(s["current"])=="PLANT" and unit_target(s["current"])==unit_target(s["teacher"])
                )

    summary["sample_full_action_exact_rate"]=safe_rate(summary["sample_full_action_exact"],summary["sampled_states"])
    summary["sample_unit_slot_exact_rate"]=safe_rate(summary["sample_unit_slot_exact"],summary["sample_unit_slot_total"])
    summary["sample_unit_family_match_rate"]=safe_rate(summary["sample_unit_family_match"],summary["sample_unit_slot_total"])
    summary["sample_move_direction_recall"]=safe_rate(summary["sample_move_direction_match"],summary["sample_move_teacher"])
    summary["sample_plant_crop_recall"]=safe_rate(summary["sample_plant_crop_match"],summary["sample_plant_teacher"])
    summary["market_op_recall"]=safe_rate(summary["market_op_matches"],summary["market_teacher_orders"])
    summary["market_target_recall"]=safe_rate(summary["market_target_matches"],summary["market_teacher_orders"])
    summary["market_exact_order_recall"]=safe_rate(summary["market_exact_order_matches"],summary["market_teacher_orders"])

    for op,a in event_acc.items():
        summary["event_metrics"][op]={
            "teacher_turns":a["teacher_turn"],
            "current_turns":a["current_turn"],
            "same_timing_turns":a["both_turn"],
            "timing_recall":safe_rate(a["both_turn"],a["teacher_turn"]),
            "timing_precision":safe_rate(a["both_turn"],a["current_turn"]),
            "teacher_orders":a["teacher_orders"],
            "current_orders":a["current_orders"],
            "target_matches":a["target_matches"],
            "target_recall":safe_rate(a["target_matches"],a["teacher_orders"]),
        }

    for ph,a in phase_acc.items():
        summary["phase_metrics"][ph]={
            "turns":a["turns"],
            "full_action_exact_rate":safe_rate(a["full_exact"],a["turns"]),
            "market_exact_turn_rate":safe_rate(a["market_exact"],a["turns"]),
        }

    return summary,rows

def child(args):
    cp=subprocess.run([sys.executable,__file__,*args],check=True,capture_output=True,text=True)
    return json.loads(cp.stdout.strip().splitlines()[-1])

def aggregate(summaries):
    out={
        "seeds":[s["seed"] for s in summaries],
        "guard_current_self_all_exact":all(s["guard"]["current_self_replay_exact"] for s in summaries),
        "guard_teacher_self_all_exact":all(s["guard"]["teacher_self_replay_exact"] for s in summaries),
        "sampled_states":sum(s["sampled_states"] for s in summaries),
        "all_turns":sum(s["all_turns"] for s in summaries),
    }
    count_fields=[
        "sample_full_action_exact","sample_unit_slot_exact","sample_unit_slot_total",
        "sample_unit_family_match","sample_move_teacher","sample_move_direction_match",
        "sample_plant_teacher","sample_plant_crop_match","market_teacher_orders",
        "market_current_orders","market_op_matches","market_target_matches","market_exact_order_matches",
    ]
    for k in count_fields:
        out[k]=sum(s[k] for s in summaries)
    out["sample_full_action_exact_rate"]=safe_rate(out["sample_full_action_exact"],out["sampled_states"])
    out["sample_unit_slot_exact_rate"]=safe_rate(out["sample_unit_slot_exact"],out["sample_unit_slot_total"])
    out["sample_unit_family_match_rate"]=safe_rate(out["sample_unit_family_match"],out["sample_unit_slot_total"])
    out["sample_move_direction_recall"]=safe_rate(out["sample_move_direction_match"],out["sample_move_teacher"])
    out["sample_plant_crop_recall"]=safe_rate(out["sample_plant_crop_match"],out["sample_plant_teacher"])
    out["market_op_recall"]=safe_rate(out["market_op_matches"],out["market_teacher_orders"])
    out["market_target_recall"]=safe_rate(out["market_target_matches"],out["market_teacher_orders"])
    out["market_exact_order_recall"]=safe_rate(out["market_exact_order_matches"],out["market_teacher_orders"])

    ev={}
    for op in MARKET_OPS:
        a=Counter()
        for s in summaries:
            m=s["event_metrics"][op]
            for k in ("teacher_turns","current_turns","same_timing_turns","teacher_orders","current_orders","target_matches"):
                a[k]+=m[k]
        ev[op]={
            **dict(a),
            "timing_recall":safe_rate(a["same_timing_turns"],a["teacher_turns"]),
            "timing_precision":safe_rate(a["same_timing_turns"],a["current_turns"]),
            "target_recall":safe_rate(a["target_matches"],a["teacher_orders"]),
        }
    out["event_metrics"]=ev
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["capture","replay"])
    ap.add_argument("--seed",type=int)
    ap.add_argument("--policy",choices=["current","teacher"])
    ap.add_argument("--state-key")
    ap.add_argument("--observed-key")
    ap.add_argument("--capture-path")
    args=ap.parse_args()

    if args.mode=="capture":
        capture(args.seed)
        return
    if args.mode=="replay":
        replay(args.policy,args.state_key,args.observed_key,args.capture_path,args.seed)
        return

    summaries=[]
    detail=[]
    with tempfile.TemporaryDirectory() as td:
        for seed in SEEDS:
            cap=child(["--mode","capture","--seed",str(seed)])
            path=Path(td)/f"capture_{seed}.json"
            path.write_text(json.dumps(cap,ensure_ascii=False),encoding="utf-8")

            own_current=child([
                "--mode","replay","--seed",str(seed),"--policy","current",
                "--state-key","current_state","--observed-key","current_observed_action",
                "--capture-path",str(path)
            ])
            own_teacher=child([
                "--mode","replay","--seed",str(seed),"--policy","teacher",
                "--state-key","teacher_state","--observed-key","teacher_observed_action",
                "--capture-path",str(path)
            ])
            cross=child([
                "--mode","replay","--seed",str(seed),"--policy","current",
                "--state-key","teacher_state","--observed-key","teacher_observed_action",
                "--capture-path",str(path)
            ])

            summary,rows=summarize_cross(cap,cross,own_current,own_teacher)
            summaries.append(summary)
            detail.append({"seed":seed,"rows":rows})
            print("SEED "+json.dumps({
                "seed":seed,
                "terminal_current":summary["terminal_current"],
                "terminal_teacher":summary["terminal_teacher"],
                "guard_current":summary["guard"]["current_self_replay_exact_rate"],
                "guard_teacher":summary["guard"]["teacher_self_replay_exact_rate"],
                "sampled_states":summary["sampled_states"],
                "full_exact":summary["sample_full_action_exact_rate"],
                "unit_family":summary["sample_unit_family_match_rate"],
                "unit_exact":summary["sample_unit_slot_exact_rate"],
                "move_direction":summary["sample_move_direction_recall"],
                "plant_crop":summary["sample_plant_crop_recall"],
                "market_op":summary["market_op_recall"],
                "market_target":summary["market_target_recall"],
            },separators=(",",":")))

    agg=aggregate(summaries)
    output={
        "schema":"teacher-forced-strong-state-imitation-v0",
        "question":"When Current is placed on the exact Seyamalam observation sequence, can it emit the same actions without rollout?",
        "current_policy":"whole_flow_control_agent from observer/wr02-value-carry-reanchor-v0-20260925",
        "teacher":"Seyamalam pinned 8b8c421eb10634c756583ce10c75189f50c83a72",
        "world":"Official Kaggriculture",
        "battle_capture":"Current seat0 vs Seyamalam seat1; only actual actions are submitted",
        "teacher_forcing":"All teacher states are fed sequentially to Current so Current's 24-turn history is populated by the teacher trajectory; no Current shadow action affects the World",
        "evaluation":"General action similarity uses six stratified hours per day (0,4,8,12,16,20); sparse market timing metrics use all turns",
        "seeds":SEEDS,
        "per_seed":summaries,
        "aggregate":agg,
        "details":detail,
    }
    Path("teacher_forced_strong_state_imitation_v0.json").write_text(
        json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(agg,separators=(",",":")))

if __name__=="__main__":
    main()
