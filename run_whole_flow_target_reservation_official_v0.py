#!/usr/bin/env python3
import argparse, json, statistics, subprocess, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import seyamalam_v21 as opponent
import whole_flow_control_agent as baseline
import strong_origin_body
import strong_origin_early_target_reservation_v0 as reserved_origin

SEEDS=list(range(7001,7011))

def candidate_agent(obs):
    old=strong_origin_body.strong_origin
    strong_origin_body.strong_origin=reserved_origin
    try:
        return baseline.agent(obs)
    finally:
        strong_origin_body.strong_origin=old

def run_one(model,seed):
    agent=baseline.agent if model=="baseline" else candidate_agent
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([agent,opponent.agent])
    rewards=[float(s.reward) for s in env.state]
    print(json.dumps({
        "model":model,"seed":seed,"self":rewards[0],"opponent":rewards[1],
        "margin":rewards[0]-rewards[1],"steps":len(env.steps)
    },separators=(",",":")))

def child(model,seed):
    cp=subprocess.run([sys.executable,__file__,"--model",model,"--seed",str(seed)],
                      check=True,capture_output=True,text=True)
    return json.loads(cp.stdout.strip().splitlines()[-1])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",choices=["baseline","candidate"])
    p.add_argument("--seed",type=int)
    a=p.parse_args()
    if a.model:
        run_one(a.model,a.seed); return

    rows=[]
    for seed in SEEDS:
        b=child("baseline",seed)
        c=child("candidate",seed)
        row={
            "seed":seed,
            "baseline_self":b["self"],
            "candidate_self":c["self"],
            "delta_self":c["self"]-b["self"],
            "baseline_margin":b["margin"],
            "candidate_margin":c["margin"],
            "delta_margin":c["margin"]-b["margin"],
        }
        rows.append(row)
        print(json.dumps(row,separators=(",",":")))

    summary={
        "baseline_absolute_mean_self":statistics.mean(r["baseline_self"] for r in rows),
        "candidate_absolute_mean_self":statistics.mean(r["candidate_self"] for r in rows),
        "delta_self":statistics.mean(r["delta_self"] for r in rows),
        "baseline_median_self":statistics.median(r["baseline_self"] for r in rows),
        "candidate_median_self":statistics.median(r["candidate_self"] for r in rows),
        "baseline_mean_margin":statistics.mean(r["baseline_margin"] for r in rows),
        "candidate_mean_margin":statistics.mean(r["candidate_margin"] for r in rows),
        "delta_margin":statistics.mean(r["delta_margin"] for r in rows),
        "improved":sum(r["delta_self"]>0 for r in rows),
        "worsened":sum(r["delta_self"]<0 for r in rows),
        "equal":sum(r["delta_self"]==0 for r in rows),
        "baseline_min_self":min(r["baseline_self"] for r in rows),
        "baseline_max_self":max(r["baseline_self"] for r in rows),
        "candidate_min_self":min(r["candidate_self"] for r in rows),
        "candidate_max_self":max(r["candidate_self"] for r in rows),
    }
    out={
        "schema":"whole-flow-target-reservation-official-v0",
        "baseline":"whole_flow_control_agent",
        "candidate":"whole_flow_control_agent + Day0-4 unit target reservation",
        "world":"official kaggle_environments kaggriculture",
        "opponent":"Seyamalam pinned 8b8c421e...",
        "self_source_ref":"observer/wr02-value-carry-reanchor-v0-20260925",
        "seeds":SEEDS,
        "summary":summary,
        "rows":rows,
        "boundary":[
            "Only the frozen Strong Origin Day0-4 work-target reservation differs.",
            "whole_flow controller and D14 logic are identical in both arms.",
            "Each model/seed runs in a fresh subprocess.",
            "Terminal rewards are official env.state rewards.",
            "This run screens terminal value only; reservation firing/state-change confirmation is a later supplemental check."
        ]
    }
    Path("whole_flow_target_reservation_official_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))

if __name__=="__main__":
    main()
