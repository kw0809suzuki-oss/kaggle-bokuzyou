#!/usr/bin/env python3
import argparse, importlib, json, statistics, subprocess, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import seyamalam_v21 as opponent

MODELS={
    "strong_origin":"strong_origin",
    "g17":"g17_agent",
    "whole_flow":"whole_flow_control_agent",
    "body_d14":"strong_origin_v2_body_only_v0",
    "wr02":"wr02_same_tile_plant_deconfliction_v0",
}
SEEDS=list(range(7001,7011))

def run_one(model,seed):
    mod=importlib.import_module(MODELS[model])
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([mod.agent,opponent.agent])
    r=[float(s.reward) for s in env.state]
    out={"model":model,"seed":seed,"self":r[0],"opponent":r[1],"margin":r[0]-r[1],"steps":len(env.steps)}
    print(json.dumps(out,separators=(",",":")))

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
    for model in MODELS:
        xs=[r for r in rows if r["model"]==model]
        summaries[model]={
            "mean_self":statistics.mean(r["self"] for r in xs),
            "median_self":statistics.median(r["self"] for r in xs),
            "min_self":min(r["self"] for r in xs),
            "max_self":max(r["self"] for r in xs),
            "mean_margin":statistics.mean(r["margin"] for r in xs),
            "wins":sum(r["margin"]>0 for r in xs),
            "losses":sum(r["margin"]<0 for r in xs),
            "ties":sum(r["margin"]==0 for r in xs),
        }
    ranked=sorted(summaries.items(),key=lambda kv:kv[1]["mean_self"],reverse=True)
    out={
        "schema":"rebench-strong-bodies-official-v0",
        "world":"official kaggle_environments kaggriculture",
        "opponent":"Seyamalam pinned 8b8c421e...",
        "self_source_ref":"observer/wr02-value-carry-reanchor-v0-20260925",
        "seeds":SEEDS,
        "models":MODELS,
        "summaries":summaries,
        "ranked_by_mean_self":[m for m,_ in ranked],
        "rows":rows,
        "boundary":[
            "Historical labels are only retrieval indexes; all scores here are freshly measured.",
            "Each model/seed runs in a fresh subprocess.",
            "Terminal rewards are official env.state rewards.",
            "No model is adopted by this screening run alone."
        ]
    }
    Path("rebench_strong_bodies_official_v0.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SUMMARY "+json.dumps(summaries,separators=(",",":")))
    print("RANKED "+json.dumps(out["ranked_by_mean_self"]))

if __name__=="__main__":
    main()
