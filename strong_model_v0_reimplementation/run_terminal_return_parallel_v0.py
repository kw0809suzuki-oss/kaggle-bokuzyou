#!/usr/bin/env python3
"""Batch six planned-commitment variants through the 24-turn gate."""
from __future__ import annotations
import importlib.util, json, os, statistics, sys
from pathlib import Path
from typing import Any
from kaggle_environments import make

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from strong_model_v0_reimplementation.agent import agent as strong_agent, reset_agent

SEEDS=[92804001,92804002]
VARIANTS=["baseline","activate","add","shrink","recover","rebalance"]
HORIZON=24
OPPONENT_PATH=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
OUT=Path(__file__).with_name("terminal_return_parallel_v0_result.json")

def load_opponent():
    spec=importlib.util.spec_from_file_location("terminal_return_opponent",OPPONENT_PATH)
    if spec is None or spec.loader is None: raise RuntimeError(str(OPPONENT_PATH))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def money(state,seat):
    obs=state[seat].observation
    farms=obs["farms"] if isinstance(obs,dict) else obs.farms
    farm=farms[seat]
    return float(farm["money"] if isinstance(farm,dict) else farm.money)

def run_one(variant,seed,seat):
    os.environ.pop("STRONG_VARIANT",None) if variant=="baseline" else os.environ.__setitem__("STRONG_VARIANT",variant)
    reset_agent()
    opponent=load_opponent()
    getattr(opponent,"reset_agent",lambda:None)()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    turn_count=0
    for turn in range(HORIZON):
        if env.done: break
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        observations=[s0["observation"],s1["observation"]]
        configs=[env.configuration,env.configuration]
        actions=[None,None]
        actions[seat]=strong_agent(observations[seat],configs[seat])
        actions[1-seat]=opponent.agent(observations[1-seat])
        env.step(actions)
        turn_count += 1
    return {"variant":variant,"seed":seed,"seat":seat,"turns":turn_count,"self_cash":money(env.state,seat),"opp_cash":money(env.state,1-seat)}

def main():
    rows=[]
    for variant in VARIANTS:
        for seed in SEEDS:
            for seat in (0,1):
                row=run_one(variant,seed,seat); rows.append(row); print("ROW "+json.dumps(row,separators=(",",":")))
    summary={}
    for v in VARIANTS:
        vals=[r["self_cash"] for r in rows if r["variant"]==v]
        summary[v]={"n":len(vals),"mean_self_cash":statistics.mean(vals),"min_self_cash":min(vals),"max_self_cash":max(vals)}
    result={"schema":"terminal-return-parallel-v0-24-turn-gate","horizon":HORIZON,"seeds":SEEDS,"variants":VARIANTS,"summary":summary,"rows":rows}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
if __name__=="__main__": main()