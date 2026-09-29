#!/usr/bin/env python3
"""Run six Return Stage / Work Allocation variants through a real 24-turn World gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import statistics
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from strong_model_v0_reimplementation.agent import agent as strong_agent, reset_agent

SEEDS=[92804001,92804002]
VARIANTS=["baseline","harvest","carry_to_shed","shed_sell","hold_investment","idle_recovery"]
HORIZON=24
OPPONENT_PATH=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
OUT=Path(__file__).with_name("terminal_return_parallel_v0_result.json")


def load_opponent():
    spec=importlib.util.spec_from_file_location("terminal_return_opponent",OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(OPPONENT_PATH))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plain(value):
    if isinstance(value,Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    if hasattr(value,"items"):
        try:
            return {str(k):plain(v) for k,v in value.items()}
        except Exception:
            pass
    if hasattr(value,"__dict__"):
        return {str(k):plain(v) for k,v in vars(value).items() if not str(k).startswith("_")}
    return str(value)


def get_obs(env,seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def world_signature(obs):
    payload=json.dumps(obs,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def worker_surface(obs,seat):
    farm=obs["farms"][seat]
    return {
        "farmer":plain(farm.get("farmer")),
        "hands":plain(farm.get("hands",[]) or []),
    }


def return_surface(obs,seat):
    farm=obs["farms"][seat]
    private=obs.get("private",{}) or {}
    harvestable=0
    productive=0
    for row in farm.get("tiles",[]) or []:
        for tile in row or []:
            if not isinstance(tile,dict):
                continue
            if tile.get("kind")=="PLANT" or tile.get("animal"):
                productive += 1
                harvestable += int(tile.get("yield_units",0) or 0)
    carried=0
    for inv in private.get("inventories",[]) or []:
        if isinstance(inv,dict):
            carried += sum(int(v or 0) for v in inv.values() if isinstance(v,(int,float)))
    shed=private.get("shed",{}) or {}
    shed_units=sum(int(v or 0) for v in shed.values() if isinstance(v,(int,float)))
    return {
        "productive_assets":productive,
        "harvestable_units":harvestable,
        "carry_units":carried,
        "shed_units":shed_units,
        "cash":float(farm.get("money",0) or 0),
    }


def run_one(variant,seed,seat):
    if variant=="baseline":
        os.environ.pop("STRONG_VARIANT",None)
    else:
        os.environ["STRONG_VARIANT"]=variant
    reset_agent()
    opponent=load_opponent()
    getattr(opponent,"reset_agent",lambda:None)()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    trace=[]
    for turn in range(HORIZON):
        if env.done:
            break
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        observations=[s0["observation"],s1["observation"]]
        actions=[None,None]
        # Important: keep the real 720-turn configuration.  The gate stops
        # observation at 24 turns; it must not make the model believe the game
        # itself ends at turn 24.
        actions[seat]=strong_agent(observations[seat],env.configuration)
        actions[1-seat]=opponent.agent(observations[1-seat])
        own_action=plain(actions[seat])
        env.step(actions)

        obs=get_obs(env,seat)
        trace.append({
            "turn":turn+1,
            "action":own_action,
            "world_signature":world_signature(obs),
            "workers":worker_surface(obs,seat),
            "return_stage":return_surface(obs,seat),
        })
    return {
        "variant":variant,
        "seed":seed,
        "seat":seat,
        "turns":len(trace),
        "trace":trace,
    }


def compare_to_baseline(baseline,row):
    bt=baseline["trace"]
    vt=row["trace"]
    n=min(len(bt),len(vt))

    def first_diff(key):
        for i in range(n):
            if bt[i][key] != vt[i][key]:
                return i+1
        if len(bt)!=len(vt):
            return n+1
        return None

    first_action=first_diff("action")
    first_world=first_diff("world_signature")
    first_workers=first_diff("workers")
    first_return=first_diff("return_stage")
    action_changed=first_action is not None
    world_changed=first_world is not None

    if world_changed:
        status="pass_world_changed"
    elif action_changed:
        status="action_only_unresolved"
    else:
        status="no_effect_unresolved"

    baseline_cash=bt[-1]["return_stage"]["cash"] if bt else None
    variant_cash=vt[-1]["return_stage"]["cash"] if vt else None
    cash_delta=(variant_cash-baseline_cash) if baseline_cash is not None and variant_cash is not None else None

    return {
        "variant":row["variant"],
        "seed":row["seed"],
        "seat":row["seat"],
        "turns":row["turns"],
        "gate_status":status,
        "terminal_eligible":world_changed,
        "first_action_divergence_turn":first_action,
        "first_world_divergence_turn":first_world,
        "first_worker_divergence_turn":first_workers,
        "first_return_stage_divergence_turn":first_return,
        "cash_after_24":variant_cash,
        "baseline_cash_after_24":baseline_cash,
        "cash_delta_after_24":cash_delta,
        "trace":vt,
    }


def main():
    raw_runs={}
    for seed in SEEDS:
        for seat in (0,1):
            baseline=run_one("baseline",seed,seat)
            raw_runs[("baseline",seed,seat)]=baseline
            for variant in VARIANTS[1:]:
                raw_runs[(variant,seed,seat)]=run_one(variant,seed,seat)

    rows=[]
    for seed in SEEDS:
        for seat in (0,1):
            baseline=raw_runs[("baseline",seed,seat)]
            rows.append({
                "variant":"baseline",
                "seed":seed,
                "seat":seat,
                "turns":baseline["turns"],
                "gate_status":"baseline",
                "terminal_eligible":False,
                "trace":baseline["trace"],
            })
            for variant in VARIANTS[1:]:
                rows.append(compare_to_baseline(baseline,raw_runs[(variant,seed,seat)]))

    summary={}
    for variant in VARIANTS:
        group=[r for r in rows if r["variant"]==variant]
        if variant=="baseline":
            cash=[r["trace"][-1]["return_stage"]["cash"] for r in group if r["trace"]]
            summary[variant]={
                "n":len(group),
                "mean_cash_after_24":statistics.mean(cash) if cash else None,
            }
            continue
        statuses={}
        for r in group:
            statuses[r["gate_status"]]=statuses.get(r["gate_status"],0)+1
        deltas=[r["cash_delta_after_24"] for r in group if r.get("cash_delta_after_24") is not None]
        summary[variant]={
            "n":len(group),
            "gate_status_counts":statuses,
            "world_changed":sum(bool(r["terminal_eligible"]) for r in group),
            "mean_cash_delta_after_24":statistics.mean(deltas) if deltas else None,
            "terminal_candidates":sum(bool(r["terminal_eligible"]) for r in group),
        }

    result={
        "schema":"terminal-return-parallel-v0-24-turn-world-gate",
        "horizon":HORIZON,
        "seeds":SEEDS,
        "variants":VARIANTS,
        "primary_gate":"observable World State divergence from same-seed/seat baseline",
        "note":"24-turn cash is diagnostic only; terminal adoption is not decided here.",
        "summary":summary,
        "rows":rows,
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))


if __name__=="__main__":
    main()
