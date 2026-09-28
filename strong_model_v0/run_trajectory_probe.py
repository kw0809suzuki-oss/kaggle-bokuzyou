#!/usr/bin/env python3
"""One-window trajectory probe for Strong Model v0 vs Independent.

Observation only. Agents are not modified.
Same seed, seat and pinned Seyamalam opponent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_environments import make

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from strong_model_v0.agent import agent as strong_agent, reset_agent as reset_strong
import astra_flow_independent_distilled_v0 as independent

SEED=92804001
SEAT=0
OUT=Path(__file__).with_name("trajectory_probe_result.json")
OPPONENT_PATH=ROOT/"opponents"/"seyamalam_v21.py"


def plain(v:Any)->Any:
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [plain(x) for x in v]
    if isinstance(v,(str,int,float,bool)) or v is None: return v
    if hasattr(v,"items"): return {str(k):plain(x) for k,x in v.items()}
    return v


def load_opponent():
    spec=importlib.util.spec_from_file_location("trajectory_probe_opponent",OPPONENT_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load opponent")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def reset(obj):
    fn=getattr(obj,"reset_agent",None)
    if callable(fn): fn()


def summary(obs:dict[str,Any])->dict[str,Any]:
    p=int(obs["player"]); farm=obs["farms"][p]; priv=obs["private"]
    plants=Counter(); animals=Counter(); yld=0; empty=0; weeds=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if tile is None:
                empty+=1
            elif isinstance(tile,dict):
                if tile.get("kind")=="PLANT":
                    plants[str(tile.get("crop"))]+=1
                    yld+=int(tile.get("yield_units",0) or 0)
                if tile.get("animal"):
                    animals[str(tile.get("animal"))]+=1
                    yld+=int(tile.get("yield_units",0) or 0)
                if tile.get("kind")=="WEED": weeds+=1
    shed={k:int(v) for k,v in (priv.get("shed",{}) or {}).items() if int(v or 0)}
    seeds={k:int(v) for k,v in (priv.get("seeds",{}) or {}).items() if int(v or 0)}
    carried=Counter()
    for inv in (priv.get("inventories",[]) or []):
        if isinstance(inv,dict):
            for k,v in inv.items():
                if int(v or 0)>0: carried[str(k)]+=int(v)
    return {
        "day":int(obs["day"]),"hour":int(obs["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "land":len(farm.get("unlocked_quadrants",[]) or []),
        "hands":len(farm.get("hands",[]) or []),
        "plants":dict(plants),"plant_total":sum(plants.values()),
        "animals":dict(animals),"animal_total":sum(animals.values()),
        "yield_total":yld,"empty":empty,"weeds":weeds,
        "seed_total":sum(seeds.values()),"seeds":seeds,
        "shed_total":sum(shed.values()),"shed":shed,
        "carried_total":sum(carried.values()),"carried":dict(carried),
    }


def count_action(counter:Counter, action:dict[str,Any]):
    unit_actions=[action.get("farmer",["PASS"])] + list(action.get("hands",[]) or [])
    for a in unit_actions:
        if isinstance(a,list) and a:
            counter[str(a[0])]+=1
    for order in action.get("market",[]) or []:
        if not (isinstance(order,list) and order): continue
        op=str(order[0]); counter[op]+=1
        if op in ("BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL") and len(order)>=3:
            counter[f"{op}:{order[1]}"]+=int(order[2])


def run(label, agent_fn, reset_fn):
    opp=load_opponent(); reset_fn(); reset(opp)
    env=make("kaggriculture",configuration={"seed":SEED},debug=False)
    env.reset(num_agents=2)

    daily=[]
    turns=[]
    day_counts=Counter()
    previous_day=0

    first=plain(env._Environment__get_shared_state(SEAT)["observation"])
    daily.append({"point":"initial","state":summary(first),"actions":{}})

    turn=0
    while not env.done:
        shared_self=env._Environment__get_shared_state(SEAT)
        shared_opp=env._Environment__get_shared_state(1-SEAT)
        obs=plain(shared_self["observation"])
        opp_obs=plain(shared_opp["observation"])

        try:
            action=plain(agent_fn(obs,env.configuration))
        except TypeError:
            action=plain(agent_fn(obs))
        opp_action=plain(opp.agent(opp_obs))

        count_action(day_counts,action)
        if turn < 120:
            turns.append({
                "turn":turn,"pre":summary(obs),"action":action
            })

        actions=[None,None]; actions[SEAT]=action; actions[1-SEAT]=opp_action
        env.step(actions)
        post=plain(env.state[SEAT].observation)

        if int(post["hour"])==0 and int(post["day"])!=previous_day:
            daily.append({
                "point":f"day_{int(post['day'])}_h0",
                "state":summary(post),
                "actions_previous_day":dict(day_counts),
            })
            previous_day=int(post["day"])
            day_counts=Counter()
        turn+=1

    final=plain(env.state[SEAT].observation)
    return {
        "model":label,"seed":SEED,"seat":SEAT,
        "daily":daily,"first_120_turns":turns,
        "final":summary(final),
    }


def main():
    result={
        "schema":"strong-model-v0-trajectory-probe-v0",
        "purpose":"Observe where Strong Model v0 trajectory departs from Independent; no intervention.",
        "independent":run("independent_distilled_v0",independent.agent,independent.reset_agent),
        "strong":run("strong_model_v0",strong_agent,reset_strong),
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    im={x["state"]["day"]:x for x in result["independent"]["daily"]}
    sm={x["state"]["day"]:x for x in result["strong"]["daily"]}
    for day in sorted(set(im)&set(sm)):
        i=im[day]["state"]; s=sm[day]["state"]
        print("DAY "+json.dumps({
            "day":day,
            "ind_cash":i["cash"],"strong_cash":s["cash"],
            "ind_land":i["land"],"strong_land":s["land"],
            "ind_plants":i["plant_total"],"strong_plants":s["plant_total"],
            "ind_animals":i["animal_total"],"strong_animals":s["animal_total"],
            "ind_yield":i["yield_total"],"strong_yield":s["yield_total"],
            "ind_seed":i["seed_total"],"strong_seed":s["seed_total"],
            "ind_shed":i["shed_total"],"strong_shed":s["shed_total"],
            "ind_carried":i["carried_total"],"strong_carried":s["carried_total"],
        },separators=(",",":"),sort_keys=True))

    print("FINAL "+json.dumps({
        "independent":result["independent"]["final"],
        "strong":result["strong"]["final"],
    },separators=(",",":"),sort_keys=True))


if __name__=="__main__":
    main()
