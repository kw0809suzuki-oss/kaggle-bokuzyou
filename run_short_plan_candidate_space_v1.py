#!/usr/bin/env python3
import json, sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as opponent
from plan_generator_entrance_v0 import bind_official_state, generate_plans

SEEDS=[7001,7004,7007]
FORBIDDEN={"score","value","rank","priority","action","actions","bundle","action_bundle",
           "productive_tiles","occupied_capacity","empty_capacity"}

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)

def expected_counts(raw):
    carried=0
    for inv in raw["private"].get("inventories",[]) or []:
        if isinstance(inv,dict) and any(isinstance(q,(int,float)) and q>0 for q in inv.values()):
            carried+=1

    shed=raw["private"].get("shed",{}) or {}
    prices=raw["market"].get("prices",{}) or {}
    sale=sum(1 for item in prices if isinstance(shed.get(item,0),(int,float)) and shed.get(item,0)>0)

    p=raw["player"]; empties=0
    for row in raw["farms"][p].get("tiles",[]) or []:
        empties+=sum(tile is None for tile in row)
    seeds=sum(1 for q in (raw["private"].get("seeds",{}) or {}).values()
              if isinstance(q,(int,float)) and q>0)
    return {
        "deliver_carried_to_shed":carried,
        "realize_shed_stock_sale":sale,
        "establish_plant":seeds*empties,
    }

def names(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            out.append(str(k)); out.extend(names(v))
    elif isinstance(x,list):
        for v in x: out.extend(names(v))
    return out

def grounded(raw,p):
    t=p["target"]; kind=p["kind"]; player=raw["player"]
    if kind=="deliver_carried_to_shed":
        i=t["unit_index"]; inv=raw["private"]["inventories"][i]
        return bool(t["carried_items"]) and all(inv.get(k,0)==q and q>0 for k,q in t["carried_items"].items())
    if kind=="realize_shed_stock_sale":
        item=t["item"]; q=t["available_quantity"]
        return raw["private"]["shed"].get(item,0)==q and q>0 and item in raw["market"]["prices"]
    if kind=="establish_plant":
        crop=t["crop"]; x,y=t["tile"]
        return raw["private"]["seeds"].get(crop,0)>0 and raw["farms"][player]["tiles"][y][x] is None
    return False

def main():
    states=0
    kind_instances=Counter()
    states_with_kind=Counter()
    first_examples={}
    max_candidates=0
    all_checks=True
    mismatch_examples=[]

    for seed in SEEDS:
        current.reset_telemetry()
        env=make("kaggriculture",configuration={"seed":seed},debug=False)
        env.reset(num_agents=2)
        step=0
        while not env.done:
            s0=env._Environment__get_shared_state(0)
            s1=env._Environment__get_shared_state(1)
            obs0=s0["observation"]; obs1=s1["observation"]
            snap=bind_official_state(obs0); raw=snap.raw()
            before=(snap.canonical_hash,snap.raw())
            plans=[p.to_dict() for p in generate_plans(snap)]
            repeat=[p.to_dict() for p in generate_plans(snap)]
            exp=expected_counts(raw)
            got=Counter(p["kind"] for p in plans)

            checks={
                "deterministic": plans==repeat,
                "non_mutating": before==(snap.canonical_hash,snap.raw()),
                "candidate_counts_exact": all(got.get(k,0)==v for k,v in exp.items()) and
                                          all(k in exp for k in got),
                "unique_ids": len({p["candidate_id"] for p in plans})==len(plans),
                "all_raw_grounded": all(grounded(raw,p) for p in plans),
                "no_hidden_policy_fields": all(not (set(names(p)) & FORBIDDEN) for p in plans),
            }
            if not all(checks.values()):
                all_checks=False
                if len(mismatch_examples)<5:
                    mismatch_examples.append({
                        "seed":seed,"step":step,"day":raw["day"],"hour":raw["hour"],
                        "expected":exp,"got":dict(got),"checks":checks
                    })

            states+=1; max_candidates=max(max_candidates,len(plans))
            for kind,count in got.items():
                kind_instances[kind]+=count
                states_with_kind[kind]+=1
                if kind not in first_examples:
                    first_examples[kind]={
                        "seed":seed,"step":step,"day":raw["day"],"hour":raw["hour"],
                        "candidate":next(p for p in plans if p["kind"]==kind),
                        "state_candidate_count":len(plans),
                    }

            a0=plain(current.agent(obs0)); a1=plain(opponent.agent(obs1))
            env.step([a0,a1]); step+=1

    acceptance={
        "states_scanned":states,
        "all_checks_pass":all_checks,
        "candidate_instances_by_kind":dict(kind_instances),
        "states_with_kind":dict(states_with_kind),
        "all_three_kinds_observed":all(k in first_examples for k in (
            "deliver_carried_to_shed","realize_shed_stock_sale","establish_plant")),
        "max_candidates_in_one_state":max_candidates,
        "mismatch_examples":mismatch_examples,
    }
    out={
        "schema":"short-plan-candidate-space-v1",
        "purpose":"Observe complete raw-grounded candidate space without first-match truncation or ranking.",
        "seeds":SEEDS,
        "acceptance":acceptance,
        "first_examples":first_examples,
        "boundary":{
            "generator_returns_all_matching_candidates":True,
            "selection_or_truncation":False,
            "not_implemented":[
                "plan ranking","plan selection","ActionBundle projection",
                "Official World completion test","PostState regeneration comparison",
                "terminal evaluation"
            ]
        }
    }
    Path("short_plan_candidate_space_v1.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(acceptance,separators=(",",":")))

if __name__=="__main__": main()
