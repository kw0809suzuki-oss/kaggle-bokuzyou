#!/usr/bin/env python3
import copy
import json
import sys
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
SAMPLE_DAYS={4,8,12,16,20,24}
SAMPLE_HOUR=0
FORBIDDEN_KEYS={
    "score","value","rank","priority","action","actions","bundle","action_bundle",
    "productive_tiles","occupied_capacity","empty_capacity",
}


def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x
    if hasattr(x,"items"):
        return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def get_path(raw,path):
    v=raw
    for k in path:
        v=v[k]
    return v


def key_names(x):
    out=[]
    if isinstance(x,dict):
        for k,v in x.items():
            out.append(str(k))
            out.extend(key_names(v))
    elif isinstance(x,list):
        for v in x:
            out.extend(key_names(v))
    return out


def candidate_grounded(raw,p):
    kind=p["kind"]
    t=p["target"]
    player=raw["player"]

    if kind=="deliver_carried_to_shed":
        idx=t["unit_index"]
        inv=raw["private"]["inventories"][idx]
        return all(inv.get(item,0)==qty and qty>0 for item,qty in t["carried_items"].items())

    if kind=="realize_shed_stock_sale":
        item=t["item"]
        qty=t["available_quantity"]
        return raw["private"]["shed"].get(item,0)==qty and qty>0 and item in raw["market"]["prices"]

    if kind=="establish_plant":
        crop=t["crop"]
        x,y=t["tile"]
        return raw["private"]["seeds"].get(crop,0)>0 and raw["farms"][player]["tiles"][y][x] is None

    return False


def main():
    rows=[]
    kind_counts=Counter()

    for seed in SEEDS:
        current.reset_telemetry()
        env=make("kaggriculture",configuration={"seed":seed},debug=False)
        env.reset(num_agents=2)

        while not env.done:
            s0=env._Environment__get_shared_state(0)
            s1=env._Environment__get_shared_state(1)
            obs0=s0["observation"]
            obs1=s1["observation"]
            day=int(obs0.get("day",0) or 0)
            hour=int(obs0.get("hour",0) or 0)

            if day in SAMPLE_DAYS and hour==SAMPLE_HOUR:
                snap=bind_official_state(obs0)
                raw_before=snap.raw()
                hash_before=snap.canonical_hash

                plans1=[p.to_dict() for p in generate_plans(snap)]
                plans2=[p.to_dict() for p in generate_plans(snap)]

                deterministic=(plans1==plans2)
                non_mutating=(snap.canonical_hash==hash_before and snap.raw()==raw_before)
                bounded=(0<=len(plans1)<=3)
                schema_ok=all(
                    set(("plan_schema_version","candidate_id","kind","target","completion","requirements","unknowns","source_paths"))
                    <= set(p)
                    for p in plans1
                )
                source_paths_resolve=True
                grounded=True
                no_forbidden=True
                unique_ids=(len({p["candidate_id"] for p in plans1})==len(plans1))
                unique_kinds=(len({p["kind"] for p in plans1})==len(plans1))

                for p in plans1:
                    kind_counts[p["kind"]]+=1
                    try:
                        for path in p["source_paths"]:
                            get_path(raw_before,path)
                    except Exception:
                        source_paths_resolve=False
                    grounded=grounded and candidate_grounded(raw_before,p)
                    no_forbidden=no_forbidden and not (set(key_names(p)) & FORBIDDEN_KEYS)

                rows.append({
                    "seed":seed,
                    "day":day,
                    "hour":hour,
                    "count":len(plans1),
                    "kinds":[p["kind"] for p in plans1],
                    "plans":plans1,
                    "checks":{
                        "deterministic":deterministic,
                        "non_mutating":non_mutating,
                        "bounded_max_3":bounded,
                        "schema_ok":schema_ok,
                        "source_paths_resolve":source_paths_resolve,
                        "raw_grounded":grounded,
                        "no_score_rank_action_bundle_or_unconfirmed_derived_keys":no_forbidden,
                        "unique_candidate_ids":unique_ids,
                        "unique_candidate_kinds":unique_kinds,
                    },
                })

            a0=plain(current.agent(obs0))
            a1=plain(opponent.agent(obs1))
            env.step([a0,a1])

    all_checks=[
        ok
        for row in rows
        for ok in row["checks"].values()
    ]
    acceptance={
        "samples":len(rows),
        "candidate_kind_counts":dict(kind_counts),
        "states_with_at_least_one_candidate":sum(row["count"]>0 for row in rows),
        "all_candidate_acceptance_pass":all(all_checks),
    }

    out={
        "schema":"short-plan-candidate-generation-v0",
        "purpose":"StateSnapshot -> small raw-grounded ShortPlan candidate set; no scoring, selection, ActionBundle projection, or economic derived fields.",
        "seeds":SEEDS,
        "sample_days":sorted(SAMPLE_DAYS),
        "sample_hour":SAMPLE_HOUR,
        "acceptance":acceptance,
        "rows":rows,
        "boundary":{
            "implemented":[
                "deliver_carried_to_shed candidate",
                "realize_shed_stock_sale candidate",
                "establish_plant candidate",
            ],
            "not_implemented":[
                "plan ranking",
                "plan selection",
                "ActionBundle projection",
                "multi-step Official World rollout",
                "terminal evaluation",
                "productive_tiles",
                "occupied_capacity",
                "empty_capacity",
            ],
        },
    }
    Path("short_plan_candidates_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(acceptance,separators=(",",":")))


if __name__=="__main__":
    main()
