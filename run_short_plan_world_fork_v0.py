#!/usr/bin/env python3
import copy
import hashlib
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as opponent

SEEDS=[7001,7004,7007]
SAMPLE_DAYS={4,8,12,16,20,24}
SAMPLE_HOUR=0

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

def stable_hash(x):
    return hashlib.sha256(
        json.dumps(plain(x),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    ).hexdigest()

def self_snapshot(env):
    obs=env.state[0].observation
    farm=plain(obs.farms[0])
    private=plain(obs.private)
    plants={}
    animals={}
    empty=0
    for row in farm.get("tiles",[]):
        for tile in row:
            if tile is None:
                empty+=1
            if isinstance(tile,dict):
                if tile.get("kind")=="PLANT":
                    c=tile.get("crop")
                    plants[c]=plants.get(c,0)+1
                if tile.get("animal"):
                    a=tile.get("animal")
                    animals[a]=animals.get(a,0)+1
    return {
        "day":int(obs.day),
        "hour":int(obs.hour),
        "cash":float(farm.get("money",0) or 0),
        "land_count":len(farm.get("unlocked_quadrants",[]) or []),
        "hands_count":len(farm.get("hands",[]) or []),
        "farmer":plain(farm.get("farmer")),
        "hands":plain(farm.get("hands",[]) or []),
        "seeds":plain(private.get("seeds",{}) or {}),
        "shed":plain(private.get("shed",{}) or {}),
        "inventories":plain(private.get("inventories",[]) or []),
        "plants":plants,
        "animals":animals,
        "empty_tiles":empty,
        "full_hash":stable_hash({
            "farm":farm,
            "private":private,
            "market":plain(obs.market),
            "town":plain(obs.town),
            "day":int(obs.day),
            "hour":int(obs.hour),
        }),
    }

def delta(after,before):
    out={
        "cash":after["cash"]-before["cash"],
        "land_count":after["land_count"]-before["land_count"],
        "hands_count":after["hands_count"]-before["hands_count"],
        "empty_tiles":after["empty_tiles"]-before["empty_tiles"],
    }
    for group in ("seeds","shed","plants","animals"):
        keys=sorted(set(before.get(group,{}))|set(after.get(group,{})))
        vals={}
        for k in keys:
            b=before.get(group,{}).get(k,0)
            a=after.get(group,{}).get(k,0)
            if isinstance(a,(int,float)) and isinstance(b,(int,float)):
                d=a-b
                if d:
                    vals[k]=d
        out[group]=vals
    return out

def all_pass(obs):
    me=obs["farms"][obs["player"]]
    return {
        "farmer":["PASS"],
        "hands":[["PASS"] for _ in (me.get("hands",[]) or [])],
        "market":[],
    }

def unit_only(action):
    out=copy.deepcopy(action)
    out["market"]=[]
    return out

def fork_step(env,self_action,opp_action):
    fork=copy.deepcopy(env)
    fork.step([copy.deepcopy(self_action),copy.deepcopy(opp_action)])
    return fork

def main():
    samples=[]
    guard_total=0
    guard_match=0

    for seed in SEEDS:
        current.reset_telemetry()
        env=make("kaggriculture",configuration={"seed":seed},debug=False)
        env.reset(num_agents=2)
        step=0

        while not env.done:
            s0=env._Environment__get_shared_state(0)
            s1=env._Environment__get_shared_state(1)
            obs0=s0["observation"]
            obs1=s1["observation"]

            current_action=plain(current.agent(obs0))
            opponent_action=plain(opponent.agent(obs1))

            day=int(obs0.get("day",0) or 0)
            hour=int(obs0.get("hour",0) or 0)
            take=(day in SAMPLE_DAYS and hour==SAMPLE_HOUR)

            if take:
                pre=self_snapshot(env)

                control_fork=fork_step(env,current_action,opponent_action)
                unit_fork=fork_step(env,unit_only(current_action),opponent_action)
                noop_fork=fork_step(env,all_pass(obs0),opponent_action)

                control_post=self_snapshot(control_fork)
                unit_post=self_snapshot(unit_fork)
                noop_post=self_snapshot(noop_fork)

                samples.append({
                    "seed":seed,
                    "step":step,
                    "day":day,
                    "hour":hour,
                    "pre":pre,
                    "bundles":{
                        "current_control":{
                            "action":current_action,
                            "post":control_post,
                            "delta":delta(control_post,pre),
                        },
                        "same_units_no_market":{
                            "action":unit_only(current_action),
                            "post":unit_post,
                            "delta":delta(unit_post,pre),
                        },
                        "all_pass_no_market":{
                            "action":all_pass(obs0),
                            "post":noop_post,
                            "delta":delta(noop_post,pre),
                        },
                    },
                })

            env.step([current_action,opponent_action])

            if take:
                guard_total+=1
                actual_post=self_snapshot(env)
                samples[-1]["actual_post"]=actual_post
                ok=(actual_post["full_hash"]==samples[-1]["bundles"]["current_control"]["post"]["full_hash"])
                samples[-1]["control_fork_matches_actual_official_world"]=ok
                guard_match+=int(ok)

            step+=1

    out={
        "schema":"short-plan-world-fork-v0",
        "purpose":"foundation only: same State -> whole ActionBundle -> forked Official World -> comparable NextState",
        "not_a_policy_test":True,
        "current_role":"control bundle only; Current decision logic is not used as the prototype planner",
        "candidate_note":"same_units_no_market and all_pass_no_market are mechanical fork checks, not proposed strategies",
        "seeds":SEEDS,
        "sample_days":sorted(SAMPLE_DAYS),
        "sample_hour":SAMPLE_HOUR,
        "guard":{
            "samples":guard_total,
            "control_fork_matches_actual":guard_match,
            "all_match":guard_total==guard_match,
        },
        "samples":samples,
    }
    Path("short_plan_world_fork_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "samples":guard_total,
        "control_fork_matches_actual":guard_match,
        "all_match":guard_total==guard_match,
        "seeds":SEEDS,
    },separators=(",",":")))

if __name__=="__main__":
    main()
