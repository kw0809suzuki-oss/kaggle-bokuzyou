#!/usr/bin/env python3
"""Local closure: Initial cash -> targeted prepare_for_plant -> establish appears."""

import copy
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"opponents"))

import seyamalam_v21 as opponent
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
)

SEED=7001
TEST_CROP="WHEAT"
TEST_TILE=[0,0]


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def same_shape(a,b):
    return (
        isinstance(a,dict) and isinstance(b,dict)
        and set(a)==set(b)=={"farmer","hands","market"}
        and len(a["hands"])==len(b["hands"])
    )


def target_establish(snapshot):
    return [
        p for p in generate_plans(snapshot)
        if p.kind=="establish_plant"
        and p.target.get("crop")==TEST_CROP
        and list(p.target.get("tile",[]))==TEST_TILE
    ]


def main():
    env=make("kaggriculture",configuration={"seed":SEED},debug=False)
    env.reset(num_agents=2)

    s0=env._Environment__get_shared_state(0)
    s1=env._Environment__get_shared_state(1)
    pre=bind_official_state(s0["observation"])
    raw=pre.raw()
    player=raw["player"]

    pre_plans=generate_plans(pre)
    prep=[
        p for p in pre_plans
        if p.kind=="prepare_for_plant"
        and p.target.get("crop")==TEST_CROP
        and list(p.target.get("tile",[]))==TEST_TILE
    ]
    if len(prep)!=1:
        raise RuntimeError(f"expected exactly one explicit test prepare plan, got {len(prep)}")
    plan=prep[0]

    establish_before=target_establish(pre)
    baseline_bundle=baseline_pass_bundle(pre)
    p1_bundle=project_short_plan(pre,plan)
    opp=plain(opponent.agent(s1["observation"]))

    baseline_env=copy.deepcopy(env)
    p1_env=copy.deepcopy(env)
    baseline_env.step([copy.deepcopy(baseline_bundle),copy.deepcopy(opp)])
    p1_env.step([copy.deepcopy(p1_bundle),copy.deepcopy(opp)])

    baseline_post=bind_official_state(baseline_env._Environment__get_shared_state(0)["observation"])
    p1_post=bind_official_state(p1_env._Environment__get_shared_state(0)["observation"])

    braw=baseline_post.raw()
    praw=p1_post.raw()
    completion=completion_from_states(plan,pre,p1_post)
    establish_after_baseline=target_establish(baseline_post)
    establish_after_p1=target_establish(p1_post)

    checklist={
        "action_difference_observed":baseline_bundle!=p1_bundle,
        "pre_and_post_state_observed":True,
        "official_world_effect_observed":(
            praw["private"]["seeds"].get(TEST_CROP,0)!=braw["private"]["seeds"].get(TEST_CROP,0)
            or praw["farms"][player]["money"]!=braw["farms"][player]["money"]
        ),
        "baseline_and_p1_compared_same_shape":same_shape(baseline_bundle,p1_bundle),
        "next_state_to_current_replan_connected":bool(establish_after_p1),
    }

    result={
        "schema":"prepare-for-plant-local-closure-v0",
        "test_selection_only":{
            "crop":TEST_CROP,
            "tile":TEST_TILE,
            "reason":"explicit local connection test; not Agent priority",
        },
        "pre":{
            "day":raw["day"],
            "hour":raw["hour"],
            "cash":raw["farms"][player]["money"],
            "seed_quantity":raw["private"]["seeds"].get(TEST_CROP,0),
            "target_tile":copy.deepcopy(raw["farms"][player]["tiles"][TEST_TILE[1]][TEST_TILE[0]]),
            "prepare_plan":plan.to_dict(),
            "target_establish_present":bool(establish_before),
            "candidate_count":len(pre_plans),
        },
        "baseline":{
            "action_bundle":baseline_bundle,
            "post_cash":braw["farms"][player]["money"],
            "post_seed_quantity":braw["private"]["seeds"].get(TEST_CROP,0),
            "target_establish_present":bool(establish_after_baseline),
            "candidate_count":len(generate_plans(baseline_post)),
        },
        "p1":{
            "action_bundle":p1_bundle,
            "post_cash":praw["farms"][player]["money"],
            "post_seed_quantity":praw["private"]["seeds"].get(TEST_CROP,0),
            "completion":completion,
            "target_establish_present":bool(establish_after_p1),
            "target_establish_candidates":[p.to_dict() for p in establish_after_p1],
            "candidate_count":len(generate_plans(p1_post)),
        },
        "official_world":{
            "same_pre_state":True,
            "same_opponent_action":opp,
        },
        "fixed_observation_checklist":checklist,
        "all_fixed_observation_conditions_met":all(checklist.values()),
        "acceptance":{
            "prepare_complete":completion["complete"],
            "target_establish_absent_before":not bool(establish_before),
            "target_establish_absent_in_baseline_post":not bool(establish_after_baseline),
            "target_establish_present_in_p1_post":bool(establish_after_p1),
        },
        "boundary":{
            "prepare_success_is_not_plant_success":True,
            "prepare_success_is_not_strength_improvement":True,
        },
    }
    result["acceptance"]["all_pass"]=(
        all(result["acceptance"].values())
        and result["all_fixed_observation_conditions_met"]
    )

    Path("prepare_for_plant_local_closure_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "all_pass":result["acceptance"]["all_pass"],
        "pre_cash":result["pre"]["cash"],
        "p1_post_cash":result["p1"]["post_cash"],
        "cash_delta":result["p1"]["completion"]["cash_delta"],
        "pre_seed":result["pre"]["seed_quantity"],
        "p1_post_seed":result["p1"]["post_seed_quantity"],
        "target_establish_before":result["pre"]["target_establish_present"],
        "target_establish_after_p1":result["p1"]["target_establish_present"],
        "fixed_5":result["all_fixed_observation_conditions_met"],
    },separators=(",",":")))

    if not result["acceptance"]["all_pass"]:
        raise RuntimeError("prepare_for_plant local closure acceptance failed")


if __name__=="__main__":
    main()
