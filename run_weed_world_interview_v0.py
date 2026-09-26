#!/usr/bin/env python3
"""World interview at the Day22 h16 stop state.

No new ShortPlan kind is added.

Replay the exact current machine to Day22 h16, then fork the same Official State:
- Baseline: PASS, PASS
- P1: move NORTH from [3,1] to target WEED [3,0], then DIG

Opponent actions are kept identical across both forks at each step.
Observe:
- target tile raw transition
- same-shape Action difference
- PostState generator output, especially establish_plant(CARROT,[3,0])
"""

import copy
import json
import random
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
    semantic_plan_match,
)

ENV_SEED=7001
POLICY_SEED=20260926
MAX_PLAN_STEPS=12
TARGET_TILE=[3,0]
TARGET_CROP="CARROT"


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def semantic_matches(plans,spec):
    return [
        p for p in plans
        if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])
    ]


def tile_at(snapshot,tile=TARGET_TILE):
    raw=snapshot.raw()
    p=raw["player"]
    x,y=tile
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def establish_target(snapshot):
    return [
        p for p in generate_plans(snapshot)
        if p.kind=="establish_plant"
        and p.target.get("crop")==TARGET_CROP
        and list(p.target.get("tile",[]))==TARGET_TILE
    ]


def state_view(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "farmer":list(farm["farmer"]),
        "target_tile":tile_at(snapshot),
        "target_seed_qty":int((raw["private"].get("seeds",{}) or {}).get(TARGET_CROP,0) or 0),
        "candidate_count":len(generate_plans(snapshot)),
        "target_establish_present":bool(establish_target(snapshot)),
    }


def replay_to_h16():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    seq=0

    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        raw=pre.raw()

        if int(raw["day"])==22 and int(raw["hour"])==16:
            if len(generate_plans(pre))!=0:
                raise RuntimeError("Day22 h16 is not candidate-zero in replay")
            if list(raw["farms"][raw["player"]]["farmer"])!=[3,1]:
                raise RuntimeError(f"unexpected farmer position: {raw['farms'][raw['player']]['farmer']}")
            t=tile_at(pre)
            if not (isinstance(t,dict) and t.get("kind")=="WEED"):
                raise RuntimeError(f"target tile is not WEED: {t!r}")
            return env,pre

        plans=generate_plans(pre)
        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not ms:
                active=None
                active_steps=0

        if active is None and plans:
            chosen=plans[rng.randrange(len(plans))]
            seq+=1
            active={"sequence":seq,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms:
                current=ms[0]

        self_bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp_bundle=plain(opponent.agent(s1["observation"]))
        env.step([plain(self_bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        if current is not None and active is not None:
            comp=completion_from_states(current,pre,post)
            active_steps+=1
            if comp.get("complete"):
                active=None
                active_steps=0

    raise RuntimeError("failed to reach Day22 h16")


def same_shape(a,b):
    return (
        isinstance(a,dict) and isinstance(b,dict)
        and set(a)==set(b)=={"farmer","hands","market"}
        and len(a["hands"])==len(b["hands"])
    )


def main():
    env,pre=replay_to_h16()

    if establish_target(pre):
        raise RuntimeError("target establish already present before interview")

    # Turn 1: same PreState. Baseline waits; P1 moves onto adjacent WEED.
    b_env=copy.deepcopy(env)
    p_env=copy.deepcopy(env)

    b0=baseline_pass_bundle(pre)
    p0=baseline_pass_bundle(pre)
    p0["farmer"]=["NORTH"]

    s1=env._Environment__get_shared_state(1)
    opp0=plain(opponent.agent(s1["observation"]))

    b_env.step([copy.deepcopy(b0),copy.deepcopy(opp0)])
    p_env.step([copy.deepcopy(p0),copy.deepcopy(opp0)])

    b1=bind_official_state(b_env.state[0].observation)
    p1=bind_official_state(p_env.state[0].observation)

    if p1.raw()["farms"][p1.raw()["player"]]["farmer"]!=TARGET_TILE:
        raise RuntimeError("P1 did not reach target WEED")

    # Turn 2: keep opponent Action identical across forks again.
    # Baseline continues PASS; P1 applies one Official DIG at the target WEED.
    b1_bundle=baseline_pass_bundle(b1)
    p1_bundle=baseline_pass_bundle(p1)
    p1_bundle["farmer"]=["DIG"]

    b_opp_state=b_env._Environment__get_shared_state(1)
    shared_opp1=plain(opponent.agent(b_opp_state["observation"]))

    b_env.step([copy.deepcopy(b1_bundle),copy.deepcopy(shared_opp1)])
    p_env.step([copy.deepcopy(p1_bundle),copy.deepcopy(shared_opp1)])

    b2=bind_official_state(b_env.state[0].observation)
    p2=bind_official_state(p_env.state[0].observation)

    b_target=tile_at(b2)
    p_target=tile_at(p2)
    target_jobs=establish_target(p2)

    checklist={
        "action_difference_observed":b0!=p0 and b1_bundle!=p1_bundle,
        "pre_and_post_state_observed":True,
        "official_world_effect_observed":b_target!=p_target,
        "baseline_and_p1_compared_same_shape":same_shape(b0,p0) and same_shape(b1_bundle,p1_bundle),
        "next_state_to_current_replan_connected":bool(target_jobs),
    }

    result={
        "schema":"weed-world-interview-v0",
        "purpose":"Ask Official World what one concrete WEED can become before naming a new Plan.",
        "target":{
            "tile":TARGET_TILE,
            "crop_for_next_job_check":TARGET_CROP,
            "selection_reason":"adjacent observed WEED at the actual Day22 h16 stop state; not a priority rule",
        },
        "pre_state":state_view(pre),
        "step_1":{
            "baseline_action":b0,
            "p1_action":p0,
            "same_opponent_action":opp0,
            "baseline_post":state_view(b1),
            "p1_post":state_view(p1),
        },
        "step_2":{
            "baseline_action":b1_bundle,
            "p1_action":p1_bundle,
            "same_opponent_action":shared_opp1,
            "baseline_post":state_view(b2),
            "p1_post":state_view(p2),
        },
        "target_establish_candidates":[p.to_dict() for p in target_jobs],
        "fixed_observation_checklist":checklist,
        "all_fixed_observation_conditions_met":all(checklist.values()),
        "boundary":{
            "no_new_plan_kind_added":True,
            "dig_is_only_an_observed_official_action_here":True,
            "surface_reuse_is_not_yet_a_strategy_claim":True,
            "target_job_appearance_does_not_claim_execution_or_profit":True,
        },
    }

    Path("weed_world_interview_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "pre_target":result["pre_state"]["target_tile"],
        "baseline_post_target":result["step_2"]["baseline_post"]["target_tile"],
        "p1_post_target":result["step_2"]["p1_post"]["target_tile"],
        "p1_seed_qty":result["step_2"]["p1_post"]["target_seed_qty"],
        "p1_candidate_count":result["step_2"]["p1_post"]["candidate_count"],
        "target_establish_present":result["step_2"]["p1_post"]["target_establish_present"],
        "fixed_5":result["all_fixed_observation_conditions_met"],
    },separators=(",",":")))

    if not result["all_fixed_observation_conditions_met"]:
        raise RuntimeError("weed world interview fixed observation checklist failed")


if __name__=="__main__":
    main()
