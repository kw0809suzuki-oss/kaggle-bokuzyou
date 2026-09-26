#!/usr/bin/env python3
"""Local closure for the first observed CARROT maintenance gap.

Replay the exact pre-maintenance provisional trajectory through the first CARROT
plant, then fork the same Day0 h9 PreState:
- Baseline: PASS
- P1: explicit maintain_plant_today(CARROT,[0,1])

Observe immediate completion/replan and continue both forks with identical PASS
self bundles and identical opponent bundles to Day1 h0.
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
TARGET_CROP="CARROT"
TARGET_TILE=[0,1]


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


def same_shape(a,b):
    return (
        isinstance(a,dict) and isinstance(b,dict)
        and set(a)==set(b)=={"farmer","hands","market"}
        and len(a["hands"])==len(b["hands"])
    )


def tile_raw(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    x,y=TARGET_TILE
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def maintain_candidates(snapshot):
    return [
        p for p in generate_plans(snapshot)
        if p.kind=="maintain_plant_today"
        and p.target.get("crop")==TARGET_CROP
        and list(p.target.get("tile",[]))==TARGET_TILE
    ]


def replay_to_first_carrot_postplant():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)
    active=None
    active_steps=0
    seq=0

    turn=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])

        # Stop at the exact PostState from prior trace: Day0 h9 with CARROT@[0,1].
        raw=pre.raw()
        if int(raw["day"])==0 and int(raw["hour"])==9:
            t=tile_raw(pre)
            if (
                isinstance(t,dict)
                and t.get("kind")=="PLANT"
                and t.get("crop")==TARGET_CROP
                and int(t.get("planted_day",-1))==0
            ):
                return env,pre,turn

        plans=generate_plans(pre)
        if active is not None:
            matches=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not matches:
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
            if ms: current=ms[0]

        bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp=plain(opponent.agent(s1["observation"]))
        env.step([plain(bundle),opp])
        post=bind_official_state(env.state[0].observation)

        if current is not None and active is not None:
            comp=completion_from_states(current,pre,post)
            active_steps+=1
            if comp.get("complete"):
                active=None
                active_steps=0

        turn+=1

    raise RuntimeError("failed to reach first CARROT post-plant state")


def main():
    env,pre,turn=replay_to_first_carrot_postplant()
    pre_tile=tile_raw(pre)
    plans=maintain_candidates(pre)
    if len(plans)!=1:
        raise RuntimeError(f"expected one target maintenance candidate, got {len(plans)}")
    plan=plans[0]

    baseline_bundle=baseline_pass_bundle(pre)
    p1_bundle=project_short_plan(pre,plan)
    s1=env._Environment__get_shared_state(1)
    opp=plain(opponent.agent(s1["observation"]))

    b_env=copy.deepcopy(env)
    p_env=copy.deepcopy(env)
    b_env.step([copy.deepcopy(baseline_bundle),copy.deepcopy(opp)])
    p_env.step([copy.deepcopy(p1_bundle),copy.deepcopy(opp)])

    b_post=bind_official_state(b_env.state[0].observation)
    p_post=bind_official_state(p_env.state[0].observation)
    completion=completion_from_states(plan,pre,p_post)

    immediate={
        "pre":{"day":pre.raw()["day"],"hour":pre.raw()["hour"],"tile":pre_tile,
               "maintenance_present":bool(maintain_candidates(pre))},
        "baseline":{"bundle":baseline_bundle,"post_tile":tile_raw(b_post),
                    "maintenance_present":bool(maintain_candidates(b_post))},
        "p1":{"bundle":p1_bundle,"post_tile":tile_raw(p_post),
              "completion":completion,
              "maintenance_present":bool(maintain_candidates(p_post))},
        "same_opponent_action":opp,
    }

    # Continue both forks to Day1 h0. Self is PASS in both. Opponent action is
    # computed from baseline and applied identically to both forks each turn.
    continuation=[]
    while True:
        b_snap=bind_official_state(b_env.state[0].observation)
        p_snap=bind_official_state(p_env.state[0].observation)
        braw=b_snap.raw()
        if int(braw["day"])>=1 and int(braw["hour"])==0:
            break

        b_opp_state=b_env._Environment__get_shared_state(1)
        shared_opp=plain(opponent.agent(b_opp_state["observation"]))
        b_self=baseline_pass_bundle(b_snap)
        p_self=baseline_pass_bundle(p_snap)
        b_env.step([copy.deepcopy(b_self),copy.deepcopy(shared_opp)])
        p_env.step([copy.deepcopy(p_self),copy.deepcopy(shared_opp)])

        continuation.append({
            "baseline_post":{"day":b_env.state[0].observation.day,
                             "hour":b_env.state[0].observation.hour,
                             "tile":plain(tile_raw(bind_official_state(b_env.state[0].observation)))},
            "p1_post":{"day":p_env.state[0].observation.day,
                       "hour":p_env.state[0].observation.hour,
                       "tile":plain(tile_raw(bind_official_state(p_env.state[0].observation)))},
        })

    b_day1=bind_official_state(b_env.state[0].observation)
    p_day1=bind_official_state(p_env.state[0].observation)
    b_tile=tile_raw(b_day1)
    p_tile=tile_raw(p_day1)

    baseline_weed=isinstance(b_tile,dict) and b_tile.get("kind")=="WEED"
    p1_survives=(
        isinstance(p_tile,dict)
        and p_tile.get("kind")=="PLANT"
        and p_tile.get("crop")==TARGET_CROP
        and int(p_tile.get("planted_day",-1))==0
    )

    checklist={
        "action_difference_observed":baseline_bundle!=p1_bundle,
        "pre_and_post_state_observed":True,
        "official_world_effect_observed":(
            immediate["baseline"]["post_tile"]!=immediate["p1"]["post_tile"]
            and b_tile!=p_tile
        ),
        "baseline_and_p1_compared_same_shape":same_shape(baseline_bundle,p1_bundle),
        "next_state_to_current_replan_connected":(
            immediate["baseline"]["maintenance_present"]
            and not immediate["p1"]["maintenance_present"]
        ),
    }

    result={
        "schema":"maintain-first-carrot-local-closure-v0",
        "replay_guard":{
            "reached_turn":turn,
            "expected_day_hour":[0,9],
            "target":[TARGET_CROP,TARGET_TILE],
        },
        "plan":plan.to_dict(),
        "immediate":immediate,
        "day1_h0":{
            "baseline_tile":b_tile,
            "p1_tile":p_tile,
            "baseline_became_weed":baseline_weed,
            "p1_same_plant_survives":p1_survives,
        },
        "fixed_observation_checklist":checklist,
        "acceptance":{
            "short_plan_completion_observed":bool(completion.get("complete")),
            "same_maintenance_candidate_disappears_after_completion":not immediate["p1"]["maintenance_present"],
            "baseline_candidate_remains_after_pass":immediate["baseline"]["maintenance_present"],
            "baseline_is_weed_at_day1_h0":baseline_weed,
            "p1_plant_survives_day_boundary":p1_survives,
            "fixed_5":all(checklist.values()),
        },
        "boundary":{
            "maintenance_completion_is_not_maturity":True,
            "survival_is_not_harvest_or_profit":True,
            "local_intervention_only":True,
        },
    }
    result["acceptance"]["all_pass"]=all(result["acceptance"].values())

    Path("maintain_first_carrot_local_closure_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "all_pass":result["acceptance"]["all_pass"],
        "pre_tile":pre_tile,
        "baseline_immediate":immediate["baseline"]["post_tile"],
        "p1_immediate":immediate["p1"]["post_tile"],
        "baseline_day1":b_tile,
        "p1_day1":p_tile,
        "fixed_5":result["acceptance"]["fixed_5"],
    },separators=(",",":")))

    if not result["acceptance"]["all_pass"]:
        raise RuntimeError("maintenance local closure failed")


if __name__=="__main__":
    main()
