#!/usr/bin/env python3
"""Replay the exact prepare-enabled provisional rollout and trace the first completed CARROT plant.

No intervention is added. This is observation only.
"""

import copy
import json
import random
import sys
from collections import Counter
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


def tile_at(snapshot,tile):
    raw=snapshot.raw()
    x,y=tile
    p=raw["player"]
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def target_related_candidates(snapshot,crop,tile):
    out=[]
    for p in generate_plans(snapshot):
        target=dict(p.target)
        if target.get("crop")==crop and list(target.get("tile",[]))==list(tile):
            out.append(p.to_dict())
    return out


def tile_view(snapshot,crop,tile):
    raw=snapshot.raw()
    value=tile_at(snapshot,tile)
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "tile":list(tile),
        "tile_raw":value,
        "is_target_plant":(
            isinstance(value,dict)
            and value.get("kind")=="PLANT"
            and value.get("crop")==crop
        ),
        "related_candidates":target_related_candidates(snapshot,crop,tile),
        "candidate_count":len(generate_plans(snapshot)),
    }


def main():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    sequence=0
    selected_counts=Counter()
    completed_counts=Counter()

    tracked=None
    life=[]
    first_nonplant_after=None
    last_plant=None

    turn=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        plans=generate_plans(pre)

        if active is not None:
            matches=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not matches:
                active=None
                active_steps=0

        if active is None and plans:
            chosen=plans[rng.randrange(len(plans))]
            sequence+=1
            active={"sequence":sequence,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0
            selected_counts[chosen.kind]+=1

        current_plan=None
        if active is not None:
            matches=semantic_matches(plans,active)
            if matches:
                current_plan=matches[0]

        if current_plan is None:
            self_bundle=baseline_pass_bundle(pre)
        else:
            self_bundle=project_short_plan(pre,current_plan)

        opp_bundle=plain(opponent.agent(s1["observation"]))
        env.step([plain(self_bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        completion=None
        if current_plan is not None and active is not None:
            completion=completion_from_states(current_plan,pre,post)
            active_steps+=1

            if (
                tracked is None
                and current_plan.kind=="establish_plant"
                and current_plan.target.get("crop")=="CARROT"
                and completion.get("complete")
            ):
                tracked={
                    "crop":"CARROT",
                    "tile":list(current_plan.target["tile"]),
                    "establish_sequence":active["sequence"],
                    "plant_completed_turn":turn,
                    "plant_completed_pre":tile_view(pre,"CARROT",current_plan.target["tile"]),
                    "plant_completed_post":tile_view(post,"CARROT",current_plan.target["tile"]),
                }
                life.append({
                    "turn":turn,
                    "self_action_bundle":plain(self_bundle),
                    "opponent_action_bundle":opp_bundle,
                    "pre":tracked["plant_completed_pre"],
                    "post":tracked["plant_completed_post"],
                    "event":"plant_completed",
                })

            if completion.get("complete"):
                completed_counts[active["kind"]]+=1
                active=None
                active_steps=0

        # Once the first completed CARROT exists, trace every later Official transition
        # through the first PostState in which the target is no longer that CARROT plant.
        if tracked is not None and turn>tracked["plant_completed_turn"] and first_nonplant_after is None:
            crop=tracked["crop"]; tile=tracked["tile"]
            pre_v=tile_view(pre,crop,tile)
            post_v=tile_view(post,crop,tile)
            life.append({
                "turn":turn,
                "self_action_bundle":plain(self_bundle),
                "opponent_action_bundle":opp_bundle,
                "pre":pre_v,
                "post":post_v,
                "event":"tracked_transition",
            })
            if pre_v["is_target_plant"]:
                last_plant={"turn":turn,"view":pre_v}
            if pre_v["is_target_plant"] and not post_v["is_target_plant"]:
                first_nonplant_after={
                    "turn":turn,
                    "self_action_bundle":plain(self_bundle),
                    "pre":pre_v,
                    "post":post_v,
                }

        turn+=1

    final=bind_official_state(env.state[0].observation)
    rewards=[]
    for st in env.state:
        try: rewards.append(float(st.reward))
        except Exception: rewards.append(None)

    guard={
        "ran_to_terminal":bool(env.done),
        "turns":turn,
        "terminal_self_cash":float(final.raw()["farms"][0]["money"]),
        "official_rewards":rewards,
        "selected_plans_by_kind":dict(selected_counts),
        "completed_plans_by_kind":dict(completed_counts),
        "matches_prior_terminal_cash_1470":float(final.raw()["farms"][0]["money"])==1470.0,
        "first_carrot_found":tracked is not None,
        "first_carrot_became_nonplant":first_nonplant_after is not None,
    }

    result={
        "schema":"first-carrot-life-trace-v0",
        "purpose":"Observe the first completed CARROT plant in the exact prepare-enabled provisional rollout; no intervention.",
        "environment":{"seed":ENV_SEED,"policy_seed":POLICY_SEED},
        "tracked":tracked,
        "life":life,
        "last_plant_state":last_plant,
        "first_nonplant_state":first_nonplant_after,
        "guard":guard,
        "boundary":{
            "observation_only":True,
            "no_new_plan_kind_added":True,
            "no_care_or_harvest_explanation_assumed":True,
        },
    }
    Path("first_carrot_life_trace_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "terminal_self_cash":guard["terminal_self_cash"],
        "first_carrot_found":guard["first_carrot_found"],
        "tile":tracked["tile"] if tracked else None,
        "plant_completed_turn":tracked["plant_completed_turn"] if tracked else None,
        "first_nonplant_turn":first_nonplant_after["turn"] if first_nonplant_after else None,
        "last_plant_raw":last_plant["view"]["tile_raw"] if last_plant else None,
        "first_nonplant_raw":first_nonplant_after["post"]["tile_raw"] if first_nonplant_after else None,
        "matches_prior_terminal_cash_1470":guard["matches_prior_terminal_cash_1470"],
    },separators=(",",":")))


if __name__=="__main__":
    main()
