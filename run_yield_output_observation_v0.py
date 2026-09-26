#!/usr/bin/env python3
"""Observe Yield -> ? at the exact Day22 h4 STRAWBERRY state.

No new ShortPlan kind is added here.

Replay the same seed/opponent/provisional policy used by the maintain-enabled
terminal rollout until the exact observed STRAWBERRY state:
  Day22 h4, tile [3,1], yield_units == 4.

Then fork the same PreState with the same opponent action:
  Baseline: PASS
  P1:       HARVEST

Observe only what Official World actually changes:
- target tile raw
- active farmer inventory
- shed
- cash
- generated candidates in the next State

This probe does not yet name or add a new Plan kind.
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
TARGET_CROP="STRAWBERRY"
TARGET_TILE=[3,1]
TARGET_DAY=22
TARGET_HOUR=4
TARGET_YIELD=4


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


def target_tile(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    x,y=TARGET_TILE
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def candidate_summary(snapshot):
    plans=generate_plans(snapshot)
    counts={}
    for p in plans:
        counts[p.kind]=counts.get(p.kind,0)+1
    relevant=[]
    for p in plans:
        t=dict(p.target)
        if (
            (t.get("crop")==TARGET_CROP and list(t.get("tile",[]))==TARGET_TILE)
            or (
                p.kind=="deliver_carried_to_shed"
                and TARGET_CROP in dict(t.get("carried_items",{}))
            )
            or (
                p.kind=="realize_shed_stock_sale"
                and t.get("item")==TARGET_CROP
            )
        ):
            relevant.append(p.to_dict())
    return {
        "count":len(plans),
        "kind_counts":counts,
        "relevant":relevant,
    }


def state_view(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    invs=raw["private"].get("inventories",[]) or []
    farmer_inv=copy.deepcopy(invs[0] if invs else {})
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "farmer_position":list(farm["farmer"]),
        "farmer_inventory":farmer_inv,
        "target_tile":target_tile(snapshot),
        "target_carried_quantity":farmer_inv.get(TARGET_CROP,0),
        "target_shed_quantity":raw["private"].get("shed",{}).get(TARGET_CROP,0),
        "candidates":candidate_summary(snapshot),
    }


def replay_to_target():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    sequence=0
    turn=0

    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        raw=pre.raw()

        if int(raw["day"])==TARGET_DAY and int(raw["hour"])==TARGET_HOUR:
            tile=target_tile(pre)
            if not (
                isinstance(tile,dict)
                and tile.get("kind")=="PLANT"
                and tile.get("crop")==TARGET_CROP
                and int(tile.get("yield_units",0))==TARGET_YIELD
            ):
                raise RuntimeError(f"target day/hour reached with unexpected tile: {tile!r}")
            if list(raw["farms"][raw["player"]]["farmer"])!=TARGET_TILE:
                raise RuntimeError(
                    f"farmer not on target tile at observed state: "
                    f"{raw['farms'][raw['player']]['farmer']}"
                )
            return env,pre,turn,plain(active),active_steps,rng.getstate()

        plans=generate_plans(pre)

        if active is not None:
            matches=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not matches:
                active=None
                active_steps=0

        if active is None and plans:
            chosen=plans[rng.randrange(len(plans))]
            sequence+=1
            active={
                "sequence":sequence,
                "kind":chosen.kind,
                "target":plain(chosen.target),
            }
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

        turn+=1

    raise RuntimeError("target state not reached")


def main():
    env,pre,turn,active,active_steps,rng_state=replay_to_target()

    baseline_bundle=baseline_pass_bundle(pre)
    p1_bundle=baseline_pass_bundle(pre)
    p1_bundle["farmer"]=["HARVEST"]

    s1=env._Environment__get_shared_state(1)
    opp_bundle=plain(opponent.agent(s1["observation"]))

    b_env=copy.deepcopy(env)
    p_env=copy.deepcopy(env)
    b_env.step([copy.deepcopy(baseline_bundle),copy.deepcopy(opp_bundle)])
    p_env.step([copy.deepcopy(p1_bundle),copy.deepcopy(opp_bundle)])

    b_post=bind_official_state(b_env.state[0].observation)
    p_post=bind_official_state(p_env.state[0].observation)

    pre_view=state_view(pre)
    b_view=state_view(b_post)
    p_view=state_view(p_post)

    # The next State is regenerated from scratch. We do not add a new Plan kind;
    # we simply record what the current generator can now see.
    p_delivery=[
        x for x in generate_plans(p_post)
        if x.kind=="deliver_carried_to_shed"
        and TARGET_CROP in dict(x.target.get("carried_items",{}))
    ]

    checklist={
        "action_difference_observed":baseline_bundle!=p1_bundle,
        "pre_and_post_state_observed":True,
        "official_world_effect_observed":(
            b_view["target_tile"]!=p_view["target_tile"]
            or b_view["farmer_inventory"]!=p_view["farmer_inventory"]
            or b_view["target_shed_quantity"]!=p_view["target_shed_quantity"]
            or b_view["cash"]!=p_view["cash"]
        ),
        "baseline_and_p1_compared_same_shape":(
            set(baseline_bundle)==set(p1_bundle)=={"farmer","hands","market"}
            and len(baseline_bundle["hands"])==len(p1_bundle["hands"])
        ),
        "next_state_to_current_replan_connected":(
            p_view["candidates"]["count"]==len(generate_plans(p_post))
        ),
    }

    result={
        "schema":"yield-output-observation-v0",
        "purpose":"Observe one Official action at the real Yield state before naming any new Plan kind.",
        "replay_guard":{
            "turn":turn,
            "day":TARGET_DAY,
            "hour":TARGET_HOUR,
            "target_crop":TARGET_CROP,
            "target_tile":TARGET_TILE,
            "target_yield":TARGET_YIELD,
            "active_provisional_plan_before_probe":active,
            "active_plan_steps":active_steps,
        },
        "pre_state":pre_view,
        "baseline":{
            "action_bundle":baseline_bundle,
            "post_state":b_view,
        },
        "p1":{
            "action_bundle":p1_bundle,
            "post_state":p_view,
            "target_delivery_candidate_present":bool(p_delivery),
            "target_delivery_candidates":[x.to_dict() for x in p_delivery],
        },
        "official_world":{
            "same_pre_state":True,
            "same_opponent_action":opp_bundle,
        },
        "fixed_observation_checklist":checklist,
        "all_fixed_observation_conditions_met":all(checklist.values()),
        "boundary":{
            "no_new_plan_kind_added":True,
            "harvest_action_observed_not_promoted_to_plan":True,
            "output_location_change_does_not_claim_cash_recovery":True,
            "no_terminal_strength_claim":True,
        },
    }

    Path("yield_output_observation_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "pre_tile":pre_view["target_tile"],
        "baseline_post_tile":b_view["target_tile"],
        "p1_post_tile":p_view["target_tile"],
        "baseline_carried":b_view["target_carried_quantity"],
        "p1_carried":p_view["target_carried_quantity"],
        "baseline_shed":b_view["target_shed_quantity"],
        "p1_shed":p_view["target_shed_quantity"],
        "baseline_cash":b_view["cash"],
        "p1_cash":p_view["cash"],
        "target_delivery_candidate_present":bool(p_delivery),
        "fixed_5":result["all_fixed_observation_conditions_met"],
    },separators=(",",":")))


if __name__=="__main__":
    main()
