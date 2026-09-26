#!/usr/bin/env python3
import copy
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as opponent
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_delivery_match,
)

SEED=7001
MAX_PLAN_STEPS=12


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


def unit_position(raw, unit_index):
    farm=raw["farms"][raw["player"]]
    if unit_index==0:
        return list(farm["farmer"])
    return list(farm["hands"][unit_index-1])


def access_tiles(raw):
    farm=raw["farms"][raw["player"]]
    n=len(farm["tiles"])
    h=n//2
    return [(h-1,h-1),(h,h-1),(h-1,h),(h,h)]


def distance_to_shed(raw, unit_index):
    x,y=unit_position(raw,unit_index)
    return min(abs(tx-x)+abs(ty-y) for tx,ty in access_tiles(raw))


def find_test_plan(snapshot, shed_capacity):
    """TEST-ONLY selection, never used by generate_plans or the Agent.

    Pick one clean delivery example:
    - one carried item only
    - item is market-visible (so sale-candidate linkage can be observed)
    - not already at shed access (must show temporal movement)
    - enough room for full transfer
    - enough same-day turns to MOVE...DROP before day boundary
    """
    raw=snapshot.raw()
    hour=raw["hour"]
    for p in generate_plans(snapshot):
        if p.kind!="deliver_carried_to_shed":
            continue
        items=dict(p.target["carried_items"])
        if len(items)!=1:
            continue
        item,qty=next(iter(items.items()))
        if item not in raw["market"].get("prices",{}):
            continue
        d=distance_to_shed(raw,int(p.target["unit_index"]))
        if d<=0:
            continue
        shed_total=sum(v for v in raw["private"].get("shed",{}).values() if isinstance(v,(int,float)))
        if shed_total+qty>shed_capacity:
            continue
        if hour+d>22:
            continue
        return p
    return None


def state_view(snapshot, unit_index, tracked_item):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    invs=raw["private"].get("inventories",[]) or []
    inv=invs[unit_index] if unit_index<len(invs) else {}
    return {
        "day":raw["day"],
        "hour":raw["hour"],
        "cash":farm.get("money",0),
        "unit_index":unit_index,
        "unit_position":unit_position(raw,unit_index),
        "unit_inventory":copy.deepcopy(inv),
        "tracked_item":tracked_item,
        "tracked_unit_qty":inv.get(tracked_item,0),
        "tracked_shed_qty":raw["private"].get("shed",{}).get(tracked_item,0),
        "shed_total":sum(v for v in raw["private"].get("shed",{}).values() if isinstance(v,(int,float))),
    }


def plan_view(snapshot, unit_index, carried_items, tracked_item):
    plans=generate_plans(snapshot)
    delivery=[
        p.to_dict() for p in plans
        if semantic_delivery_match(p,unit_index,carried_items)
    ]
    sale=[
        p.to_dict() for p in plans
        if p.kind=="realize_shed_stock_sale" and p.target.get("item")==tracked_item
    ]
    return {
        "total_candidates":len(plans),
        "same_delivery_candidate_present":bool(delivery),
        "same_delivery_candidates":delivery,
        "tracked_sale_candidate_present":bool(sale),
        "tracked_sale_candidates":sale,
        "tracked_sale_quantities":[p["target"]["available_quantity"] for p in sale],
    }


def same_shape(a,b):
    return (
        isinstance(a,dict) and isinstance(b,dict)
        and set(a)==set(b)=={"farmer","hands","market"}
        and len(a["hands"])==len(b["hands"])
    )


def main():
    current.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":SEED},debug=False)
    env.reset(num_agents=2)
    shed_capacity=int(getattr(env.configuration,"shedCapacity",100))

    # Follow the actual Current-vs-Seyamalam Official battle only until one
    # clean delivery job exists. This scan criterion is test selection only.
    selected=None
    reference_steps=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        obs0=s0["observation"]; obs1=s1["observation"]
        snap=bind_official_state(obs0)
        selected=find_test_plan(snap,shed_capacity)
        if selected is not None:
            break
        env.step([plain(current.agent(obs0)),plain(opponent.agent(obs1))])
        reference_steps+=1

    if selected is None:
        raise RuntimeError("No clean delivery test plan found")

    initial_snapshot=bind_official_state(env._Environment__get_shared_state(0)["observation"])
    unit_index=int(selected.target["unit_index"])
    carried_items=dict(selected.target["carried_items"])
    tracked_item=next(iter(carried_items))
    initial_target_qty=carried_items[tracked_item]

    trace=[]
    completion=None
    guard_all=True

    for plan_step in range(MAX_PLAN_STEPS):
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        pre_view=state_view(pre,unit_index,tracked_item)
        pre_plan_view=plan_view(pre,unit_index,carried_items,tracked_item)

        # Replan from the new State every turn. Never replay a prebuilt action sequence.
        matching=[
            p for p in generate_plans(pre)
            if semantic_delivery_match(p,unit_index,carried_items)
        ]
        if not matching:
            completion=completion_from_states(selected,initial_snapshot,pre)
            if completion["complete"]:
                break
            raise RuntimeError("Delivery job disappeared before completion")
        current_plan=matching[0]

        p1_bundle=project_short_plan(pre,current_plan)
        baseline_bundle=baseline_pass_bundle(pre)
        shape_ok=same_shape(baseline_bundle,p1_bundle)
        action_diff=(baseline_bundle!=p1_bundle)

        # Same PreState, same opponent response, two own ActionBundles.
        opp_action=plain(opponent.agent(s1["observation"]))
        baseline_env=copy.deepcopy(env)
        p1_env=copy.deepcopy(env)
        baseline_env.step([copy.deepcopy(baseline_bundle),copy.deepcopy(opp_action)])
        p1_env.step([copy.deepcopy(p1_bundle),copy.deepcopy(opp_action)])

        baseline_post=bind_official_state(baseline_env._Environment__get_shared_state(0)["observation"])
        p1_post=bind_official_state(p1_env._Environment__get_shared_state(0)["observation"])

        baseline_completion=completion_from_states(current_plan,pre,baseline_post)
        p1_step_completion=completion_from_states(current_plan,pre,p1_post)
        whole_completion=completion_from_states(selected,initial_snapshot,p1_post)

        baseline_post_view=state_view(baseline_post,unit_index,tracked_item)
        p1_post_view=state_view(p1_post,unit_index,tracked_item)
        baseline_plans=plan_view(baseline_post,unit_index,carried_items,tracked_item)
        p1_plans=plan_view(p1_post,unit_index,carried_items,tracked_item)

        next_bundle=None
        next_delivery_present=p1_plans["same_delivery_candidate_present"]
        if not whole_completion["complete"] and next_delivery_present:
            next_matching=[
                p for p in generate_plans(p1_post)
                if semantic_delivery_match(p,unit_index,carried_items)
            ]
            next_bundle=project_short_plan(p1_post,next_matching[0])

        record={
            "plan_step":plan_step,
            "pre_state":pre_view,
            "plan_before":current_plan.to_dict(),
            "baseline":{
                "action_bundle":baseline_bundle,
                "post_state":baseline_post_view,
                "completion_this_step":baseline_completion,
                "post_plan_view":baseline_plans,
            },
            "p1":{
                "action_bundle":p1_bundle,
                "post_state":p1_post_view,
                "completion_this_step":p1_step_completion,
                "completion_from_initial_target":whole_completion,
                "post_plan_view":p1_plans,
                "next_replanned_action_bundle":next_bundle,
            },
            "official_world":{
                "same_pre_state":True,
                "same_opponent_action":opp_action,
                "baseline_and_p1_bundle_same_shape":shape_ok,
                "own_action_diff_present":action_diff,
                "observed_effect":{
                    "unit_position_baseline":baseline_post_view["unit_position"],
                    "unit_position_p1":p1_post_view["unit_position"],
                    "tracked_unit_qty_baseline":baseline_post_view["tracked_unit_qty"],
                    "tracked_unit_qty_p1":p1_post_view["tracked_unit_qty"],
                    "tracked_shed_qty_baseline":baseline_post_view["tracked_shed_qty"],
                    "tracked_shed_qty_p1":p1_post_view["tracked_shed_qty"],
                },
            },
        }
        trace.append(record)
        guard_all=guard_all and shape_ok and action_diff

        env=p1_env
        completion=whole_completion
        if whole_completion["complete"]:
            break

    final_snapshot=bind_official_state(env._Environment__get_shared_state(0)["observation"])
    final_view=state_view(final_snapshot,unit_index,tracked_item)
    final_plan_view=plan_view(final_snapshot,unit_index,carried_items,tracked_item)

    # Fixed observation-completion checklist for this Probe.
    checklist={
        "action_difference_observed":all(r["official_world"]["own_action_diff_present"] for r in trace),
        "pre_and_post_state_observed":all("pre_state" in r and "post_state" in r["p1"] for r in trace),
        "official_world_effect_observed":all("observed_effect" in r["official_world"] for r in trace),
        "baseline_and_p1_compared_same_shape":all(r["official_world"]["baseline_and_p1_bundle_same_shape"] for r in trace),
        "next_state_to_replan_connected":all(
            r["p1"]["completion_from_initial_target"]["complete"]
            or r["p1"]["next_replanned_action_bundle"] is not None
            for r in trace
        ),
    }

    result={
        "schema":"deliver-plan-minimal-loop-v0",
        "seed":SEED,
        "reference_current_steps_before_test":reference_steps,
        "test_selection_only":{
            "reason":"connection-observability, not policy priority",
            "criteria":"single market-visible carried item; non-shed position; full shed room; completes before day boundary",
        },
        "initial":{
            "state":state_view(initial_snapshot,unit_index,tracked_item),
            "plan":selected.to_dict(),
            "plan_view":plan_view(initial_snapshot,unit_index,carried_items,tracked_item),
            "tracked_item":tracked_item,
            "tracked_quantity":initial_target_qty,
        },
        "trace":trace,
        "final":{
            "state":final_view,
            "completion":completion,
            "plan_view":final_plan_view,
        },
        "guards":{
            "steps_executed":len(trace),
            "bundle_shape_and_action_diff_all_steps":guard_all,
            "completed_within_limit":bool(completion and completion["complete"]),
            "fixed_observation_checklist":checklist,
            "all_fixed_observation_conditions_met":all(checklist.values()),
        },
        "boundary":{
            "this_is_not_plan_selection_policy":True,
            "this_does_not_measure_terminal_strength":True,
            "supported_projector_kind":["deliver_carried_to_shed"],
            "unsupported":["plan ranking","plan selection","terminal evaluation"],
        },
    }
    Path("deliver_plan_minimal_loop_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "seed":SEED,
        "reference_steps":reference_steps,
        "initial_day":result["initial"]["state"]["day"],
        "initial_hour":result["initial"]["state"]["hour"],
        "unit_index":unit_index,
        "tracked_item":tracked_item,
        "tracked_quantity":initial_target_qty,
        "steps_executed":len(trace),
        "completed":result["guards"]["completed_within_limit"],
        "all_fixed_observation_conditions_met":result["guards"]["all_fixed_observation_conditions_met"],
        "initial_sale_quantities":result["initial"]["plan_view"]["tracked_sale_quantities"],
        "final_sale_quantities":result["final"]["plan_view"]["tracked_sale_quantities"],
        "final_delivery_present":result["final"]["plan_view"]["same_delivery_candidate_present"],
    },separators=(",",":")))


if __name__=="__main__":
    main()
