#!/usr/bin/env python3
"""Observe physical completion feasibility of the last MELON establish job.

No machine behavior is changed.

Replay the exact P1 policy from Exclude Confirmed Blocked A/B v0.
Detect the moment the final establish_plant(MELON,[4,0]) job is first selected,
then record every Official action turn until terminal.

Measure only:
- selection turn/day/hour and unit position
- Official action turns remaining until terminal
- actual actions taken
- minimum remaining movement + PLANT actions at selection
- minimum remaining movement + PLANT actions at terminal snapshot
- whether the job completed

This does not add terminal-aware logic or infer value.
"""

from __future__ import annotations
import copy, importlib.util, json, random
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT=Path(__file__).resolve().parent
OPP_PATH=ROOT/"opponents"/"seyamalam_v21.py"

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
TARGET_KIND="establish_plant"
TARGET_CROP="MELON"
TARGET_TILE=[4,0]


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def load_opponent():
    spec=importlib.util.spec_from_file_location("seyamalam_last_melon",OPP_PATH)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def semantic_matches(plans,spec):
    return [p for p in plans if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])]


def first_action_status(snapshot,plan):
    if plan.kind=="prepare_for_plant":
        raw=snapshot.raw(); p=raw["player"]
        crop=str(plan.target["crop"])
        qty=int(plan.target["missing_seed_quantity"])
        cash=float(raw["farms"][p].get("money",0) or 0)
        required=int(CROPS[crop]["seed"])*qty
        if cash < required:
            return {"status":"BLOCKED","cash":cash,"required_cost":required}
    return {"status":"UNKNOWN"}


def is_target(plan):
    return (
        plan is not None
        and plan.kind==TARGET_KIND
        and str(plan.target.get("crop"))==TARGET_CROP
        and list(plan.target.get("tile",[]))==TARGET_TILE
    )


def unit_positions(snapshot):
    raw=snapshot.raw(); p=raw["player"]
    farm=raw["farms"][p]
    return [list(farm["farmer"])] + [list(x) for x in (farm.get("hands",[]) or [])]


def nearest_unit_and_min_actions(snapshot):
    positions=unit_positions(snapshot)
    tx,ty=TARGET_TILE
    distances=[abs(pos[0]-tx)+abs(pos[1]-ty) for pos in positions]
    i=min(range(len(distances)), key=lambda j:(distances[j],j))
    # establish_plant projector needs movement to the tile, then PLANT once on it.
    return {
        "unit_index":i,
        "unit_position":positions[i],
        "manhattan_distance":distances[i],
        "minimum_actions_to_complete":distances[i]+1,
    }


def tile_at(snapshot):
    raw=snapshot.raw(); p=raw["player"]; x,y=TARGET_TILE
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def main():
    opponent=load_opponent()
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    seq=0
    turn=0

    selected=None
    trace=[]

    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        plans=generate_plans(pre)

        blocked_ids={
            q.candidate_id for q in plans
            if first_action_status(pre,q)["status"]=="BLOCKED"
        }
        selectable=[q for q in plans if q.candidate_id not in blocked_ids]

        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not ms:
                active=None; active_steps=0
            elif first_action_status(pre,ms[0])["status"]=="BLOCKED":
                active=None; active_steps=0

        newly_selected=False
        if active is None and selectable:
            while True:
                drawn=plans[rng.randrange(len(plans))]
                if drawn.candidate_id not in blocked_ids:
                    chosen=drawn
                    break
            seq+=1
            active={"sequence":seq,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0
            newly_selected=True

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms: current=ms[0]

        if newly_selected and is_target(current):
            if selected is not None:
                raise RuntimeError("target MELON job selected more than once")
            raw=pre.raw()
            selected={
                "turn":turn,
                "day":int(raw["day"]),
                "hour":int(raw["hour"]),
                "active_sequence":active["sequence"],
                "target":plain(current.target),
                "status":first_action_status(pre,current),
                "cash":float(raw["farms"][raw["player"]].get("money",0) or 0),
                "seed_qty":int((raw["private"].get("seeds",{}) or {}).get(TARGET_CROP,0) or 0),
                "target_tile_before":tile_at(pre),
                "geometry":nearest_unit_and_min_actions(pre),
            }

        bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp_bundle=plain(opponent.agent(s1["observation"]))

        if selected is not None and active is not None and active["sequence"]==selected["active_sequence"]:
            trace.append({
                "turn":turn,
                "day":int(pre.raw()["day"]),
                "hour":int(pre.raw()["hour"]),
                "pre_unit_positions":unit_positions(pre),
                "target_tile_pre":tile_at(pre),
                "projected_action_bundle":plain(bundle),
                "geometry_before_action":nearest_unit_and_min_actions(pre),
            })

        env.step([plain(bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        comp=None
        if current is not None and active is not None:
            comp=completion_from_states(current,pre,post)
            active_steps+=1
            if selected is not None and active["sequence"]==selected["active_sequence"] and trace:
                trace[-1]["completion_after_action"]=plain(comp)
                trace[-1]["post_unit_positions"]=unit_positions(post)
                trace[-1]["target_tile_post"]=tile_at(post)
                trace[-1]["env_done_after_action"]=bool(env.done)
            if comp.get("complete"):
                active=None; active_steps=0

        turn+=1

    if selected is None:
        raise RuntimeError("target MELON establish job was never selected")

    final=bind_official_state(env.state[0].observation)
    final_plans=generate_plans(final)
    same_target_final=any(
        q.kind==TARGET_KIND
        and str(q.target.get("crop"))==TARGET_CROP
        and list(q.target.get("tile",[]))==TARGET_TILE
        for q in final_plans
    )

    actions_available=len(trace)
    initial_need=selected["geometry"]["minimum_actions_to_complete"]
    final_need=nearest_unit_and_min_actions(final)["minimum_actions_to_complete"]

    result={
        "schema":"last-melon-feasibility-observation-v0",
        "purpose":"Determine whether the final MELON establish job was already physically impossible to finish when selected.",
        "selection":selected,
        "trace":trace,
        "terminal":{
            "turns_executed_after_selection_including_selection_turn":actions_available,
            "final_day":int(final.raw()["day"]),
            "final_hour":int(final.raw()["hour"]),
            "env_done":bool(env.done),
            "target_tile":tile_at(final),
            "same_semantic_candidate_visible":same_target_final,
            "geometry":nearest_unit_and_min_actions(final),
        },
        "comparison":{
            "official_action_turns_available_from_selection_through_terminal":actions_available,
            "minimum_actions_required_at_selection":initial_need,
            "action_deficit_at_selection":max(0,initial_need-actions_available),
            "minimum_actions_remaining_at_terminal_snapshot":final_need,
            "completed":not same_target_final and tile_at(final) is not None,
            "selection_time_completion_physically_possible_under_existing_projector":initial_need<=actions_available,
        },
        "boundary":{
            "no_terminal_aware_logic_added":True,
            "no_value_judgment_added":True,
            "movement_plus_plant_only_for_existing_establish_projector":True,
        },
    }

    Path("last_melon_feasibility_observation_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps({
        "selected_turn":selected["turn"],
        "selected_day_hour":[selected["day"],selected["hour"]],
        "selected_position":selected["geometry"]["unit_position"],
        "target":TARGET_TILE,
        "minimum_actions_required_at_selection":initial_need,
        "official_action_turns_available":actions_available,
        "action_deficit_at_selection":result["comparison"]["action_deficit_at_selection"],
        "trace":[
            {
                "turn":r["turn"],
                "day_hour":[r["day"],r["hour"]],
                "action":r["projected_action_bundle"],
                "pre_positions":r["pre_unit_positions"],
                "post_positions":r.get("post_unit_positions"),
                "complete":r.get("completion_after_action",{}).get("complete"),
                "done_after":r.get("env_done_after_action"),
            }
            for r in trace
        ],
        "terminal_position":result["terminal"]["geometry"]["unit_position"],
        "minimum_actions_remaining_terminal":final_need,
        "same_candidate_terminal":same_target_final,
        "physically_possible_at_selection":result["comparison"]["selection_time_completion_physically_possible_under_existing_projector"],
    },separators=(",",":")))


if __name__=="__main__":
    main()
