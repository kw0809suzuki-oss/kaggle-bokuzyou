#!/usr/bin/env python3
"""One full-season rollout of the current ShortPlan machine after adding plant maintenance.

This is deliberately NOT a strength test. Selection is an explicit provisional
policy:
- re-generate candidates from every Official State
- continue the same semantic Plan while it remains present and under a fixed
  step limit
- otherwise choose one candidate uniformly using a fixed RNG seed
- if no candidate exists, submit a legal all-PASS bundle
- never fill a missing job with an ad-hoc action

The purpose is to observe what machine currently exists from game start to
terminal and where work becomes unavailable.
"""

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


def semantic_matches(plans, spec):
    return [
        p for p in plans
        if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])
    ]


def state_summary(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    private=raw["private"]
    plants=Counter()
    animals=Counter()
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if isinstance(tile,dict):
                if tile.get("kind")=="PLANT":
                    plants[str(tile.get("crop"))]+=1
                if tile.get("animal"):
                    animals[str(tile.get("animal"))]+=1
    shed={k:v for k,v in (private.get("shed",{}) or {}).items() if isinstance(v,(int,float)) and v}
    seeds={k:v for k,v in (private.get("seeds",{}) or {}).items() if isinstance(v,(int,float)) and v}
    invs=private.get("inventories",[]) or []
    carried_total=0
    carried={}
    for i,inv in enumerate(invs):
        if not isinstance(inv,dict): continue
        nonzero={k:v for k,v in inv.items() if isinstance(v,(int,float)) and v}
        if nonzero:
            carried[str(i)]=nonzero
            carried_total+=sum(nonzero.values())
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "hands":len(farm.get("hands",[]) or []),
        "land_quadrants":list(farm.get("unlocked_quadrants",[]) or []),
        "seed_total":sum(seeds.values()) if seeds else 0,
        "seeds":seeds,
        "shed_total":sum(shed.values()) if shed else 0,
        "shed":shed,
        "carried_total":carried_total,
        "carried":carried,
        "plants":dict(plants),
        "animals":dict(animals),
    }


def main():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    plan_sequence=0

    turn_records=[]
    daily=[]
    selected_counts=Counter()
    completed_counts=Counter()
    invalidated_counts=Counter()
    timeout_counts=Counter()
    no_candidate_turns=0
    pass_turns=0
    execution_errors=[]

    initial=bind_official_state(env._Environment__get_shared_state(0)["observation"])
    daily.append({"point":"initial","state":state_summary(initial)})

    turn=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        plans=generate_plans(pre)

        transition_reason=None
        selection_reason=None

        if active is not None:
            matches=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS:
                timeout_counts[active["kind"]]+=1
                transition_reason="active_plan_step_limit"
                active=None
                active_steps=0
            elif not matches:
                invalidated_counts[active["kind"]]+=1
                transition_reason="active_plan_no_longer_present_before_observed_completion"
                active=None
                active_steps=0

        if active is None and plans:
            chosen=plans[rng.randrange(len(plans))]
            plan_sequence+=1
            active={
                "sequence":plan_sequence,
                "kind":chosen.kind,
                "target":plain(chosen.target),
            }
            active_steps=0
            selected_counts[chosen.kind]+=1
            selection_reason="fixed_rng_uniform_over_current_candidates"

        current_plan=None
        if active is not None:
            matches=semantic_matches(plans,active)
            if matches:
                current_plan=matches[0]

        if current_plan is None:
            bundle=baseline_pass_bundle(pre)
            no_candidate_turns+=1 if not plans else 0
            pass_turns+=1
            action_mode="pass_no_executable_selected_plan"
        else:
            try:
                bundle=project_short_plan(pre,current_plan)
                action_mode="project_active_plan"
            except Exception as exc:
                execution_errors.append({
                    "turn":turn,
                    "day":pre.raw()["day"],
                    "hour":pre.raw()["hour"],
                    "active":plain(active),
                    "error":repr(exc),
                })
                bundle=baseline_pass_bundle(pre)
                pass_turns+=1
                action_mode="pass_projector_error"
                active=None
                active_steps=0

        opp_bundle=plain(opponent.agent(s1["observation"]))
        env.step([plain(bundle),opp_bundle])

        post=bind_official_state(env.state[0].observation)
        completion=None
        active_after=plain(active) if active is not None else None
        same_plan_present_post=False

        if current_plan is not None and active is not None:
            completion=completion_from_states(current_plan,pre,post)
            post_plans=generate_plans(post)
            same_plan_present_post=bool(semantic_matches(post_plans,active))
            active_steps+=1
            if completion["complete"]:
                completed_counts[active["kind"]]+=1
                active=None
                active_steps=0

        turn_records.append({
            "turn":turn,
            "pre":{"day":pre.raw()["day"],"hour":pre.raw()["hour"],"candidate_count":len(plans)},
            "selection_reason":selection_reason,
            "transition_reason":transition_reason,
            "active_before_action":active_after,
            "action_mode":action_mode,
            "action_bundle":plain(bundle),
            "completion":plain(completion),
            "same_plan_present_in_post":same_plan_present_post,
            "post":{"day":post.raw()["day"],"hour":post.raw()["hour"],"candidate_count":len(generate_plans(post))},
        })

        post_raw=post.raw()
        if int(post_raw["hour"])==0:
            daily.append({"point":f"day_{int(post_raw['day'])}_h0","state":state_summary(post)})

        turn+=1

    final=bind_official_state(env.state[0].observation)
    final_self=state_summary(final)

    # reward is recorded if exposed by the official environment; final cash is
    # retained separately and is the stable self-state observation.
    rewards=[]
    for st in env.state:
        try:
            rewards.append(float(st.reward))
        except Exception:
            rewards.append(None)

    # Compress no-candidate intervals so the outer trajectory can be inspected
    # without reading all 720 actions.
    streaks=[]
    start=None
    for r in turn_records:
        empty=(r["pre"]["candidate_count"]==0)
        if empty and start is None:
            start=r["turn"]
        if not empty and start is not None:
            streaks.append({"start_turn":start,"end_turn":r["turn"]-1,"length":r["turn"]-start})
            start=None
    if start is not None:
        streaks.append({"start_turn":start,"end_turn":turn_records[-1]["turn"],"length":turn_records[-1]["turn"]-start+1})

    result={
        "schema":"short-plan-terminal-rollout-v0",
        "purpose":"Run the current three-template Plan machine from initial Official State to terminal under an explicit provisional selection policy.",
        "environment":{"seed":ENV_SEED,"opponent":"Seyamalam pinned v21"},
        "provisional_policy":{
            "policy_seed":POLICY_SEED,
            "selection":"uniform RNG over all current ShortPlan candidates",
            "continuation":"keep same semantic Plan while candidate remains and steps < limit",
            "max_plan_steps":MAX_PLAN_STEPS,
            "empty_candidates":"legal all-PASS",
            "projector_failure":"record reason, PASS, reselect next State",
            "strength_assumption":None,
        },
        "summary":{
            "ran_to_terminal":bool(env.done),
            "turns":turn,
            "terminal_self_cash":final_self["cash"],
            "official_rewards":rewards,
            "selected_plans_by_kind":dict(selected_counts),
            "completed_plans_by_kind":dict(completed_counts),
            "invalidated_plans_by_kind":dict(invalidated_counts),
            "timed_out_plans_by_kind":dict(timeout_counts),
            "no_candidate_turns":no_candidate_turns,
            "pass_turns":pass_turns,
            "execution_error_count":len(execution_errors),
            "no_candidate_streaks":streaks,
        },
        "initial_state":state_summary(initial),
        "daily_trajectory":daily,
        "final_state":final_self,
        "execution_errors":execution_errors,
        "turns":turn_records,
        "boundary":{
            "not_a_strength_verdict":True,
            "candidate_templates_only":[
                "deliver_carried_to_shed",
                "realize_shed_stock_sale",
                "establish_plant",
                "prepare_for_plant",
                "maintain_plant_today",
                "collect_plant_output",
                "prepare_surface_for_plant",
            ],
            "missing_jobs_are_not_filled_ad_hoc":True,
        },
    }

    Path("short_plan_terminal_rollout_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(result["summary"],separators=(",",":")))


if __name__=="__main__":
    main()
