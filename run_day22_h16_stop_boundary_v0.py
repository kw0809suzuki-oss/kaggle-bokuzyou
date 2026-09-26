#!/usr/bin/env python3
"""Observe the terminal no-candidate boundary in the existing Yield-to-Cash rollout.

No new Plan or action is introduced.

Replay the exact seed/opponent/provisional policy and capture:
  Day22 h15 State
  -> the exact already-selected ActionBundle
  -> Official World
  -> Day22 h16 State

Then continue the same machine unchanged to terminal and record whether any
candidate ever reappears, plus raw World material that remains while the
Generator reports zero jobs.
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
BOUNDARY_DAY=22
BOUNDARY_PRE_HOUR=15
BOUNDARY_POST_HOUR=16


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


def candidate_view(plans):
    counts=Counter(p.kind for p in plans)
    return {
        "count":len(plans),
        "kind_counts":dict(counts),
        "candidates":[p.to_dict() for p in plans],
    }


def world_material(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    private=raw["private"]

    tiles={
        "none":[],
        "locked":[],
        "weed":[],
        "plants":[],
        "other":[],
    }
    for y,row in enumerate(farm.get("tiles",[]) or []):
        for x,tile in enumerate(row):
            if tile is None:
                tiles["none"].append([x,y])
            elif tile=="LOCKED":
                tiles["locked"].append([x,y])
            elif isinstance(tile,dict) and tile.get("kind")=="WEED":
                tiles["weed"].append([x,y])
            elif isinstance(tile,dict) and tile.get("kind")=="PLANT":
                tiles["plants"].append({"tile":[x,y],"raw":copy.deepcopy(tile)})
            else:
                tiles["other"].append({"tile":[x,y],"raw":copy.deepcopy(tile)})

    invs=[]
    for i,inv in enumerate(private.get("inventories",[]) or []):
        nz={str(k):v for k,v in (inv or {}).items() if isinstance(v,(int,float)) and v}
        if nz:
            invs.append({"unit_index":i,"items":nz})

    shed={
        str(k):v for k,v in (private.get("shed",{}) or {}).items()
        if isinstance(v,(int,float)) and v
    }
    seeds={
        str(k):v for k,v in (private.get("seeds",{}) or {}).items()
        if isinstance(v,(int,float)) and v
    }

    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "farmer":plain(farm.get("farmer")),
        "hands":plain(farm.get("hands",[]) or []),
        "unlocked_quadrants":plain(farm.get("unlocked_quadrants",[]) or []),
        "seeds":seeds,
        "shed":shed,
        "inventories":invs,
        "tiles":tiles,
        "market_prices":plain((raw.get("market",{}) or {}).get("prices",{}) or {}),
        "town":plain(raw.get("town",{}) or {}),
    }


def main():
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    plan_sequence=0

    boundary=None
    post_zero_samples=[]
    first_reappearance=None
    zero_started=False
    zero_start_turn=None
    zero_turns_after_boundary=0

    selected_counts=Counter()
    completed_counts=Counter()
    invalidated_counts=Counter()
    timeout_counts=Counter()
    execution_errors=[]

    turn=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        raw=pre.raw()
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
            action_mode="pass_no_executable_selected_plan"
        else:
            try:
                bundle=project_short_plan(pre,current_plan)
                action_mode="project_active_plan"
            except Exception as exc:
                execution_errors.append({
                    "turn":turn,
                    "day":raw["day"],
                    "hour":raw["hour"],
                    "active":plain(active),
                    "error":repr(exc),
                })
                bundle=baseline_pass_bundle(pre)
                action_mode="pass_projector_error"
                active=None
                active_steps=0

        is_boundary_pre=(
            int(raw["day"])==BOUNDARY_DAY
            and int(raw["hour"])==BOUNDARY_PRE_HOUR
        )
        if is_boundary_pre:
            boundary={
                "turn":turn,
                "pre_state":{
                    "world":world_material(pre),
                    "generator":candidate_view(plans),
                },
                "active_before_action":plain(active),
                "active_steps_before_action":active_steps,
                "transition_reason":transition_reason,
                "selection_reason":selection_reason,
                "action_mode":action_mode,
                "self_action_bundle":plain(bundle),
            }

        opp_bundle=plain(opponent.agent(s1["observation"]))
        if is_boundary_pre:
            boundary["opponent_action_bundle"]=copy.deepcopy(opp_bundle)

        env.step([plain(bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        completion=None
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

        if is_boundary_pre:
            post_raw=post.raw()
            if not (
                int(post_raw["day"])==BOUNDARY_DAY
                and int(post_raw["hour"])==BOUNDARY_POST_HOUR
            ):
                raise RuntimeError(f"unexpected boundary PostState: {post_raw['day']} h{post_raw['hour']}")
            post_plans=generate_plans(post)
            boundary["completion"]=plain(completion)
            boundary["same_plan_present_post"]=same_plan_present_post
            boundary["post_state"]={
                "world":world_material(post),
                "generator":candidate_view(post_plans),
            }
            boundary["disappeared_candidate_ids"]=sorted(
                set(p.candidate_id for p in plans)-set(p.candidate_id for p in post_plans)
            )
            boundary["appeared_candidate_ids"]=sorted(
                set(p.candidate_id for p in post_plans)-set(p.candidate_id for p in plans)
            )
            if len(post_plans)==0:
                zero_started=True
                zero_start_turn=turn+1

        # From Day22 h16 onward, record daily h0 plus terminal-adjacent snapshots;
        # also watch every State for any candidate reappearance.
        if zero_started:
            now=post
            now_raw=now.raw()
            now_plans=generate_plans(now)
            if len(now_plans)>0 and first_reappearance is None:
                first_reappearance={
                    "turn":turn+1,
                    "state":{"world":world_material(now),"generator":candidate_view(now_plans)},
                }
            if len(now_plans)==0:
                zero_turns_after_boundary+=1
            if (
                int(now_raw["hour"])==0
                or int(now_raw["day"])==22 and int(now_raw["hour"])==16
                or int(now_raw["day"])>=29 and int(now_raw["hour"])>=20
            ):
                post_zero_samples.append({
                    "turn":turn+1,
                    "world":world_material(now),
                    "generator":candidate_view(now_plans),
                })

        turn+=1

    final=bind_official_state(env.state[0].observation)
    rewards=[]
    for st in env.state:
        try: rewards.append(float(st.reward))
        except Exception: rewards.append(None)

    if boundary is None:
        raise RuntimeError("Day22 h15 boundary not observed")

    pre_world=boundary["pre_state"]["world"]
    post_world=boundary["post_state"]["world"]

    result={
        "schema":"day22-h16-stop-boundary-v0",
        "purpose":"Observe what disappears/remains when the current machine enters its final candidate-zero interval.",
        "environment":{
            "seed":ENV_SEED,
            "policy_seed":POLICY_SEED,
            "opponent":"Seyamalam pinned v21",
        },
        "boundary":boundary,
        "material_delta":{
            "cash":post_world["cash"]-pre_world["cash"],
            "seeds_before":pre_world["seeds"],
            "seeds_after":post_world["seeds"],
            "shed_before":pre_world["shed"],
            "shed_after":post_world["shed"],
            "inventories_before":pre_world["inventories"],
            "inventories_after":post_world["inventories"],
            "plants_before":pre_world["tiles"]["plants"],
            "plants_after":post_world["tiles"]["plants"],
            "none_count_before":len(pre_world["tiles"]["none"]),
            "none_count_after":len(post_world["tiles"]["none"]),
            "weed_count_before":len(pre_world["tiles"]["weed"]),
            "weed_count_after":len(post_world["tiles"]["weed"]),
            "locked_count_before":len(pre_world["tiles"]["locked"]),
            "locked_count_after":len(post_world["tiles"]["locked"]),
        },
        "after_boundary":{
            "zero_start_turn":zero_start_turn,
            "zero_states_observed":zero_turns_after_boundary,
            "first_candidate_reappearance":first_reappearance,
            "sampled_states":post_zero_samples,
        },
        "terminal":{
            "ran_to_terminal":bool(env.done),
            "turns":turn,
            "self_cash":float(final.raw()["farms"][0]["money"]),
            "official_rewards":rewards,
            "world":world_material(final),
            "generator":candidate_view(generate_plans(final)),
        },
        "replay_guard":{
            "terminal_cash_matches_prior_2990":float(final.raw()["farms"][0]["money"])==2990.0,
            "selected_plans_by_kind":dict(selected_counts),
            "completed_plans_by_kind":dict(completed_counts),
            "invalidated_plans_by_kind":dict(invalidated_counts),
            "timed_out_plans_by_kind":dict(timeout_counts),
            "execution_error_count":len(execution_errors),
        },
        "boundary_note":{
            "candidate_zero_means_current_generator_reads_no_job":True,
            "candidate_zero_does_not_mean_official_world_has_no_possible_action":True,
            "no_new_plan_or_intervention_added":True,
        },
    }

    Path("day22_h16_stop_boundary_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "terminal_cash":result["terminal"]["self_cash"],
        "replay_matches_2990":result["replay_guard"]["terminal_cash_matches_prior_2990"],
        "h15_candidates":boundary["pre_state"]["generator"]["count"],
        "h15_active":boundary["active_before_action"],
        "h15_action":boundary["self_action_bundle"],
        "h16_candidates":boundary["post_state"]["generator"]["count"],
        "cash_delta":result["material_delta"]["cash"],
        "seeds_after":result["material_delta"]["seeds_after"],
        "shed_after":result["material_delta"]["shed_after"],
        "inventories_after":result["material_delta"]["inventories_after"],
        "plants_after":result["material_delta"]["plants_after"],
        "none_count_after":result["material_delta"]["none_count_after"],
        "weed_count_after":result["material_delta"]["weed_count_after"],
        "locked_count_after":result["material_delta"]["locked_count_after"],
        "candidate_reappeared":first_reappearance is not None,
        "zero_states_observed":zero_turns_after_boundary,
    },separators=(",",":")))


if __name__=="__main__":
    main()
