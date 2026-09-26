#!/usr/bin/env python3
"""Observe the next post-restart blockage and its later reopening.

No machine behavior is changed.

Replay the P1 policy from Exclude Confirmed Blocked A/B v0 and capture:
1) turn 670 / Day27 h22 -> Day27 h23
   The final currently-selectable maintenance job completes and the machine
   enters C_t>0 but selectable=0.
2) turn 695 / Day28 h23 -> Day29 h0
   With PASS only, Official time changes the State and selectable work reappears.

The purpose is to observe the raw World transitions before naming any new
mechanism.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from collections import Counter
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
ENTRY_TURN=670
REOPEN_TURN=695


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def load_opponent():
    spec=importlib.util.spec_from_file_location("seyamalam_post_restart_boundary",OPP_PATH)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def semantic_matches(plans,spec):
    return [p for p in plans if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])]


def first_action_status(snapshot,plan):
    if plan.kind=="prepare_for_plant":
        raw=snapshot.raw()
        p=raw["player"]
        crop=str(plan.target["crop"])
        qty=int(plan.target["missing_seed_quantity"])
        cash=float(raw["farms"][p].get("money",0) or 0)
        required=int(CROPS[crop]["seed"])*qty
        if cash < required:
            return {
                "status":"BLOCKED",
                "reason":"cash_below_official_seed_cost",
                "cash":cash,
                "required_cost":required,
            }
    return {"status":"UNKNOWN","reason":"not_closed_by_current_evidence"}


def plant_rows(snapshot):
    raw=snapshot.raw(); p=raw["player"]
    rows=[]
    for y,row in enumerate(raw["farms"][p].get("tiles",[]) or []):
        for x,tile in enumerate(row):
            if isinstance(tile,dict) and tile.get("kind")=="PLANT":
                rows.append({"tile":[x,y],"raw":copy.deepcopy(tile)})
    return rows


def state_view(snapshot):
    raw=snapshot.raw(); p=raw["player"]
    plans=generate_plans(snapshot)
    statuses=[first_action_status(snapshot,q) for q in plans]
    blocked=sum(1 for s in statuses if s["status"]=="BLOCKED")
    selectable=len(plans)-blocked

    kinds=Counter(q.kind for q in plans)
    selectable_kinds=Counter(
        q.kind for q,s in zip(plans,statuses) if s["status"]!="BLOCKED"
    )

    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(raw["farms"][p].get("money",0) or 0),
        "farmer":plain(raw["farms"][p].get("farmer")),
        "hands":plain(raw["farms"][p].get("hands",[]) or []),
        "seeds":plain(raw["private"].get("seeds",{}) or {}),
        "shed":plain(raw["private"].get("shed",{}) or {}),
        "inventories":plain(raw["private"].get("inventories",[]) or []),
        "plants":plant_rows(snapshot),
        "candidate_count":len(plans),
        "blocked_count":blocked,
        "selectable_count":selectable,
        "candidate_kind_counts":dict(kinds),
        "selectable_kind_counts":dict(selectable_kinds),
    }


def candidate_rows(snapshot):
    rows=[]
    for q in generate_plans(snapshot):
        rows.append({
            "candidate_id":q.candidate_id,
            "kind":q.kind,
            "target":plain(q.target),
            "status":first_action_status(snapshot,q),
        })
    return rows


def main():
    opponent=load_opponent()
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    seq=0
    turn=0

    entry=None
    reopen=None

    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        plans=generate_plans(pre)
        blocked_ids={
            p.candidate_id
            for p in plans
            if first_action_status(pre,p)["status"]=="BLOCKED"
        }
        selectable=[p for p in plans if p.candidate_id not in blocked_ids]

        transition_reason=None
        selection_reason=None

        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS:
                transition_reason="active_plan_step_limit"
                active=None; active_steps=0
            elif not ms:
                transition_reason="active_plan_no_longer_present_before_observed_completion"
                active=None; active_steps=0
            elif first_action_status(pre,ms[0])["status"]=="BLOCKED":
                transition_reason="active_plan_now_confirmed_blocked"
                active=None; active_steps=0

        if active is None and selectable:
            while True:
                drawn=plans[rng.randrange(len(plans))]
                if drawn.candidate_id not in blocked_ids:
                    chosen=drawn
                    break
            seq+=1
            active={"sequence":seq,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0
            selection_reason="coupled_rng_reject_only_confirmed_blocked"

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms: current=ms[0]

        bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp_bundle=plain(opponent.agent(s1["observation"]))

        capture=(turn in (ENTRY_TURN,REOPEN_TURN))
        if capture:
            cap={
                "turn":turn,
                "pre_state":state_view(pre),
                "pre_candidates":candidate_rows(pre),
                "active_before_action":plain(active) if active is not None else None,
                "selection_reason":selection_reason,
                "transition_reason":transition_reason,
                "self_action_bundle":plain(bundle),
                "opponent_action_bundle":copy.deepcopy(opp_bundle),
            }

        env.step([plain(bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        completion=None
        if current is not None and active is not None:
            completion=completion_from_states(current,pre,post)
            active_steps+=1
            if completion.get("complete"):
                active=None; active_steps=0

        if capture:
            cap["completion"]=plain(completion)
            cap["post_state"]=state_view(post)
            cap["post_candidates"]=candidate_rows(post)
            if turn==ENTRY_TURN:
                entry=cap
            else:
                reopen=cap

        if turn==REOPEN_TURN:
            break

        turn+=1

    if entry is None or reopen is None:
        raise RuntimeError("required boundary captures missing")

    if not (
        entry["pre_state"]["day"]==27 and entry["pre_state"]["hour"]==22
        and entry["post_state"]["day"]==27 and entry["post_state"]["hour"]==23
    ):
        raise RuntimeError("entry boundary day/hour mismatch")

    if not (
        reopen["pre_state"]["day"]==28 and reopen["pre_state"]["hour"]==23
        and reopen["post_state"]["day"]==29 and reopen["post_state"]["hour"]==0
    ):
        raise RuntimeError("reopen boundary day/hour mismatch")

    result={
        "schema":"post-restart-next-boundary-v0",
        "purpose":"Observe the next all-BLOCKED interval after the Day25 restart and the Official-time reopening that follows.",
        "environment":{
            "seed":ENV_SEED,
            "policy_seed":POLICY_SEED,
            "opponent":"Seyamalam pinned v21",
        },
        "entry_into_all_blocked":entry,
        "reopen_from_all_blocked":reopen,
        "boundary":{
            "no_new_plan_kind":True,
            "no_new_status_rule":True,
            "no_new_selection_rule":True,
            "entry_post_selectable_zero":entry["post_state"]["selectable_count"]==0,
            "entry_post_candidates_remain":entry["post_state"]["candidate_count"]>0,
            "reopen_pre_selectable_zero":reopen["pre_state"]["selectable_count"]==0,
            "reopen_post_selectable_positive":reopen["post_state"]["selectable_count"]>0,
            "reopen_self_action_is_pass":reopen["self_action_bundle"]=={
                "farmer":["PASS"],"hands":[],"market":[]
            },
        },
    }

    Path("post_restart_next_boundary_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "entry":{
            "pre_day_hour":[entry["pre_state"]["day"],entry["pre_state"]["hour"]],
            "post_day_hour":[entry["post_state"]["day"],entry["post_state"]["hour"]],
            "active":entry["active_before_action"],
            "action":entry["self_action_bundle"],
            "completion":entry["completion"],
            "pre_C_B_S":[entry["pre_state"]["candidate_count"],entry["pre_state"]["blocked_count"],entry["pre_state"]["selectable_count"]],
            "post_C_B_S":[entry["post_state"]["candidate_count"],entry["post_state"]["blocked_count"],entry["post_state"]["selectable_count"]],
            "plants_before":entry["pre_state"]["plants"],
            "plants_after":entry["post_state"]["plants"],
        },
        "reopen":{
            "pre_day_hour":[reopen["pre_state"]["day"],reopen["pre_state"]["hour"]],
            "post_day_hour":[reopen["post_state"]["day"],reopen["post_state"]["hour"]],
            "action":reopen["self_action_bundle"],
            "pre_C_B_S":[reopen["pre_state"]["candidate_count"],reopen["pre_state"]["blocked_count"],reopen["pre_state"]["selectable_count"]],
            "post_C_B_S":[reopen["post_state"]["candidate_count"],reopen["post_state"]["blocked_count"],reopen["post_state"]["selectable_count"]],
            "post_selectable_kinds":reopen["post_state"]["selectable_kind_counts"],
            "plants_before":reopen["pre_state"]["plants"],
            "plants_after":reopen["post_state"]["plants"],
        },
    },separators=(",",":")))


if __name__=="__main__":
    main()
