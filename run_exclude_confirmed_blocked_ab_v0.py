#!/usr/bin/env python3
"""A/B whole-season test: exclude only confirmed BLOCKED candidates.

Baseline:
    Selection(C_t)

P1:
    Selection(C_t \ BLOCKED_t)

The generator/projector/completion logic is unchanged.

Current v0 FirstActionStatus rule is intentionally narrow:
- prepare_for_plant is BLOCKED only when current cash is below the Official
  fixed seed cost for its projected BUY_SEED order.
- everything else is UNKNOWN.

UNKNOWN remains selectable. No score, priority, utility, or plan-completion
prediction is introduced.

The runner executes Baseline and P1 independently with the same environment
seed, pinned opponent, and provisional RNG seed, then finds the first turn
where their ActionBundles diverge and records:
PreState -> Action difference -> Official World -> PostState -> next replan.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT=Path(__file__).resolve().parent
OPP_PATH=ROOT/"opponents"/"seyamalam_v21.py"
sys.path.insert(0,str(ROOT/"opponents"))

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


def load_opponent(tag):
    spec=importlib.util.spec_from_file_location(f"seyamalam_v21_{tag}",OPP_PATH)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def semantic_matches(plans,spec):
    return [
        p for p in plans
        if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])
    ]


def first_action_status(snapshot,plan):
    """Return only evidence-supported status. Default UNKNOWN."""
    if plan.kind=="prepare_for_plant":
        raw=snapshot.raw()
        p=raw["player"]
        crop=str(plan.target["crop"])
        qty=int(plan.target["missing_seed_quantity"])
        cash=float(raw["farms"][p].get("money",0) or 0)
        unit_cost=int(CROPS[crop]["seed"])
        required=unit_cost*qty
        if cash < required:
            return {
                "status":"BLOCKED",
                "reason":"cash_below_official_seed_cost",
                "cash":cash,
                "unit_cost":unit_cost,
                "quantity":qty,
                "required_cost":required,
            }
    return {"status":"UNKNOWN","reason":"not_closed_by_current_evidence"}


def candidate_status_rows(snapshot,plans):
    rows=[]
    for p in plans:
        st=first_action_status(snapshot,p)
        rows.append({
            "candidate_id":p.candidate_id,
            "kind":p.kind,
            "target":plain(p.target),
            **st,
        })
    return rows


def is_blocked(snapshot,plan):
    return first_action_status(snapshot,plan)["status"]=="BLOCKED"


def state_summary(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    private=raw["private"]
    plants=Counter()
    weeds=0
    empty=0
    locked=0
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            if tile is None:
                empty+=1
            elif tile=="LOCKED":
                locked+=1
            elif isinstance(tile,dict):
                if tile.get("kind")=="PLANT":
                    plants[str(tile.get("crop"))]+=1
                elif tile.get("kind")=="WEED":
                    weeds+=1
    shed={k:v for k,v in (private.get("shed",{}) or {}).items() if isinstance(v,(int,float)) and v}
    seeds={k:v for k,v in (private.get("seeds",{}) or {}).items() if isinstance(v,(int,float)) and v}
    invs=private.get("inventories",[]) or []
    carried={}
    for i,inv in enumerate(invs):
        if not isinstance(inv,dict): continue
        nz={k:v for k,v in inv.items() if isinstance(v,(int,float)) and v}
        if nz: carried[str(i)]=nz
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "farmer":plain(farm.get("farmer")),
        "hands":plain(farm.get("hands",[]) or []),
        "seeds":seeds,
        "shed":shed,
        "carried":carried,
        "plants":dict(plants),
        "empty_tiles":empty,
        "weeds":weeds,
        "locked_tiles":locked,
    }


def run_policy(mode):
    if mode not in ("baseline","exclude_blocked"):
        raise ValueError(mode)

    opponent=load_opponent(mode)
    rng=random.Random(POLICY_SEED)
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    active=None
    active_steps=0
    plan_sequence=0

    selected_counts=Counter()
    completed_counts=Counter()
    invalidated_counts=Counter()
    timeout_counts=Counter()
    blocked_active_drops=Counter()
    blocked_candidate_turns=0
    blocked_candidate_instances=0
    all_blocked_turns=0
    pass_turns=0
    execution_errors=[]
    turns=[]

    turn=0
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre=bind_official_state(s0["observation"])
        plans=generate_plans(pre)
        statuses=candidate_status_rows(pre,plans)
        blocked_ids={r["candidate_id"] for r in statuses if r["status"]=="BLOCKED"}
        blocked_count=len(blocked_ids)
        if blocked_count:
            blocked_candidate_turns+=1
            blocked_candidate_instances+=blocked_count

        selectable=plans if mode=="baseline" else [p for p in plans if p.candidate_id not in blocked_ids]
        if plans and not selectable:
            all_blocked_turns+=1

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
            elif mode=="exclude_blocked" and is_blocked(pre,matches[0]):
                blocked_active_drops[active["kind"]]+=1
                transition_reason="active_plan_now_confirmed_blocked"
                active=None
                active_steps=0

        if active is None and selectable:
            if mode=="baseline":
                chosen=plans[rng.randrange(len(plans))]
                selection_reason="fixed_rng_uniform_over_all_candidates"
            else:
                # Coupled rejection sampling: draw from the same full C_t
                # distribution as Baseline and reject only confirmed BLOCKED.
                # Therefore the paired runs remain identical until Baseline
                # actually draws a BLOCKED candidate.
                while True:
                    drawn=plans[rng.randrange(len(plans))]
                    if drawn.candidate_id not in blocked_ids:
                        chosen=drawn
                        break
                selection_reason="coupled_rng_reject_only_confirmed_blocked"
            plan_sequence+=1
            active={
                "sequence":plan_sequence,
                "kind":chosen.kind,
                "target":plain(chosen.target),
            }
            active_steps=0
            selected_counts[chosen.kind]+=1

        current_plan=None
        current_status=None
        if active is not None:
            matches=semantic_matches(plans,active)
            if matches:
                current_plan=matches[0]
                current_status=first_action_status(pre,current_plan)

        if current_plan is None:
            bundle=baseline_pass_bundle(pre)
            pass_turns+=1
            action_mode="pass_no_selected_plan"
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
        pre_hash=pre.canonical_hash
        pre_summary=state_summary(pre)
        active_before=plain(active) if active is not None else None

        env.step([plain(bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)
        completion=None

        if current_plan is not None and active is not None:
            completion=completion_from_states(current_plan,pre,post)
            active_steps+=1
            if completion.get("complete"):
                completed_counts[active["kind"]]+=1
                active=None
                active_steps=0

        post_plans=generate_plans(post)
        post_statuses=candidate_status_rows(post,post_plans)
        post_blocked=sum(1 for r in post_statuses if r["status"]=="BLOCKED")
        post_selectable=len(post_plans) if mode=="baseline" else len(post_plans)-post_blocked

        turns.append({
            "turn":turn,
            "pre_hash":pre_hash,
            "pre_state":pre_summary,
            "candidate_count":len(plans),
            "blocked_count":blocked_count,
            "selectable_count":len(selectable),
            "active_before_action":active_before,
            "current_status":plain(current_status),
            "selection_reason":selection_reason,
            "transition_reason":transition_reason,
            "action_mode":action_mode,
            "action_bundle":plain(bundle),
            "opponent_action_bundle":opp_bundle,
            "completion":plain(completion),
            "post_hash":post.canonical_hash,
            "post_state":state_summary(post),
            "next_replan":{
                "candidate_count":len(post_plans),
                "blocked_count":post_blocked,
                "selectable_count":post_selectable,
                "status_counts":dict(Counter(r["status"] for r in post_statuses)),
                "candidate_kind_counts":dict(Counter(p.kind for p in post_plans)),
            },
        })

        turn+=1

    final=bind_official_state(env.state[0].observation)
    rewards=[]
    for st in env.state:
        try: rewards.append(float(st.reward))
        except Exception: rewards.append(None)

    return {
        "mode":mode,
        "summary":{
            "ran_to_terminal":bool(env.done),
            "turns":turn,
            "terminal_self_cash":state_summary(final)["cash"],
            "official_rewards":rewards,
            "selected_plans_by_kind":dict(selected_counts),
            "completed_plans_by_kind":dict(completed_counts),
            "invalidated_plans_by_kind":dict(invalidated_counts),
            "timed_out_plans_by_kind":dict(timeout_counts),
            "blocked_active_drops_by_kind":dict(blocked_active_drops),
            "blocked_candidate_turns":blocked_candidate_turns,
            "blocked_candidate_instances":blocked_candidate_instances,
            "all_candidates_blocked_turns":all_blocked_turns,
            "pass_turns":pass_turns,
            "execution_error_count":len(execution_errors),
        },
        "final_state":state_summary(final),
        "turns":turns,
        "execution_errors":execution_errors,
    }


def first_action_divergence(b,p):
    n=min(len(b["turns"]),len(p["turns"]))
    for i in range(n):
        rb=b["turns"][i]
        rp=p["turns"][i]
        if rb["action_bundle"]!=rp["action_bundle"]:
            return i,rb,rp
    return None,None,None


def main():
    baseline=run_policy("baseline")
    p1=run_policy("exclude_blocked")

    idx,brow,prow=first_action_divergence(baseline,p1)
    if idx is None:
        divergence=None
    else:
        same_pre=(brow["pre_hash"]==prow["pre_hash"])
        same_opp=(brow["opponent_action_bundle"]==prow["opponent_action_bundle"])
        next_b=baseline["turns"][idx+1] if idx+1<len(baseline["turns"]) else None
        next_p=p1["turns"][idx+1] if idx+1<len(p1["turns"]) else None
        checklist={
            "action_difference_observed":brow["action_bundle"]!=prow["action_bundle"],
            "pre_and_post_state_observed":True,
            "official_world_effect_observed":brow["post_hash"]!=prow["post_hash"],
            "baseline_and_p1_compared_same_shape":(
                set(brow["action_bundle"])==set(prow["action_bundle"])=={"farmer","hands","market"}
                and len(brow["action_bundle"]["hands"])==len(prow["action_bundle"]["hands"])
            ),
            "next_state_to_current_replan_connected":(
                next_b is not None and next_p is not None
                and brow["next_replan"] is not None and prow["next_replan"] is not None
            ),
        }
        divergence={
            "turn":idx,
            "same_pre_state":same_pre,
            "same_opponent_action":same_opp,
            "baseline":brow,
            "p1":prow,
            "next_turn":{
                "baseline":{
                    "pre_state":next_b["pre_state"],
                    "active_before_action":next_b["active_before_action"],
                    "current_status":next_b["current_status"],
                    "action_bundle":next_b["action_bundle"],
                },
                "p1":{
                    "pre_state":next_p["pre_state"],
                    "active_before_action":next_p["active_before_action"],
                    "current_status":next_p["current_status"],
                    "action_bundle":next_p["action_bundle"],
                },
            },
            "fixed_observation_checklist":checklist,
            "fixed_5":all(checklist.values()) and same_pre and same_opp,
        }

    result={
        "schema":"exclude-confirmed-blocked-ab-v0",
        "purpose":"Test only the newly exposed Candidate -> confirmed BLOCKED -> Selection boundary.",
        "environment":{
            "seed":ENV_SEED,
            "policy_seed":POLICY_SEED,
            "opponent":"Seyamalam pinned v21, isolated module instance per run",
        },
        "status_rule":{
            "default":"UNKNOWN",
            "confirmed_blocked_only":{
                "kind":"prepare_for_plant",
                "condition":"current cash < Official seed cost * missing_seed_quantity",
            },
            "unknown_is_selectable":True,
            "executable_is_not_assumed":True,
            "executable_does_not_mean_valuable":True,
            "first_action_status_does_not_mean_plan_completion":True,
        },
        "comparison":{
            "baseline_selection":"uniform RNG over C_t",
            "p1_selection":"coupled rejection sampling over full C_t; reject only confirmed BLOCKED draws",
            "generator_changed":False,
            "projector_changed":False,
            "completion_changed":False,
            "priority_or_score_added":False,
        },
        "baseline":baseline,
        "p1":p1,
        "first_action_divergence":divergence,
        "terminal_delta_self":p1["summary"]["terminal_self_cash"]-baseline["summary"]["terminal_self_cash"],
    }

    Path("exclude_confirmed_blocked_ab_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "baseline_terminal":baseline["summary"]["terminal_self_cash"],
        "p1_terminal":p1["summary"]["terminal_self_cash"],
        "terminal_delta":result["terminal_delta_self"],
        "baseline_blocked_candidate_turns":baseline["summary"]["blocked_candidate_turns"],
        "p1_blocked_candidate_turns":p1["summary"]["blocked_candidate_turns"],
        "baseline_timed_out":baseline["summary"]["timed_out_plans_by_kind"],
        "p1_timed_out":p1["summary"]["timed_out_plans_by_kind"],
        "baseline_selected":baseline["summary"]["selected_plans_by_kind"],
        "p1_selected":p1["summary"]["selected_plans_by_kind"],
        "first_divergence_turn":divergence["turn"] if divergence else None,
        "first_divergence_same_pre":divergence["same_pre_state"] if divergence else None,
        "first_divergence_same_opponent":divergence["same_opponent_action"] if divergence else None,
        "first_divergence_fixed_5":divergence["fixed_5"] if divergence else None,
    },separators=(",",":")))


if __name__=="__main__":
    main()
