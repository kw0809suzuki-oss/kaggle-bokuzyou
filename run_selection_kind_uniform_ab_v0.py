#!/usr/bin/env python3
"""Selection Kind-Uniform A/B v0.

World-exposed reason for this experiment:
The first observed lost plant had a maintain_plant_today Candidate available
during fresh Selection opportunities, yet other jobs were selected.

Baseline:
    frozen Confirmed Circulation Body v0
    current candidate-uniform Selection with confirmed BLOCKED rejection

P1:
    same Body behavior except at fresh Selection:
    choose Plan kind uniformly among selectable kinds,
    then choose a Candidate uniformly within that kind.

No changes to Generator, status/BLOCKED rule, active Plan continuation,
Projector, Completion, PASS behavior, or Official World.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_plan_match,
)

ENV_SEED=body.ENV_SEED
POLICY_SEED=body.POLICY_SEED
MAX_PLAN_STEPS=body.MAX_PLAN_STEPS


def plain(x):
    return body.plain(x)


def semantic_matches(plans,spec):
    return [
        p for p in plans
        if semantic_plan_match(p,kind=spec["kind"],target=spec["target"])
    ]


def run_kind_uniform():
    opponent=body.load_opponent("kind_uniform")
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
        statuses=body.candidate_status_rows(pre,plans)
        blocked_ids={r["candidate_id"] for r in statuses if r["status"]=="BLOCKED"}
        blocked_count=len(blocked_ids)
        if blocked_count:
            blocked_candidate_turns+=1
            blocked_candidate_instances+=blocked_count

        selectable=[p for p in plans if p.candidate_id not in blocked_ids]
        if plans and not selectable:
            all_blocked_turns+=1

        transition_reason=None
        selection_reason=None
        selected_kind=None
        selectable_kind_counts=dict(Counter(p.kind for p in selectable))

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
            elif body.is_blocked(pre,matches[0]):
                blocked_active_drops[active["kind"]]+=1
                transition_reason="active_plan_now_confirmed_blocked"
                active=None
                active_steps=0

        if active is None and selectable:
            # Only experimental change:
            # kind-uniform -> candidate-uniform inside the chosen kind.
            kinds=list(dict.fromkeys(p.kind for p in selectable))
            selected_kind=kinds[rng.randrange(len(kinds))]
            within=[p for p in selectable if p.kind==selected_kind]
            chosen=within[rng.randrange(len(within))]

            plan_sequence+=1
            active={
                "sequence":plan_sequence,
                "kind":chosen.kind,
                "target":plain(chosen.target),
            }
            active_steps=0
            selected_counts[chosen.kind]+=1
            selection_reason="uniform_plan_kind_then_uniform_candidate_within_kind"

        current_plan=None
        current_status=None
        if active is not None:
            matches=semantic_matches(plans,active)
            if matches:
                current_plan=matches[0]
                current_status=body.first_action_status(pre,current_plan)

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
        pre_summary=body.state_summary(pre)
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
        post_statuses=body.candidate_status_rows(post,post_plans)
        post_blocked=sum(1 for r in post_statuses if r["status"]=="BLOCKED")
        post_selectable=len(post_plans)-post_blocked

        turns.append({
            "turn":turn,
            "pre_hash":pre_hash,
            "pre_state":pre_summary,
            "candidate_count":len(plans),
            "blocked_count":blocked_count,
            "selectable_count":len(selectable),
            "selectable_kind_counts":selectable_kind_counts,
            "selected_kind":selected_kind,
            "active_before_action":active_before,
            "current_status":plain(current_status),
            "selection_reason":selection_reason,
            "transition_reason":transition_reason,
            "action_mode":action_mode,
            "action_bundle":plain(bundle),
            "opponent_action_bundle":opp_bundle,
            "completion":plain(completion),
            "post_hash":post.canonical_hash,
            "post_state":body.state_summary(post),
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
        "mode":"kind_uniform",
        "summary":{
            "ran_to_terminal":bool(env.done),
            "turns":turn,
            "terminal_self_cash":body.state_summary(final)["cash"],
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
        "final_state":body.state_summary(final),
        "turns":turns,
        "execution_errors":execution_errors,
    }


def tile_class(tile):
    if tile is None: return "EMPTY"
    if tile=="LOCKED": return "LOCKED"
    if isinstance(tile,dict):
        if tile.get("kind")=="PLANT":
            return "PLANT:"+str(tile.get("crop"))
        return str(tile.get("kind"))
    return str(tile)


def aggregate_inventory(private):
    c=Counter()
    for inv in private.get("inventories",[]) or []:
        if not isinstance(inv,dict): continue
        for k,v in inv.items():
            if isinstance(v,(int,float)) and v:
                c[str(k)]+=float(v)
    return c


def replay_metrics(run):
    env=make("kaggriculture",configuration={"seed":ENV_SEED},debug=False)
    env.reset(num_agents=2)

    planted=Counter()
    weeded=Counter()
    harvested=Counter()
    sold=Counter()
    seed_spend=0.0
    cash_return=0.0
    first_cash_return_turn=None
    first_cash_return_day_hour=None
    positive_cash_events=0

    for i,row in enumerate(run["turns"]):
        pre=bind_official_state(env._Environment__get_shared_state(0)["observation"])
        if pre.canonical_hash!=row["pre_hash"]:
            raise RuntimeError(f"replay pre mismatch {run['mode']} turn {i}")
        pre_raw=pre.raw(); p=pre_raw["player"]
        pre_cash=float(pre_raw["farms"][p].get("money",0) or 0)
        pre_inv=aggregate_inventory(pre_raw["private"])
        pre_shed=pre_raw["private"].get("shed",{}) or {}

        b=plain(row["action_bundle"])
        o=plain(row["opponent_action_bundle"])
        env.step([b,o])
        post=bind_official_state(env.state[0].observation)
        if post.canonical_hash!=row["post_hash"]:
            raise RuntimeError(f"replay post mismatch {run['mode']} turn {i}")
        post_raw=post.raw()
        post_cash=float(post_raw["farms"][p].get("money",0) or 0)
        post_inv=aggregate_inventory(post_raw["private"])
        post_shed=post_raw["private"].get("shed",{}) or {}

        cash_delta=post_cash-pre_cash
        market=b.get("market",[]) or []
        if cash_delta<0 and any(isinstance(op,list) and op and op[0]=="BUY_SEED" for op in market):
            seed_spend+=-cash_delta
        if cash_delta>0:
            cash_return+=cash_delta
            positive_cash_events+=1
            if first_cash_return_turn is None:
                first_cash_return_turn=i
                first_cash_return_day_hour=[int(pre_raw["day"]),int(pre_raw["hour"])]

        pre_tiles=pre_raw["farms"][p]["tiles"]
        post_tiles=post_raw["farms"][p]["tiles"]
        for y in range(len(pre_tiles)):
            for x in range(len(pre_tiles[y])):
                a=pre_tiles[y][x]; z=post_tiles[y][x]
                ac=tile_class(a); zc=tile_class(z)
                if not ac.startswith("PLANT:") and zc.startswith("PLANT:"):
                    planted[zc.split(":",1)[1]]+=1
                if ac.startswith("PLANT:") and zc=="WEED":
                    weeded[ac.split(":",1)[1]]+=1

        tokens=[]
        tokens.extend(b.get("farmer",[]) or [])
        for h in b.get("hands",[]) or []:
            if isinstance(h,list): tokens.extend(h)
            else: tokens.append(h)
        if "HARVEST" in tokens:
            for prod in set(pre_inv)|set(post_inv):
                d=post_inv[prod]-pre_inv[prod]
                if d>0: harvested[prod]+=d

        for op in market:
            if isinstance(op,list) and len(op)>=2 and op[0]=="SELL":
                prod=str(op[1])
                before=float(pre_shed.get(prod,0) or 0)
                after=float(post_shed.get(prod,0) or 0)
                if before>after: sold[prod]+=before-after

    active_turns=sum(1 for r in run["turns"] if r.get("active_before_action") is not None)
    pass_turns=sum(1 for r in run["turns"] if str(r.get("action_mode","")).startswith("pass_"))
    all_blocked_wait=sum(
        1 for r in run["turns"]
        if r["candidate_count"]>0 and r["selectable_count"]==0
        and str(r.get("action_mode","")).startswith("pass_")
    )

    return {
        "terminal_self":run["summary"]["terminal_self_cash"],
        "seed_spend_total":seed_spend,
        "cash_return_total":cash_return,
        "positive_cash_events":positive_cash_events,
        "first_cash_return_turn":first_cash_return_turn,
        "first_cash_return_day_hour":first_cash_return_day_hour,
        "planted_units":sum(planted.values()),
        "planted_by_crop":dict(planted),
        "weedized_units":sum(weeded.values()),
        "weedized_by_crop":dict(weeded),
        "harvested_units":sum(harvested.values()),
        "harvested_by_product":dict(harvested),
        "sold_units":sum(sold.values()),
        "sold_by_product":dict(sold),
        "active_turns":active_turns,
        "pass_turns":pass_turns,
        "all_blocked_pass_turns":all_blocked_wait,
        "selected_plans_by_kind":run["summary"]["selected_plans_by_kind"],
        "completed_plans_by_kind":run["summary"]["completed_plans_by_kind"],
        "terminal_state":run["final_state"],
    }


def first_divergence(b,p):
    for i,(rb,rp) in enumerate(zip(b["turns"],p["turns"])):
        if rb["action_bundle"]!=rp["action_bundle"]:
            same_pre=rb["pre_hash"]==rp["pre_hash"]
            same_opp=rb["opponent_action_bundle"]==rp["opponent_action_bundle"]
            return {
                "turn":i,
                "same_pre_state":same_pre,
                "same_opponent_action":same_opp,
                "baseline":{
                    "candidate_count":rb["candidate_count"],
                    "blocked_count":rb["blocked_count"],
                    "selectable_count":rb["selectable_count"],
                    "active":rb["active_before_action"],
                    "action":rb["action_bundle"],
                    "post_hash":rb["post_hash"],
                    "next_replan":rb["next_replan"],
                },
                "p1":{
                    "candidate_count":rp["candidate_count"],
                    "blocked_count":rp["blocked_count"],
                    "selectable_count":rp["selectable_count"],
                    "selectable_kind_counts":rp.get("selectable_kind_counts"),
                    "selected_kind":rp.get("selected_kind"),
                    "active":rp["active_before_action"],
                    "action":rp["action_bundle"],
                    "post_hash":rp["post_hash"],
                    "next_replan":rp["next_replan"],
                },
                "official_post_difference":rb["post_hash"]!=rp["post_hash"],
            }
    return None


def main():
    # Baseline is the frozen Body itself, not a reimplementation.
    baseline=body.run_policy("exclude_blocked")
    p1=run_kind_uniform()

    bmetrics=replay_metrics(baseline)
    pmetrics=replay_metrics(p1)
    divergence=first_divergence(baseline,p1)

    result={
        "schema":"selection-kind-uniform-ab-v0",
        "purpose":"Test whether Candidate multiplicity changes time allocation across job kinds.",
        "trigger_evidence":{
            "probe":"One Lost Plant Trace v0",
            "observed_case":1,
            "meaning":"maintenance Candidate existed during fresh Selection opportunities, but another Candidate was selected",
        },
        "environment":{
            "seed":ENV_SEED,
            "policy_seed":POLICY_SEED,
            "opponent":"Seyamalam pinned v21",
        },
        "comparison":{
            "baseline":"frozen Body: uniform Candidate selection with confirmed BLOCKED rejection",
            "p1":"uniform selectable Plan kind, then uniform Candidate within selected kind",
            "generator_changed":False,
            "blocked_rule_changed":False,
            "active_plan_continuation_changed":False,
            "projector_changed":False,
            "completion_changed":False,
            "priority_score_utility_added":False,
        },
        "baseline":{"summary":baseline["summary"],"metrics":bmetrics},
        "p1":{"summary":p1["summary"],"metrics":pmetrics},
        "delta":{
            "terminal_self":pmetrics["terminal_self"]-bmetrics["terminal_self"],
            "seed_spend":pmetrics["seed_spend_total"]-bmetrics["seed_spend_total"],
            "cash_return":pmetrics["cash_return_total"]-bmetrics["cash_return_total"],
            "planted_units":pmetrics["planted_units"]-bmetrics["planted_units"],
            "weedized_units":pmetrics["weedized_units"]-bmetrics["weedized_units"],
            "harvested_units":pmetrics["harvested_units"]-bmetrics["harvested_units"],
            "sold_units":pmetrics["sold_units"]-bmetrics["sold_units"],
            "active_turns":pmetrics["active_turns"]-bmetrics["active_turns"],
            "pass_turns":pmetrics["pass_turns"]-bmetrics["pass_turns"],
        },
        "first_action_divergence":divergence,
        "boundary":{
            "single_seed_only":True,
            "not_an_adoption_decision":True,
            "does_not_establish_kind_uniform_as_correct_policy":True,
        },
    }

    Path("selection_kind_uniform_ab_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "baseline":bmetrics,
        "p1":pmetrics,
        "delta":result["delta"],
        "first_divergence":divergence,
    },separators=(",",":")))


if __name__=="__main__":
    main()
