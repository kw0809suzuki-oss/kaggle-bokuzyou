#!/usr/bin/env python3
"""Observe whether the last active job is blocked or cut off by terminal.

No machine behavior is changed.

Replay the P1 policy from Exclude Confirmed Blocked A/B v0 to turn 718
(Day29 h22). Fork the exact Official PreState:
- Baseline: PASS
- P1: the actual projected first action of the active job

Use the same opponent action in both forks and observe:
- Official PostState
- whether each fork is terminal
- whether the P1 job completed
- whether the same semantic job is still visible in the terminal PostState
- what generate_plans() sees on that terminal snapshot

This does not invent a post-terminal continuation.
"""

from __future__ import annotations
import copy, importlib.util, json, random
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
TARGET_TURN=718


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def load_opponent():
    spec=importlib.util.spec_from_file_location("seyamalam_terminal_cut",OPP_PATH)
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
            return {
                "status":"BLOCKED",
                "reason":"cash_below_official_seed_cost",
                "cash":cash,
                "required_cost":required,
            }
    return {"status":"UNKNOWN","reason":"not_closed_by_current_evidence"}


def state_view(snapshot):
    raw=snapshot.raw(); p=raw["player"]
    plans=generate_plans(snapshot)
    status_rows=[first_action_status(snapshot,q) for q in plans]
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(raw["farms"][p].get("money",0) or 0),
        "farmer":plain(raw["farms"][p].get("farmer")),
        "seeds":plain(raw["private"].get("seeds",{}) or {}),
        "candidate_count":len(plans),
        "status_counts":dict(Counter(s["status"] for s in status_rows)),
        "candidate_kind_counts":dict(Counter(q.kind for q in plans)),
    }


def main():
    opponent=load_opponent()
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
        plans=generate_plans(pre)
        blocked_ids={
            q.candidate_id
            for q in plans
            if first_action_status(pre,q)["status"]=="BLOCKED"
        }
        selectable=[q for q in plans if q.candidate_id not in blocked_ids]

        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not ms:
                active=None; active_steps=0
            elif first_action_status(pre,ms[0])["status"]=="BLOCKED":
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

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms:
                current=ms[0]

        if turn==TARGET_TURN:
            if current is None:
                raise RuntimeError("no active current plan at target turn")
            raw=pre.raw()
            if not (int(raw["day"])==29 and int(raw["hour"])==22):
                raise RuntimeError(f"target state mismatch: {raw['day']} h{raw['hour']}")

            actual_bundle=project_short_plan(pre,current)
            pass_bundle=baseline_pass_bundle(pre)
            opp_bundle=plain(opponent.agent(s1["observation"]))

            b_env=copy.deepcopy(env)
            p_env=copy.deepcopy(env)
            b_env.step([copy.deepcopy(pass_bundle),copy.deepcopy(opp_bundle)])
            p_env.step([copy.deepcopy(actual_bundle),copy.deepcopy(opp_bundle)])

            b_post=bind_official_state(b_env.state[0].observation)
            p_post=bind_official_state(p_env.state[0].observation)

            p_completion=completion_from_states(current,pre,p_post)
            b_completion=completion_from_states(current,pre,b_post)

            p_post_plans=generate_plans(p_post)
            b_post_plans=generate_plans(b_post)

            same_p=bool(semantic_matches(p_post_plans,active))
            same_b=bool(semantic_matches(b_post_plans,active))

            result={
                "schema":"terminal-cut-boundary-v0",
                "purpose":"Observe whether the last active job fails in Official World or is merely still in progress when the season terminates.",
                "pre_state":state_view(pre),
                "active_plan":{
                    "sequence":active["sequence"],
                    "kind":current.kind,
                    "target":plain(current.target),
                    "first_action_status":first_action_status(pre,current),
                    "active_steps_before_action":active_steps,
                },
                "same_opponent_action_bundle":opp_bundle,
                "baseline":{
                    "action_bundle":plain(pass_bundle),
                    "post_state":state_view(b_post),
                    "completion_against_same_plan":plain(b_completion),
                    "same_semantic_plan_present_post":same_b,
                    "env_done_after_step":bool(b_env.done),
                    "official_reward":[float(st.reward) for st in b_env.state],
                },
                "p1":{
                    "action_bundle":plain(actual_bundle),
                    "post_state":state_view(p_post),
                    "completion_against_same_plan":plain(p_completion),
                    "same_semantic_plan_present_post":same_p,
                    "env_done_after_step":bool(p_env.done),
                    "official_reward":[float(st.reward) for st in p_env.state],
                },
                "next_state_visibility":{
                    "terminal_snapshot_exists":True,
                    "p1_generator_candidate_count":len(p_post_plans),
                    "p1_same_plan_still_visible":same_p,
                    "no_further_official_action_turn_because_env_done":bool(p_env.done),
                },
                "fixed_observation_checklist":{
                    "action_difference_observed":plain(pass_bundle)!=plain(actual_bundle),
                    "pre_and_post_state_observed":True,
                    "official_world_effect_observed":p_post.canonical_hash!=b_post.canonical_hash,
                    "baseline_and_p1_compared_same_shape":(
                        set(pass_bundle)==set(actual_bundle)=={"farmer","hands","market"}
                        and len(pass_bundle["hands"])==len(actual_bundle["hands"])
                    ),
                    "next_state_to_current_visibility_checked":True,
                },
                "boundary":{
                    "no_terminal_aware_logic_added":True,
                    "no_post_terminal_continuation_invented":True,
                    "plan_completion_and_terminal_are_kept_separate":True,
                },
            }
            result["fixed_5"]=all(result["fixed_observation_checklist"].values())

            Path("terminal_cut_boundary_v0.json").write_text(
                json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
            )
            print("SUMMARY "+json.dumps({
                "pre_day_hour":[result["pre_state"]["day"],result["pre_state"]["hour"]],
                "active_plan":result["active_plan"],
                "baseline_action":result["baseline"]["action_bundle"],
                "p1_action":result["p1"]["action_bundle"],
                "baseline_post":result["baseline"]["post_state"],
                "p1_post":result["p1"]["post_state"],
                "baseline_done":result["baseline"]["env_done_after_step"],
                "p1_done":result["p1"]["env_done_after_step"],
                "p1_complete":result["p1"]["completion_against_same_plan"]["complete"],
                "p1_same_plan_post":result["p1"]["same_semantic_plan_present_post"],
                "p1_post_candidates":result["next_state_visibility"]["p1_generator_candidate_count"],
                "fixed_5":result["fixed_5"],
            },separators=(",",":")))
            return

        self_bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp_bundle=plain(opponent.agent(s1["observation"]))
        env.step([plain(self_bundle),opp_bundle])
        post=bind_official_state(env.state[0].observation)

        if current is not None and active is not None:
            comp=completion_from_states(current,pre,post)
            active_steps+=1
            if comp.get("complete"):
                active=None; active_steps=0

        turn+=1

    raise RuntimeError("target turn not reached")


if __name__=="__main__":
    main()
