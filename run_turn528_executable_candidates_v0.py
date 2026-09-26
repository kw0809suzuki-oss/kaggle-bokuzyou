#!/usr/bin/env python3
"""Ask Official World which turn-528 PlanCandidates are executable now.

No generator, projector, completion, or selection rule is changed.

Replay the exact Surface-Reuse whole trajectory to turn 528 / Day22 h0.
At that one Official PreState:
  C_t = all generated PlanCandidates

For every c in C_t:
  - project exactly one ActionBundle with the existing projector
  - fork the same PreState
  - apply the same pinned opponent Action
  - compare against one all-PASS baseline through Official World

Define "first_action_effect_observed" only as:
  the self-controlled Official post-state (own farm/private plus shared market)
  differs from the PASS baseline.

This is an observation of the projected first action, not a proof that the whole
Plan can eventually complete.
"""

import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

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
TARGET_TURN=528
TARGET_DAY=22
TARGET_HOUR=0


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


def self_effect_view(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    return {
        "farm":plain(raw["farms"][p]),
        "private":plain(raw["private"]),
        "market":plain(raw["market"]),
    }


def money(snapshot):
    raw=snapshot.raw()
    return float(raw["farms"][raw["player"]].get("money",0) or 0)


def seed_qty(snapshot,crop):
    raw=snapshot.raw()
    return int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)


def same_semantic_candidate_present(post, plan):
    return any(
        semantic_plan_match(p,kind=plan.kind,target=plain(plan.target))
        for p in generate_plans(post)
    )


def first_action_label(bundle):
    nonpass=[]
    if bundle.get("farmer") != ["PASS"]:
        nonpass.append(["farmer",plain(bundle.get("farmer"))])
    for i,a in enumerate(bundle.get("hands",[]) or []):
        if a != ["PASS"]:
            nonpass.append([f"hand_{i}",plain(a)])
    for o in bundle.get("market",[]) or []:
        nonpass.append(["market",plain(o)])
    return nonpass


def replay_to_turn():
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

        # Apply the same active-plan transition and selection logic used by
        # run_short_plan_terminal_rollout_v0.py, but stop before executing
        # turn 528 so the exact selected candidate is also known.
        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not ms:
                active=None
                active_steps=0

        selected_now=None
        if active is None and plans:
            chosen=plans[rng.randrange(len(plans))]
            seq+=1
            active={"sequence":seq,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0
            selected_now=chosen

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms:
                current=ms[0]

        if turn==TARGET_TURN:
            raw=pre.raw()
            if not (int(raw["day"])==TARGET_DAY and int(raw["hour"])==TARGET_HOUR):
                raise RuntimeError(
                    f"turn {TARGET_TURN} replay mismatch: {raw['day']} h{raw['hour']}"
                )
            return {
                "env":env,
                "pre":pre,
                "plans":plans,
                "active":plain(active),
                "selected_now":selected_now,
                "current":current,
                "opponent_observation":plain(s1["observation"]),
            }

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

    raise RuntimeError("target turn not reached")


def main():
    replay=replay_to_turn()
    env=replay["env"]
    pre=replay["pre"]
    plans=replay["plans"]
    active=replay["active"]
    selected_now=replay["selected_now"]
    current=replay["current"]

    if len(plans)!=86:
        raise RuntimeError(f"expected 86 candidates, got {len(plans)}")

    # One pinned opponent action for all 87 forks (PASS baseline + 86 candidates).
    opp_bundle=plain(opponent.agent(replay["opponent_observation"]))

    baseline_bundle=baseline_pass_bundle(pre)
    b_env=copy.deepcopy(env)
    b_env.step([copy.deepcopy(baseline_bundle),copy.deepcopy(opp_bundle)])
    baseline_post=bind_official_state(b_env.state[0].observation)
    baseline_effect=self_effect_view(baseline_post)

    rows=[]
    executable_counts=Counter()
    action_counts=Counter()
    kind_total=Counter()
    kind_exec=Counter()

    for i,plan in enumerate(plans):
        kind_total[plan.kind]+=1
        bundle=project_short_plan(pre,plan)
        action_label=first_action_label(bundle)
        action_key=json.dumps(action_label,sort_keys=True,separators=(",",":"))
        action_counts[action_key]+=1

        p_env=copy.deepcopy(env)
        p_env.step([plain(bundle),copy.deepcopy(opp_bundle)])
        post=bind_official_state(p_env.state[0].observation)

        effect=(self_effect_view(post)!=baseline_effect)
        if effect:
            kind_exec[plan.kind]+=1

        comp=completion_from_states(plan,pre,post)
        same_present=same_semantic_candidate_present(post,plan)

        row={
            "index":i,
            "candidate":plan.to_dict(),
            "projected_action_bundle":plain(bundle),
            "projected_nonpass_actions":action_label,
            "first_action_effect_observed":effect,
            "completion_after_one_turn":plain(comp),
            "same_semantic_candidate_present_post":same_present,
            "pre_cash":money(pre),
            "post_cash":money(post),
            "baseline_post_cash":money(baseline_post),
        }

        if plan.kind=="prepare_for_plant":
            crop=str(plan.target["crop"])
            row["prepare_details"]={
                "crop":crop,
                "requested_quantity":int(plan.target["missing_seed_quantity"]),
                "official_unit_seed_cost":int(CROPS[crop]["seed"]),
                "required_cost_for_projected_order":int(CROPS[crop]["seed"])*int(plan.target["missing_seed_quantity"]),
                "pre_seed_quantity":seed_qty(pre,crop),
                "post_seed_quantity":seed_qty(post,crop),
                "baseline_post_seed_quantity":seed_qty(baseline_post,crop),
            }

        rows.append(row)

    executable=[r for r in rows if r["first_action_effect_observed"]]
    nonexec=[r for r in rows if not r["first_action_effect_observed"]]

    selected_row=None
    if current is not None:
        for r in rows:
            if (
                r["candidate"]["kind"]==current.kind
                and r["candidate"]["target"]==plain(current.target)
            ):
                selected_row=r
                break

    result={
        "schema":"turn528-executable-candidates-v0",
        "purpose":"Observe C_t versus first-action-effect set E_t at the exact turn-528 Official State without changing the machine.",
        "definition":{
            "C_t":"all current PlanCandidates returned by the unchanged generator",
            "E_t":"c in C_t whose existing projected first ActionBundle produces an Official self/market state difference relative to same-State all-PASS, with identical opponent Action",
            "boundary":"E_t is first-action executability only; it does not prove eventual Plan completion or value",
        },
        "pre_state":{
            "turn":TARGET_TURN,
            "day":pre.raw()["day"],
            "hour":pre.raw()["hour"],
            "cash":money(pre),
            "candidate_count":len(plans),
            "candidate_kind_counts":dict(kind_total),
            "active_after_selection":active,
            "selected_candidate":current.to_dict() if current is not None else None,
        },
        "official_control":{
            "baseline_action_bundle":baseline_bundle,
            "same_opponent_action_bundle":opp_bundle,
            "baseline_post_cash":money(baseline_post),
        },
        "summary":{
            "C_count":len(rows),
            "E_count":len(executable),
            "non_executable_count":len(nonexec),
            "E_by_kind":dict(kind_exec),
            "C_by_kind":dict(kind_total),
            "selected_candidate_first_action_effect_observed":(
                selected_row["first_action_effect_observed"] if selected_row else None
            ),
            "selected_candidate_same_semantic_candidate_present_post":(
                selected_row["same_semantic_candidate_present_post"] if selected_row else None
            ),
        },
        "selected_candidate_observation":selected_row,
        "executable_candidates":executable,
        "non_executable_candidates":nonexec,
        "all_candidates":rows,
    }

    Path("turn528_executable_candidates_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    sel=selected_row or {}
    print("SUMMARY "+json.dumps({
        "C_count":result["summary"]["C_count"],
        "E_count":result["summary"]["E_count"],
        "non_executable_count":result["summary"]["non_executable_count"],
        "C_by_kind":result["summary"]["C_by_kind"],
        "E_by_kind":result["summary"]["E_by_kind"],
        "selected_kind":sel.get("candidate",{}).get("kind"),
        "selected_target":sel.get("candidate",{}).get("target"),
        "selected_action":sel.get("projected_nonpass_actions"),
        "selected_effect":sel.get("first_action_effect_observed"),
        "selected_prepare_details":sel.get("prepare_details"),
        "selected_same_candidate_post":sel.get("same_semantic_candidate_present_post"),
    },separators=(",",":")))


if __name__=="__main__":
    main()
