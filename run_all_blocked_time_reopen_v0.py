#!/usr/bin/env python3
"""Observe the first all-BLOCKED -> selectable reopening caused by Official time.

No generator, status rule, selector, projector, or action is changed.

Replay the P1 policy from Exclude Confirmed Blocked A/B v0:
- full Candidate set is preserved
- confirmed BLOCKED means only prepare_for_plant with cash below Official seed cost
- UNKNOWN remains selectable
- selection uses coupled-style rejection of confirmed BLOCKED draws

Capture the exact transition:
  turn 599 / Day24 h23
  -> existing PASS
  -> Official World
  -> turn 600 / Day25 h0

Record raw plant State and Candidate/Status changes.
"""

from __future__ import annotations
import copy, importlib.util, json, random, sys
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
TARGET_TURN=599


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def load_opponent():
    spec=importlib.util.spec_from_file_location("seyamalam_v21_time_reopen",OPP_PATH)
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
        cost=int(CROPS[crop]["seed"])*qty
        if cash < cost:
            return {
                "status":"BLOCKED",
                "reason":"cash_below_official_seed_cost",
                "cash":cash,
                "required_cost":cost,
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


def candidate_rows(snapshot):
    plans=generate_plans(snapshot)
    out=[]
    for p in plans:
        out.append({
            "candidate_id":p.candidate_id,
            "kind":p.kind,
            "target":plain(p.target),
            "status":first_action_status(snapshot,p),
        })
    return out


def compact(snapshot):
    raw=snapshot.raw(); p=raw["player"]
    rows=candidate_rows(snapshot)
    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(raw["farms"][p].get("money",0) or 0),
        "seeds":plain(raw["private"].get("seeds",{}) or {}),
        "farmer":plain(raw["farms"][p].get("farmer")),
        "plants":plant_rows(snapshot),
        "candidate_count":len(rows),
        "status_counts":dict(Counter(r["status"]["status"] for r in rows)),
        "kind_counts":dict(Counter(r["kind"] for r in rows)),
        "candidate_rows":rows,
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
        blocked={p.candidate_id for p in plans if first_action_status(pre,p)["status"]=="BLOCKED"}
        selectable=[p for p in plans if p.candidate_id not in blocked]

        if active is not None:
            ms=semantic_matches(plans,active)
            if active_steps>=MAX_PLAN_STEPS or not ms:
                active=None; active_steps=0
            elif first_action_status(pre,ms[0])["status"]=="BLOCKED":
                active=None; active_steps=0

        if active is None and selectable:
            while True:
                drawn=plans[rng.randrange(len(plans))]
                if drawn.candidate_id not in blocked:
                    chosen=drawn
                    break
            seq+=1
            active={"sequence":seq,"kind":chosen.kind,"target":plain(chosen.target)}
            active_steps=0

        current=None
        if active is not None:
            ms=semantic_matches(plans,active)
            if ms: current=ms[0]

        bundle=baseline_pass_bundle(pre) if current is None else project_short_plan(pre,current)
        opp_bundle=plain(opponent.agent(s1["observation"]))

        if turn==TARGET_TURN:
            before=compact(pre)
            if not (before["day"]==24 and before["hour"]==23):
                raise RuntimeError(f"target mismatch: {before['day']} h{before['hour']}")
            if before["candidate_count"]<=0 or before["status_counts"]!={"BLOCKED":before["candidate_count"]}:
                raise RuntimeError("target pre-state is not all BLOCKED")
            if plain(bundle)!={"farmer":["PASS"],"hands":[],"market":[]}:
                raise RuntimeError(f"unexpected self action at target: {bundle}")

            env.step([plain(bundle),opp_bundle])
            post=bind_official_state(env.state[0].observation)
            after=compact(post)

            before_ids={r["candidate_id"] for r in before["candidate_rows"]}
            appeared=[r for r in after["candidate_rows"] if r["candidate_id"] not in before_ids]
            appeared_nonblocked=[r for r in appeared if r["status"]["status"]!="BLOCKED"]

            result={
                "schema":"all-blocked-time-reopen-v0",
                "purpose":"Observe the first Official time transition that reopens selection from an all-confirmed-BLOCKED state.",
                "pre_state":before,
                "self_action_bundle":plain(bundle),
                "opponent_action_bundle":opp_bundle,
                "post_state":after,
                "appeared_candidates":appeared,
                "appeared_nonblocked_candidates":appeared_nonblocked,
                "boundary":{
                    "no_new_intervention":True,
                    "generator_unchanged":True,
                    "status_rule_unchanged":True,
                    "selector_unchanged":True,
                    "pre_all_candidates_blocked":before["status_counts"]=={"BLOCKED":before["candidate_count"]},
                    "post_has_nonblocked_candidates":sum(
                        v for k,v in after["status_counts"].items() if k!="BLOCKED"
                    )>0,
                },
            }
            Path("all_blocked_time_reopen_v0.json").write_text(
                json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
            )
            print("SUMMARY "+json.dumps({
                "pre_day_hour":[before["day"],before["hour"]],
                "post_day_hour":[after["day"],after["hour"]],
                "pre_status_counts":before["status_counts"],
                "post_status_counts":after["status_counts"],
                "pre_kind_counts":before["kind_counts"],
                "post_kind_counts":after["kind_counts"],
                "plants_before":before["plants"],
                "plants_after":after["plants"],
                "appeared_nonblocked":[
                    {"kind":r["kind"],"target":r["target"],"status":r["status"]}
                    for r in appeared_nonblocked
                ],
            },separators=(",",":")))
            return

        env.step([plain(bundle),opp_bundle])
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
