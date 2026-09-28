#!/usr/bin/env python3
"""Contract checks for the Strong Model v0 reimplementation.

These checks target the failure mode of the previous implementation:
purpose must survive across turns, continuation must be valued from current
Official State, bundles may contain concurrent work, and terminal feasibility
comes from Official timing rather than a fixed calendar cutoff.
"""
from __future__ import annotations

import copy
import json

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from strong_model_v0_reimplementation.agent import agent, reset_agent, debug_state
from strong_model_v0_reimplementation.jobs import (
    Job,fresh_jobs,materialize_active,preferred_units,unit_action,
)
from strong_model_v0_reimplementation.planner import Settings,choose,schedule,terminal_envelope,plan_bundle


def plain(v):
    if isinstance(v,dict): return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [plain(x) for x in v]
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    if hasattr(v,"items"): return {str(k):plain(x) for k,x in v.items()}
    return v


def initial_raw(seed=92809901):
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    return plain(env._Environment__get_shared_state(0)["observation"])


def feed_purpose_check():
    raw=initial_raw()
    raw["step"]=0
    p=raw["player"]
    cow=rules._new_animal("COW",0)
    cow["consecutive_unfed"]=1
    cow["fed_today"]=False
    raw["farms"][p]["tiles"][0][0]=cow
    raw["private"]["shed"]["WHEAT"]=1
    raw["farms"][p]["farmer"]=[4,4]
    raw["private"]["inventories"]=[{}]

    key="ext:feed:0:0:0:COW"
    job=Job(key,"feed_animal","maintenance",{"tile":[0,0],"animal":"COW","day":0},1000,0,True,False)
    a,_=unit_action(job,raw,0,10)
    assert a==["PICKUP","WHEAT",1],a

    nxt=copy.deepcopy(raw)
    nxt["private"]["shed"]["WHEAT"]=0
    nxt["private"]["inventories"]=[{"WHEAT":1}]
    spec=job.spec()
    carried=materialize_active(spec,nxt)
    assert carried is not None
    units=preferred_units(carried,nxt,{0},10)
    assert units==[0]
    a2,_=unit_action(carried,nxt,0,10)
    assert a2[0]!="DROP",a2
    assert a2 in (["WEST"],["NORTH"],["FEED"]),a2
    return {"pickup":a,"next":a2}


def continuation_value_check():
    raw=initial_raw()
    raw["step"]=0
    p=raw["player"]
    cow=rules._new_animal("COW",0)
    cow["consecutive_unfed"]=1
    cow["fed_today"]=False
    raw["farms"][p]["tiles"][0][0]=cow
    raw["private"]["shed"]["WHEAT"]=1
    key="ext:feed:0:0:0:COW"
    spec={"key":key,"kind":"feed_animal","category":"maintenance","target":{"tile":[0,0],"animal":"COW","day":0}}
    chosen,_,active,continuation=choose(raw,Settings(),{key:spec})
    row=[j for j in active if j.key==key]
    assert row and row[0].central_delta>0,row
    assert key in continuation.scheduled,continuation.scheduled
    return {
        "active_value":row[0].central_delta,
        "continuation_scheduled":continuation.scheduled,
        "chosen_scheduled":chosen.scheduled,
    }


def parallel_bundle_check():
    raw=initial_raw()
    raw["step"]=0
    p=raw["player"]
    raw["farms"][p]["hands"]=[[4,4]]
    raw["private"]["inventories"]=[{},{}]

    plant=rules._new_plant("WHEAT",0,24)
    plant["consecutive_unwatered"]=1
    raw["farms"][p]["tiles"][0][0]=plant
    animal=rules._new_animal("GOOSE",0)
    animal["yield_units"]=1
    raw["farms"][p]["tiles"][1][0]=animal

    jobs=fresh_jobs(raw,720,24,10)
    water=next(j for j in jobs if j.kind=="maintain_plant_today" and j.target["tile"]==[0,0])
    harvest=next(j for j in jobs if j.kind=="harvest_animal" and j.target["tile"]==[0,1])
    action,scheduled=schedule(raw,Settings(),[water,harvest],{water.key,harvest.key})
    unit_actions=[action["farmer"],*action["hands"]]
    nonpass=sum(1 for a in unit_actions if a!=["PASS"])
    assert nonpass==2,(action,scheduled)
    return {"scheduled":scheduled,"actions":unit_actions}


def timing_not_fixed_day_check():
    raw=initial_raw()
    raw["step"]=27*24
    raw["day"]=27
    raw["hour"]=0
    jobs=fresh_jobs(raw,720,24,10)
    wheat27=[j for j in jobs if j.target.get("crop")=="WHEAT" and j.category=="production_start"]
    assert wheat27 and max(j.central_delta for j in wheat27)>0,wheat27

    raw2=copy.deepcopy(raw)
    raw2["step"]=28*24
    raw2["day"]=28
    jobs2=fresh_jobs(raw2,720,24,10)
    wheat28=[j for j in jobs2 if j.target.get("crop")=="WHEAT" and j.category=="production_start"]
    assert wheat28 and max(j.central_delta for j in wheat28)<=0,wheat28
    return {
        "day27_best_wheat":max(j.central_delta for j in wheat27),
        "day28_best_wheat":max(j.central_delta for j in wheat28),
    }


def existing_shortplan_bridge_check():
    raw=initial_raw()
    raw["step"]=0
    jobs=fresh_jobs(raw,720,24,10)
    base=[j for j in jobs if j.base_plan is not None]
    assert base, "existing ShortPlan candidates were not reused"
    assert any(j.base_plan.kind=="prepare_for_plant" for j in base),[j.kind for j in base]
    return {"base_jobs":len(base),"kinds":sorted({j.base_plan.kind for j in base})}


def initial_choice_probe():
    raw=initial_raw(92804001)
    raw["step"]=0
    cfg=Settings()
    jobs=fresh_jobs(raw,cfg.episodeSteps,cfg.turnsPerDay,cfg.boardSize)
    prep=[j for j in jobs if j.kind=="prepare_for_plant"]
    assert prep, "no prepare_for_plant candidates"

    continuation=plan_bundle(raw,cfg,[])
    rows=[]
    for j in prep:
        b=plan_bundle(raw,cfg,[j],{j.key})
        rows.append({
            "key":j.key,
            "crop":j.target.get("crop"),
            "tile":j.target.get("tile"),
            "job_central_delta":j.central_delta,
            "job_strict_delta":j.strict_delta,
            "action":plain(b.action),
            "scheduled":list(b.scheduled),
            "strict_terminal_cash":b.envelope.strict_cash,
            "central_terminal_cash":b.envelope.central_cash,
            "immediate_cash":b.immediate_cash,
        })
    best=max(rows,key=lambda r:(r["strict_terminal_cash"],r["central_terminal_cash"],r["immediate_cash"],r["key"]))
    chosen,chosen_rep,active,choose_continuation=choose(raw,cfg,{})
    return {
        "prepare_candidate_count":len(prep),
        "continuation":{
            "action":plain(continuation.action),
            "strict_terminal_cash":continuation.envelope.strict_cash,
            "central_terminal_cash":continuation.envelope.central_cash,
            "immediate_cash":continuation.immediate_cash,
        },
        "best_prepare_for_plant":best,
        "choose_result":{
            "action":plain(chosen.action),
            "scheduled":list(chosen.scheduled),
            "strict_terminal_cash":chosen.envelope.strict_cash,
            "central_terminal_cash":chosen.envelope.central_cash,
            "representative":None if chosen_rep is None else chosen_rep.spec(),
            "active_count":len(active),
        },
        "prepare_beats_continuation":{
            "strict":best["strict_terminal_cash"]>continuation.envelope.strict_cash,
            "central":best["central_terminal_cash"]>continuation.envelope.central_cash,
            "immediate":best["immediate_cash"]>continuation.immediate_cash,
        },
    }


def envelope_check():
    raw=initial_raw()
    raw["step"]=0
    raw["private"]["shed"]["WHEAT"]=2
    e=terminal_envelope(raw,Settings())
    assert e.central_cash>=e.strict_cash,(e.strict_cash,e.central_cash)
    return {"strict":e.strict_cash,"central":e.central_cash}


def runtime_smoke():
    reset_agent()
    env=make("kaggriculture",configuration={"seed":92809902},debug=False)
    env.reset(num_agents=2)
    turns=0
    trace=[]
    while not env.done and turns<48:
        s0=env._Environment__get_shared_state(0)
        obs0=s0["observation"]
        pre_cash=float(obs0["farms"][0]["money"])
        a0=agent(obs0,env.configuration)
        dbg=debug_state(0)
        hands=len(obs0["farms"][0].get("hands",[]) or [])
        assert isinstance(a0,dict)
        assert len(a0.get("hands",[]))==hands,(turns,hands,a0)
        env.step([plain(a0),{"farmer":["PASS"],"hands":[],"market":[]}])
        post_cash=float(env.state[0].observation.farms[0]["money"])
        if turns<12:
            trace.append({
                "turn":turns,
                "pre_cash":pre_cash,
                "action":plain(a0),
                "post_cash":post_cash,
                "debug":plain(dbg),
            })
        turns+=1
    assert turns==48,turns
    return {"turns":turns,"cash":float(env.state[0].observation.farms[0]["money"]),"first12":trace}


def main():
    result={
        "feed_purpose":feed_purpose_check(),
        "continuation_value":continuation_value_check(),
        "parallel_bundle":parallel_bundle_check(),
        "terminal_timing":timing_not_fixed_day_check(),
        "shortplan_bridge":existing_shortplan_bridge_check(),
        "initial_choice_probe":initial_choice_probe(),
        "envelope":envelope_check(),
        "runtime_smoke":runtime_smoke(),
    }
    print("CONTRACT "+json.dumps(result,ensure_ascii=False,separators=(",",":")))


if __name__=="__main__":
    main()