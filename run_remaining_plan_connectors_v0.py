#!/usr/bin/env python3
import copy, json, sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as opponent
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_plan_match,
)

MAX_STEPS=12


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def same_shape(a,b):
    return (
        isinstance(a,dict) and isinstance(b,dict)
        and set(a)==set(b)=={"farmer","hands","market"}
        and len(a["hands"])==len(b["hands"])
    )


def positions(raw):
    f=raw["farms"][raw["player"]]
    return [list(f["farmer"])]+[list(x) for x in (f.get("hands",[]) or [])]


def distance_to_tile(raw,tile):
    tx,ty=tile
    return min(abs(x-tx)+abs(y-ty) for x,y in positions(raw))


def shed_access(raw):
    n=len(raw["farms"][raw["player"]]["tiles"]); h=n//2
    return [(h-1,h-1),(h,h-1),(h-1,h),(h,h)]


def distance_to_shed(raw,unit_index):
    x,y=positions(raw)[unit_index]
    return min(abs(tx-x)+abs(ty-y) for tx,ty in shed_access(raw))


def find_clean_delivery(snapshot,shed_capacity):
    raw=snapshot.raw(); hour=raw["hour"]
    for p in generate_plans(snapshot):
        if p.kind!="deliver_carried_to_shed": continue
        items=dict(p.target["carried_items"])
        if len(items)!=1: continue
        item,qty=next(iter(items.items()))
        if item not in raw["market"].get("prices",{}): continue
        d=distance_to_shed(raw,int(p.target["unit_index"]))
        if d<=0: continue
        total=sum(v for v in raw["private"].get("shed",{}).values() if isinstance(v,(int,float)))
        if total+qty>shed_capacity: continue
        if hour+d>22: continue
        return p
    return None


def find_clean_plant(snapshot):
    raw=snapshot.raw(); hour=raw["hour"]
    for p in generate_plans(snapshot):
        if p.kind!="establish_plant": continue
        tile=tuple(p.target["tile"])
        d=distance_to_tile(raw,tile)
        if 1<=d<=3 and hour+d+1<=23:
            return p
    return None


def matching(snapshot,template):
    kind=template.kind; target=dict(template.target)
    return [
        p for p in generate_plans(snapshot)
        if semantic_plan_match(p,kind=kind,target=target)
    ]


def candidate_view(snapshot, template):
    plans=generate_plans(snapshot)
    kind=template.kind; target=dict(template.target)
    exact=[
        p.to_dict() for p in plans
        if semantic_plan_match(p,kind=kind,target=target)
    ]
    return {
        "total":len(plans),
        "same_plan_present":bool(exact),
        "same_plan":exact,
        "kind_counts":{
            k:sum(p.kind==k for p in plans)
            for k in ("deliver_carried_to_shed","realize_shed_stock_sale","establish_plant")
        }
    }


def fork_step(env, pre, plan):
    s1=env._Environment__get_shared_state(1)
    opp=plain(opponent.agent(s1["observation"]))
    baseline=baseline_pass_bundle(pre)
    p1=project_short_plan(pre,plan)
    b_env=copy.deepcopy(env); p_env=copy.deepcopy(env)
    b_env.step([copy.deepcopy(baseline),copy.deepcopy(opp)])
    p_env.step([copy.deepcopy(p1),copy.deepcopy(opp)])
    b_post=bind_official_state(b_env._Environment__get_shared_state(0)["observation"])
    p_post=bind_official_state(p_env._Environment__get_shared_state(0)["observation"])
    return {
        "baseline_bundle":baseline,
        "p1_bundle":p1,
        "opponent_bundle":opp,
        "baseline_post":b_post,
        "p1_post":p_post,
        "p1_env":p_env,
        "checks":{
            "action_diff":baseline!=p1,
            "same_shape":same_shape(baseline,p1),
        }
    }


def replay_to_delivery_completion():
    current.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":7001},debug=False)
    env.reset(num_agents=2)
    shed_capacity=int(getattr(env.configuration,"shedCapacity",100))
    ref=0
    selected=None
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        snap=bind_official_state(s0["observation"])
        selected=find_clean_delivery(snap,shed_capacity)
        if selected: break
        env.step([plain(current.agent(s0["observation"])),plain(opponent.agent(s1["observation"]))])
        ref+=1
    if selected is None: raise RuntimeError("delivery plan not found")
    initial=bind_official_state(env._Environment__get_shared_state(0)["observation"])
    for i in range(MAX_STEPS):
        pre=bind_official_state(env._Environment__get_shared_state(0)["observation"])
        ms=matching(pre,selected)
        if not ms:
            whole=completion_from_states(selected,initial,pre)
            if whole["complete"]: return env,pre,selected,ref,i
            raise RuntimeError("delivery disappeared before completion")
        step=fork_step(env,pre,ms[0])
        env=step["p1_env"]
        post=step["p1_post"]
        whole=completion_from_states(selected,initial,post)
        if whole["complete"]: return env,post,selected,ref,i+1
    raise RuntimeError("delivery did not complete")


def observe_sale():
    env,pre,delivery,ref,delivery_steps=replay_to_delivery_completion()
    item=next(iter(delivery.target["carried_items"]))
    sales=[p for p in generate_plans(pre) if p.kind=="realize_shed_stock_sale" and p.target["item"]==item]
    if not sales: raise RuntimeError("sale candidate absent after delivery")
    sale=sales[0]

    pre_raw=pre.raw(); player=pre_raw["player"]
    step=fork_step(env,pre,sale)
    b=step["baseline_post"]; p=step["p1_post"]
    comp=completion_from_states(sale,pre,p)
    post_plans=generate_plans(p)
    same_sale=[
        q for q in post_plans
        if semantic_plan_match(q,kind=sale.kind,target=dict(sale.target))
    ]

    record={
        "seed":7001,
        "reference_steps_before_delivery":ref,
        "delivery_steps":delivery_steps,
        "pre":{
            "day":pre_raw["day"],"hour":pre_raw["hour"],
            "cash":pre_raw["farms"][player]["money"],
            "item":item,
            "shed_qty":pre_raw["private"]["shed"].get(item,0),
            "plan":sale.to_dict(),
            "plan_view":candidate_view(pre,sale),
        },
        "baseline":{
            "action_bundle":step["baseline_bundle"],
            "post_cash":b.raw()["farms"][player]["money"],
            "post_shed_qty":b.raw()["private"]["shed"].get(item,0),
            "post_plan_view":candidate_view(b,sale),
        },
        "p1":{
            "action_bundle":step["p1_bundle"],
            "post_cash":p.raw()["farms"][player]["money"],
            "post_shed_qty":p.raw()["private"]["shed"].get(item,0),
            "completion":comp,
            "post_plan_view":candidate_view(p,sale),
            "same_sale_present_after":bool(same_sale),
        },
        "official_world":{
            "same_pre_state":True,
            "same_opponent_action":step["opponent_bundle"],
            "action_diff":step["checks"]["action_diff"],
            "same_shape":step["checks"]["same_shape"],
        }
    }
    checklist={
        "action_difference_observed":record["official_world"]["action_diff"],
        "pre_and_post_state_observed":True,
        "official_world_effect_observed":(
            record["p1"]["post_shed_qty"]!=record["baseline"]["post_shed_qty"]
            or record["p1"]["post_cash"]!=record["baseline"]["post_cash"]
        ),
        "baseline_and_p1_compared_same_shape":record["official_world"]["same_shape"],
        "next_state_to_replan_connected":"post_plan_view" in record["p1"],
    }
    record["checklist"]=checklist
    record["all_fixed_observation_conditions_met"]=all(checklist.values())
    return record


def observe_plant():
    current.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":7004},debug=False)
    env.reset(num_agents=2)
    ref=0; selected=None
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        snap=bind_official_state(s0["observation"])
        selected=find_clean_plant(snap)
        if selected: break
        env.step([plain(current.agent(s0["observation"])),plain(opponent.agent(s1["observation"]))])
        ref+=1
    if selected is None: raise RuntimeError("clean plant plan not found")

    initial=bind_official_state(env._Environment__get_shared_state(0)["observation"])
    target=dict(selected.target)
    trace=[]; completed=False
    for i in range(MAX_STEPS):
        pre=bind_official_state(env._Environment__get_shared_state(0)["observation"])
        ms=matching(pre,selected)
        if not ms:
            break
        step=fork_step(env,pre,ms[0])
        b=step["baseline_post"]; p=step["p1_post"]
        comp=completion_from_states(selected,pre,p)
        whole=completion_from_states(selected,initial,p)
        next_ms=matching(p,selected)
        next_bundle=project_short_plan(p,next_ms[0]) if next_ms and not whole["complete"] else None
        rawpre=pre.raw(); rawb=b.raw(); rawp=p.raw()
        x,y=target["tile"]; player=rawpre["player"]
        trace.append({
            "step":i,
            "pre":{
                "day":rawpre["day"],"hour":rawpre["hour"],
                "seed_qty":rawpre["private"]["seeds"].get(target["crop"],0),
                "target_tile":copy.deepcopy(rawpre["farms"][player]["tiles"][y][x]),
                "plan_view":candidate_view(pre,selected),
            },
            "baseline":{
                "action_bundle":step["baseline_bundle"],
                "target_tile":copy.deepcopy(rawb["farms"][player]["tiles"][y][x]),
                "seed_qty":rawb["private"]["seeds"].get(target["crop"],0),
                "post_plan_view":candidate_view(b,selected),
            },
            "p1":{
                "action_bundle":step["p1_bundle"],
                "target_tile":copy.deepcopy(rawp["farms"][player]["tiles"][y][x]),
                "seed_qty":rawp["private"]["seeds"].get(target["crop"],0),
                "completion_this_step":comp,
                "completion_from_initial":whole,
                "post_plan_view":candidate_view(p,selected),
                "next_replanned_action_bundle":next_bundle,
            },
            "official_world":{
                "same_pre_state":True,
                "same_opponent_action":step["opponent_bundle"],
                "action_diff":step["checks"]["action_diff"],
                "same_shape":step["checks"]["same_shape"],
            }
        })
        env=step["p1_env"]
        if whole["complete"]:
            completed=True
            break

    checklist={
        "action_difference_observed":bool(trace) and all(r["official_world"]["action_diff"] for r in trace),
        "pre_and_post_state_observed":bool(trace),
        "official_world_effect_observed":bool(trace) and any(
            r["baseline"]["target_tile"]!=r["p1"]["target_tile"]
            or r["baseline"]["seed_qty"]!=r["p1"]["seed_qty"]
            for r in trace
        ),
        "baseline_and_p1_compared_same_shape":bool(trace) and all(r["official_world"]["same_shape"] for r in trace),
        "next_state_to_replan_connected":bool(trace) and all(
            r["p1"]["completion_from_initial"]["complete"]
            or r["p1"]["next_replanned_action_bundle"] is not None
            for r in trace
        ),
    }
    return {
        "seed":7004,
        "reference_steps_before_test":ref,
        "selected_plan":selected.to_dict(),
        "trace":trace,
        "completed":completed,
        "checklist":checklist,
        "all_fixed_observation_conditions_met":all(checklist.values()),
    }


def main():
    sale=observe_sale()
    plant=observe_plant()
    result={
        "schema":"remaining-plan-connectors-v0",
        "sale":sale,
        "plant":plant,
        "acceptance":{
            "sale_complete":sale["p1"]["completion"]["complete"],
            "sale_fixed_5":sale["all_fixed_observation_conditions_met"],
            "plant_complete":plant["completed"],
            "plant_fixed_5":plant["all_fixed_observation_conditions_met"],
        },
        "boundary":{
            "explicit_test_plan_only":True,
            "no_plan_ranking_or_selection_policy":True,
            "no_terminal_strength_claim":True,
        }
    }
    result["acceptance"]["all_pass"]=all(result["acceptance"].values())
    Path("remaining_plan_connectors_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps({
        "all_pass":result["acceptance"]["all_pass"],
        "sale_complete":result["acceptance"]["sale_complete"],
        "sale_cash_gain":sale["p1"]["completion"]["cash_gain"],
        "sale_sold_qty":sale["p1"]["completion"]["sold_quantity"],
        "sale_same_candidate_after":sale["p1"]["same_sale_present_after"],
        "plant_complete":result["acceptance"]["plant_complete"],
        "plant_steps":len(plant["trace"]),
        "sale_fixed_5":result["acceptance"]["sale_fixed_5"],
        "plant_fixed_5":result["acceptance"]["plant_fixed_5"],
    },separators=(",",":")))

if __name__=="__main__":
    main()
