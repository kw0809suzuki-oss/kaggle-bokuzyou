#!/usr/bin/env python3
"""One Lost Plant Trace v0.

Observe exactly one lost plant from the frozen Confirmed Circulation Body v0.
No Body behavior is changed.

Selection rule for the observed plant:
1) Find the earliest Official transition containing PLANT -> WEED.
2) If multiple plants become WEED on that same transition, choose the one
   whose PLANT establishment was observed earliest in the same replay.
3) Trace that plant from establishment through WEED conversion.

For each turn in its life, record:
- raw target tile state
- whether the exact maintain_plant_today candidate exists
- whether a fresh Selection happened that turn
- active/current plan
- actual ActionBundle
- Official post tile
- next-turn active/current visibility

Finally classify only into the user-defined boundary:
1 = maintenance candidate existed + Selection opportunity existed + another job selected
2 = maintenance candidate existed + no Selection opportunity because another active Plan continued
3 = maintenance candidate did not exist when maintenance was needed
"""

from __future__ import annotations
import copy, json
from pathlib import Path
from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
from plan_generator_entrance_v0 import bind_official_state, generate_plans

BODY_MODE="exclude_blocked"


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def tile_at(snapshot,xy):
    raw=snapshot.raw(); p=raw["player"]; x,y=xy
    return copy.deepcopy(raw["farms"][p]["tiles"][y][x])


def semantic_maintain_match(plan,crop,tile,planted_day):
    if plan.kind!="maintain_plant_today":
        return False
    t=plan.target
    return (
        str(t.get("crop"))==crop
        and list(t.get("tile",[]))==list(tile)
        and int(t.get("planted_day",-999))==int(planted_day)
    )


def active_matches_maintain(active,crop,tile,planted_day):
    if not active or active.get("kind")!="maintain_plant_today":
        return False
    t=active.get("target",{})
    return (
        str(t.get("crop"))==crop
        and list(t.get("tile",[]))==list(tile)
        and int(t.get("planted_day",-999))==int(planted_day)
    )


def main():
    frozen=body.run_policy(BODY_MODE)

    env=make("kaggriculture",configuration={"seed":body.ENV_SEED},debug=False)
    env.reset(num_agents=2)

    establishment_turn={}
    establishment_meta={}
    raw_turns=[]
    first_weed_turn=None
    weeded_keys=[]

    for i,row in enumerate(frozen["turns"]):
        pre=bind_official_state(env._Environment__get_shared_state(0)["observation"])
        pre_raw=pre.raw(); p=pre_raw["player"]

        if pre.canonical_hash!=row["pre_hash"]:
            raise RuntimeError(f"pre hash mismatch at {i}")

        plans=generate_plans(pre)
        plan_rows=[{
            "kind":q.kind,
            "target":plain(q.target),
        } for q in plans]

        self_bundle=plain(row["action_bundle"])
        opp_bundle=plain(row["opponent_action_bundle"])

        env.step([self_bundle,opp_bundle])
        post=bind_official_state(env.state[0].observation)
        if post.canonical_hash!=row["post_hash"]:
            raise RuntimeError(f"post hash mismatch at {i}")

        post_raw=post.raw()

        # Observe PLANT establishments and PLANT->WEED conversions.
        pre_tiles=pre_raw["farms"][p]["tiles"]
        post_tiles=post_raw["farms"][p]["tiles"]
        this_weed=[]
        for y in range(len(pre_tiles)):
            for x in range(len(pre_tiles[y])):
                a=pre_tiles[y][x]
                b=post_tiles[y][x]
                if not (isinstance(a,dict) and a.get("kind")=="PLANT") and (
                    isinstance(b,dict) and b.get("kind")=="PLANT"
                ):
                    key=(str(b.get("crop")),x,y,int(b.get("planted_day",post_raw["day"])))
                    establishment_turn.setdefault(key,i)
                    establishment_meta.setdefault(key,{
                        "turn":i,
                        "day":int(pre_raw["day"]),
                        "hour":int(pre_raw["hour"]),
                        "post_tile":copy.deepcopy(b),
                    })
                if (
                    isinstance(a,dict) and a.get("kind")=="PLANT"
                    and isinstance(b,dict) and b.get("kind")=="WEED"
                ):
                    key=(str(a.get("crop")),x,y,int(a.get("planted_day",pre_raw["day"])))
                    this_weed.append(key)

        raw_turns.append({
            "turn":i,
            "pre_day":int(pre_raw["day"]),
            "pre_hour":int(pre_raw["hour"]),
            "pre_snapshot":pre,
            "post_snapshot":post,
            "plans":plan_rows,
            "body_row":row,
        })

        if this_weed and first_weed_turn is None:
            first_weed_turn=i
            weeded_keys=this_weed
            break

    if first_weed_turn is None:
        raise RuntimeError("No PLANT->WEED transition observed")

    # Deterministic representative: earliest established among first weed event.
    def est_key(k):
        return (establishment_turn.get(k,10**9), k[1], k[2], k[0])

    target=min(weeded_keys,key=est_key)
    crop,x,y,planted_day=target
    tile=[x,y]
    est_turn=establishment_turn.get(target)
    if est_turn is None:
        raise RuntimeError(f"Target establishment turn not observed for {target}")

    life=[]
    classification_evidence=[]

    for rec in raw_turns:
        i=rec["turn"]
        if i<est_turn+1 or i>first_weed_turn:
            continue

        pre=rec["pre_snapshot"]
        post=rec["post_snapshot"]
        pre_tile=tile_at(pre,tile)
        post_tile=tile_at(post,tile)
        row=rec["body_row"]

        exact_maint=[
            pr for pr in rec["plans"]
            if pr["kind"]=="maintain_plant_today"
            and str(pr["target"].get("crop"))==crop
            and list(pr["target"].get("tile",[]))==tile
            and int(pr["target"].get("planted_day",-999))==planted_day
        ]
        maintain_present=bool(exact_maint)

        maintenance_needed=(
            isinstance(pre_tile,dict)
            and pre_tile.get("kind")=="PLANT"
            and not bool(pre_tile.get("watered_today",False))
            and int(pre_tile.get("consecutive_unwatered",0) or 0)>=1
        )

        selection_opportunity=row.get("selection_reason") is not None
        selected_active=plain(row.get("active_before_action"))
        selected_target_maint=active_matches_maintain(
            selected_active,crop,tile,planted_day
        )

        next_row=(
            frozen["turns"][i+1]
            if i+1<len(frozen["turns"])
            else None
        )

        life_row={
            "turn":i,
            "day":int(pre.raw()["day"]),
            "hour":int(pre.raw()["hour"]),
            "pre_tile":pre_tile,
            "maintenance_needed_by_current_generator_condition":maintenance_needed,
            "exact_maintenance_candidate_present":maintain_present,
            "exact_maintenance_candidate":exact_maint[0] if exact_maint else None,
            "selection_opportunity_this_turn":selection_opportunity,
            "selected_active_plan":selected_active,
            "selected_target_maintenance":selected_target_maint,
            "action_bundle":plain(row["action_bundle"]),
            "post_tile":post_tile,
            "post_became_weed":(
                isinstance(post_tile,dict) and post_tile.get("kind")=="WEED"
            ),
            "next_state_current":{
                "active_before_action":plain(next_row.get("active_before_action")) if next_row else None,
                "selection_reason":next_row.get("selection_reason") if next_row else None,
                "action_bundle":plain(next_row.get("action_bundle")) if next_row else None,
            },
        }
        life.append(life_row)

        if maintenance_needed:
            classification_evidence.append(life_row)

    # Apply only the predeclared three-way split.
    case1=[
        r for r in classification_evidence
        if r["exact_maintenance_candidate_present"]
        and r["selection_opportunity_this_turn"]
        and not r["selected_target_maintenance"]
    ]
    case2=[
        r for r in classification_evidence
        if r["exact_maintenance_candidate_present"]
        and not r["selection_opportunity_this_turn"]
        and not r["selected_target_maintenance"]
        and r["selected_active_plan"] is not None
    ]
    case3=[
        r for r in classification_evidence
        if not r["exact_maintenance_candidate_present"]
    ]

    if case1:
        classification={
            "case":1,
            "label":"maintenance candidate existed; Selection opportunity existed; another Candidate was selected",
            "first_supporting_turn":case1[0]["turn"],
        }
    elif case2:
        classification={
            "case":2,
            "label":"maintenance candidate existed; no fresh Selection opportunity because another active Plan continued",
            "first_supporting_turn":case2[0]["turn"],
        }
    elif case3:
        classification={
            "case":3,
            "label":"maintenance candidate did not exist when maintenance was needed",
            "first_supporting_turn":case3[0]["turn"],
        }
    else:
        classification={
            "case":None,
            "label":"not classified by the predeclared three-way split",
            "first_supporting_turn":None,
        }

    result={
        "schema":"one-lost-plant-trace-v0",
        "body_snapshot":{
            "source_commit":"696ac759b66e3fe0463dd9d0bc60fc33b1d714c2",
            "mode":BODY_MODE,
            "environment_seed":body.ENV_SEED,
            "policy_seed":body.POLICY_SEED,
        },
        "representative_selection":{
            "rule":"earliest-established plant among the first Official PLANT->WEED transition",
            "first_weed_transition_turn":first_weed_turn,
            "all_plants_weeded_on_first_transition":[
                {
                    "crop":k[0],
                    "tile":[k[1],k[2]],
                    "planted_day":k[3],
                    "establishment_turn":establishment_turn.get(k),
                } for k in sorted(weeded_keys,key=est_key)
            ],
            "chosen":{
                "crop":crop,
                "tile":tile,
                "planted_day":planted_day,
                "establishment_turn":est_turn,
                "establishment":establishment_meta[target],
            },
        },
        "trace":life,
        "classification":classification,
        "classification_support":{
            "maintenance_needed_turns":len(classification_evidence),
            "case1_support_turns":[r["turn"] for r in case1],
            "case2_support_turns":[r["turn"] for r in case2],
            "case3_support_turns":[r["turn"] for r in case3],
        },
        "boundary":{
            "body_changed":False,
            "new_policy_added":False,
            "selection_ab_run":False,
            "classification_limited_to_predeclared_three_cases":True,
        },
    }

    Path("one_lost_plant_trace_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("SUMMARY "+json.dumps({
        "first_weed_turn":first_weed_turn,
        "weeded_on_first_transition":result["representative_selection"]["all_plants_weeded_on_first_transition"],
        "chosen":result["representative_selection"]["chosen"],
        "classification":classification,
        "maintenance_needed_turns":len(classification_evidence),
        "case1_support_turns":[r["turn"] for r in case1],
        "case2_support_turns":[r["turn"] for r in case2],
        "case3_support_turns":[r["turn"] for r in case3],
        "trace":[
            {
                "turn":r["turn"],
                "day_hour":[r["day"],r["hour"]],
                "pre_tile":r["pre_tile"],
                "maint_needed":r["maintenance_needed_by_current_generator_condition"],
                "maint_candidate":r["exact_maintenance_candidate_present"],
                "selection_opportunity":r["selection_opportunity_this_turn"],
                "active":r["selected_active_plan"],
                "action":r["action_bundle"],
                "post_tile":r["post_tile"],
                "weed":r["post_became_weed"],
            } for r in life
        ],
    },separators=(",",":")))


if __name__=="__main__":
    main()
