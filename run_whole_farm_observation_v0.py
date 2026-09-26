#!/usr/bin/env python3
"""Whole Farm Observation v0.

Confirmed Circulation Body v0 is NOT reimplemented here.
The frozen body is the existing:
    run_exclude_confirmed_blocked_ab_v0.run_policy("exclude_blocked")

Guard:
1) Run the frozen Body twice with identical fixed conditions.
2) Require exact equality of self ActionBundle sequence, opponent ActionBundle
   sequence, terminal cash, rewards, and turn count.
3) Replay the second run's exact action sequences through a fresh Official
   World while observing raw state.
4) Require every replay Pre/Post canonical hash to match the frozen Body trace.

Only after all guards pass do we summarize the farm's 30-day life.
No Candidate, status, selection, continuation, projection, completion, or
Official action logic is modified by this file.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

import run_exclude_confirmed_blocked_ab_v0 as body
from plan_generator_entrance_v0 import bind_official_state, generate_plans

BODY_SOURCE_COMMIT="696ac759b66e3fe0463dd9d0bc60fc33b1d714c2"
BODY_MODE="exclude_blocked"


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def nz_numeric(d):
    return {
        str(k):float(v) if isinstance(v,float) else int(v)
        for k,v in (d or {}).items()
        if isinstance(v,(int,float)) and v
    }


def tile_class(tile):
    if tile is None:
        return "EMPTY"
    if tile=="LOCKED":
        return "LOCKED"
    if isinstance(tile,dict):
        kind=str(tile.get("kind"))
        if kind=="PLANT":
            return "PLANT:"+str(tile.get("crop"))
        if kind=="WEED":
            return "WEED"
        return kind
    return str(tile)


def aggregate_inventory(private):
    out=Counter()
    for inv in private.get("inventories",[]) or []:
        if not isinstance(inv,dict):
            continue
        for k,v in inv.items():
            if isinstance(v,(int,float)) and v:
                out[str(k)]+=float(v)
    return out


def plant_rows(raw):
    p=raw["player"]
    out=[]
    for y,row in enumerate(raw["farms"][p].get("tiles",[]) or []):
        for x,tile in enumerate(row):
            if isinstance(tile,dict) and tile.get("kind")=="PLANT":
                out.append({
                    "tile":[x,y],
                    "crop":str(tile.get("crop")),
                    "planted_day":tile.get("planted_day"),
                    "watered_today":bool(tile.get("watered_today",False)),
                    "consecutive_unwatered":tile.get("consecutive_unwatered"),
                    "yield_units":float(tile.get("yield_units",0) or 0),
                    "max_lifespan_step":tile.get("max_lifespan_step"),
                })
    return out


def state_observation(snapshot):
    raw=snapshot.raw()
    p=raw["player"]
    farm=raw["farms"][p]
    private=raw["private"]

    surface=Counter()
    yield_by_crop=Counter()
    for row in farm.get("tiles",[]) or []:
        for tile in row:
            cls=tile_class(tile)
            if cls.startswith("PLANT:"):
                surface["PLANT"]+=1
                crop=cls.split(":",1)[1]
                if isinstance(tile,dict):
                    yield_by_crop[crop]+=float(tile.get("yield_units",0) or 0)
            else:
                surface[cls]+=1

    carried=aggregate_inventory(private)
    plans=generate_plans(snapshot)
    statuses=[body.first_action_status(snapshot,pn) for pn in plans]
    status_counts=Counter(s["status"] for s in statuses)

    return {
        "day":int(raw["day"]),
        "hour":int(raw["hour"]),
        "cash":float(farm.get("money",0) or 0),
        "farmer":plain(farm.get("farmer")),
        "hands":plain(farm.get("hands",[]) or []),
        "seeds":nz_numeric(private.get("seeds",{}) or {}),
        "shed":nz_numeric(private.get("shed",{}) or {}),
        "carried":dict(carried),
        "surface":dict(surface),
        "plants":plant_rows(raw),
        "yield_by_crop":dict(yield_by_crop),
        "candidate_count":len(plans),
        "status_counts":dict(status_counts),
        "candidate_kind_counts":dict(Counter(pn.kind for pn in plans)),
    }


def sum_product(mapping,product):
    return float((mapping or {}).get(product,0) or 0)


def surface_transition_counts(pre_raw,post_raw):
    p=pre_raw["player"]
    before=pre_raw["farms"][p].get("tiles",[]) or []
    after=post_raw["farms"][p].get("tiles",[]) or []
    c=Counter()
    for y in range(min(len(before),len(after))):
        for x in range(min(len(before[y]),len(after[y]))):
            a=tile_class(before[y][x])
            b=tile_class(after[y][x])
            if a!=b:
                c[f"{a}->{b}"]+=1
    return c


def yield_delta_by_crop(pre_raw,post_raw):
    p=pre_raw["player"]
    before=pre_raw["farms"][p].get("tiles",[]) or []
    after=post_raw["farms"][p].get("tiles",[]) or []
    c=Counter()
    for y in range(min(len(before),len(after))):
        for x in range(min(len(before[y]),len(after[y]))):
            a=before[y][x]
            b=after[y][x]
            if (
                isinstance(a,dict) and a.get("kind")=="PLANT"
                and isinstance(b,dict) and b.get("kind")=="PLANT"
                and a.get("crop")==b.get("crop")
            ):
                crop=str(a.get("crop"))
                da=float(b.get("yield_units",0) or 0)-float(a.get("yield_units",0) or 0)
                if da>0:
                    c[crop]+=da
    return c


def market_labels(bundle):
    labels=[]
    for op in bundle.get("market",[]) or []:
        labels.append(plain(op))
    return labels


def action_sequences(run):
    return [r["action_bundle"] for r in run["turns"]]


def opponent_sequences(run):
    return [r["opponent_action_bundle"] for r in run["turns"]]


def selected_total(summary):
    return sum(int(v) for v in summary.get("selected_plans_by_kind",{}).values())


def outcome_total(summary,key):
    return sum(int(v) for v in summary.get(key,{}).values())


def main():
    # Guard A/B: same frozen Body, no behavior changes.
    reference=body.run_policy(BODY_MODE)
    observed_source=body.run_policy(BODY_MODE)

    self_actions_equal=action_sequences(reference)==action_sequences(observed_source)
    opp_actions_equal=opponent_sequences(reference)==opponent_sequences(observed_source)
    terminal_equal=(
        reference["summary"]["terminal_self_cash"]==observed_source["summary"]["terminal_self_cash"]
        and reference["summary"]["official_rewards"]==observed_source["summary"]["official_rewards"]
        and reference["summary"]["turns"]==observed_source["summary"]["turns"]
    )

    if not (self_actions_equal and opp_actions_equal and terminal_equal):
        raise RuntimeError("Observer guard failed before observation replay")

    # External replay: observer never participates in Body decision making.
    env=make("kaggriculture",configuration={"seed":body.ENV_SEED},debug=False)
    env.reset(num_agents=2)

    life_trace=[]
    surface_transitions=Counter()
    positive_yield=Counter()
    planted=Counter()
    harvested=Counter()
    sold=Counter()
    cash_in_events=[]
    cash_out_events=[]
    market_events=[]
    all_hashes_match=True

    daily_first={}
    daily_last={}

    for i,row in enumerate(observed_source["turns"]):
        s0=env._Environment__get_shared_state(0)
        pre=bind_official_state(s0["observation"])
        pre_raw=pre.raw()

        if pre.canonical_hash!=row["pre_hash"]:
            all_hashes_match=False
            raise RuntimeError(f"Replay pre hash mismatch at turn {i}")

        pre_obs=state_observation(pre)
        daily_first.setdefault(pre_obs["day"],pre_obs)

        self_bundle=plain(row["action_bundle"])
        opp_bundle=plain(row["opponent_action_bundle"])

        env.step([self_bundle,opp_bundle])
        post=bind_official_state(env.state[0].observation)
        post_raw=post.raw()

        if post.canonical_hash!=row["post_hash"]:
            all_hashes_match=False
            raise RuntimeError(f"Replay post hash mismatch at turn {i}")

        post_obs=state_observation(post)
        daily_last[pre_obs["day"]]=post_obs

        # Surface transitions.
        st=surface_transition_counts(pre_raw,post_raw)
        surface_transitions.update(st)
        for k,v in st.items():
            if "->PLANT:" in k:
                crop=k.split("->PLANT:",1)[1]
                planted[crop]+=v

        # Yield generated on continuing plants.
        yd=yield_delta_by_crop(pre_raw,post_raw)
        positive_yield.update(yd)

        # Harvested product: actual increase in carried inventory on HARVEST turn.
        pre_car=aggregate_inventory(pre_raw["private"])
        post_car=aggregate_inventory(post_raw["private"])
        action_tokens=[]
        action_tokens.extend(self_bundle.get("farmer",[]) or [])
        for h in self_bundle.get("hands",[]) or []:
            if isinstance(h,list):
                action_tokens.extend(h)
            else:
                action_tokens.append(h)
        if "HARVEST" in action_tokens:
            for product in set(pre_car)|set(post_car):
                delta=post_car[product]-pre_car[product]
                if delta>0:
                    harvested[product]+=delta

        # Actual shed decrease on SELL turn.
        pre_shed=pre_raw["private"].get("shed",{}) or {}
        post_shed=post_raw["private"].get("shed",{}) or {}
        for op in self_bundle.get("market",[]) or []:
            if isinstance(op,list) and len(op)>=2 and op[0]=="SELL":
                product=str(op[1])
                actual=max(0.0,sum_product(pre_shed,product)-sum_product(post_shed,product))
                if actual:
                    sold[product]+=actual

        cash_delta=post_obs["cash"]-pre_obs["cash"]
        event={
            "turn":i,
            "day":pre_obs["day"],
            "hour":pre_obs["hour"],
            "cash_before":pre_obs["cash"],
            "cash_after":post_obs["cash"],
            "cash_delta":cash_delta,
            "market_actions":market_labels(self_bundle),
            "active_plan":plain(row.get("active_before_action")),
            "action_bundle":self_bundle,
        }
        if cash_delta>0:
            cash_in_events.append(event)
        elif cash_delta<0:
            cash_out_events.append(event)
        if self_bundle.get("market"):
            market_events.append(event)

        life_trace.append({
            "turn":i,
            "pre":pre_obs,
            "body":{
                "candidate_count":row["candidate_count"],
                "blocked_count":row["blocked_count"],
                "selectable_count":row["selectable_count"],
                "active_plan":plain(row.get("active_before_action")),
                "current_status":plain(row.get("current_status")),
                "action_mode":row.get("action_mode"),
                "action_bundle":self_bundle,
                "completion":plain(row.get("completion")),
                "transition_reason":row.get("transition_reason"),
            },
            "post":post_obs,
        })

    if not env.done:
        raise RuntimeError("Replay did not reach terminal")

    final=bind_official_state(env.state[0].observation)
    final_obs=state_observation(final)
    replay_rewards=[]
    for st in env.state:
        try: replay_rewards.append(float(st.reward))
        except Exception: replay_rewards.append(None)

    replay_terminal_equal=(
        final_obs["cash"]==observed_source["summary"]["terminal_self_cash"]
        and replay_rewards==observed_source["summary"]["official_rewards"]
    )
    if not replay_terminal_equal:
        raise RuntimeError("Replay terminal differs from frozen Body")

    summary=observed_source["summary"]
    selected=selected_total(summary)
    completed=outcome_total(summary,"completed_plans_by_kind")
    invalidated=outcome_total(summary,"invalidated_plans_by_kind")
    timed_out=outcome_total(summary,"timed_out_plans_by_kind")
    blocked_drops=outcome_total(summary,"blocked_active_drops_by_kind")
    terminal_residual=max(0,selected-completed-invalidated-timed_out-blocked_drops)

    cash_series=[r["pre"]["cash"] for r in life_trace]+[final_obs["cash"]]
    pass_turns=sum(1 for r in life_trace if str(r["body"]["action_mode"]).startswith("pass_"))
    all_blocked_pass=sum(
        1 for r in life_trace
        if r["body"]["candidate_count"]>0
        and r["body"]["selectable_count"]==0
        and str(r["body"]["action_mode"]).startswith("pass_")
    )
    no_candidate_pass=sum(
        1 for r in life_trace
        if r["body"]["candidate_count"]==0
        and str(r["body"]["action_mode"]).startswith("pass_")
    )
    other_pass=pass_turns-all_blocked_pass-no_candidate_pass

    day_rows=[]
    for day in sorted(daily_first):
        start=daily_first[day]
        end=daily_last.get(day,start)
        day_rows.append({
            "day":day,
            "start_cash":start["cash"],
            "end_cash":end["cash"],
            "start_surface":start["surface"],
            "end_surface":end["surface"],
            "end_plants":end["plants"],
            "end_carried":end["carried"],
            "end_shed":end["shed"],
            "end_candidate_count":end["candidate_count"],
            "end_status_counts":end["status_counts"],
        })

    result={
        "schema":"whole-farm-observation-v0",
        "body_snapshot":{
            "name":"Confirmed Circulation Body v0",
            "source_commit":BODY_SOURCE_COMMIT,
            "mode":BODY_MODE,
            "environment_seed":body.ENV_SEED,
            "policy_seed":body.POLICY_SEED,
            "opponent":"Seyamalam pinned v21",
            "body_reimplemented":False,
        },
        "observer_guard":{
            "reference_vs_second_body_self_action_sequence_equal":self_actions_equal,
            "reference_vs_second_body_opponent_action_sequence_equal":opp_actions_equal,
            "reference_vs_second_body_terminal_equal":terminal_equal,
            "replay_all_pre_post_hashes_equal":all_hashes_match,
            "replay_terminal_equal":replay_terminal_equal,
            "passed":(
                self_actions_equal and opp_actions_equal and terminal_equal
                and all_hashes_match and replay_terminal_equal
            ),
        },
        "whole_life":{
            "turns":summary["turns"],
            "terminal_self":summary["terminal_self_cash"],
            "official_rewards":summary["official_rewards"],
            "cash":{
                "start":cash_series[0],
                "terminal":final_obs["cash"],
                "minimum":min(cash_series),
                "maximum":max(cash_series),
                "positive_cash_transitions":len(cash_in_events),
                "negative_cash_transitions":len(cash_out_events),
                "total_positive_cash_delta":sum(e["cash_delta"] for e in cash_in_events),
                "total_negative_cash_delta":sum(e["cash_delta"] for e in cash_out_events),
                "cash_return_events":cash_in_events,
                "cash_outflow_events":cash_out_events,
                "market_events":market_events,
            },
            "production":{
                "planted_units_by_crop":dict(planted),
                "positive_yield_units_generated_on_continuing_plants_by_crop":dict(positive_yield),
                "harvested_units_to_carried_by_product":dict(harvested),
                "sold_units_from_shed_by_product":dict(sold),
            },
            "surface":{
                "transition_counts":dict(surface_transitions),
            },
            "time_use":{
                "active_plan_turns":sum(1 for r in life_trace if r["body"]["active_plan"] is not None),
                "pass_turns":pass_turns,
                "all_blocked_pass_turns":all_blocked_pass,
                "no_candidate_pass_turns":no_candidate_pass,
                "other_pass_turns":other_pass,
            },
            "work_lifecycle":{
                "selected_total":selected,
                "selected_by_kind":summary["selected_plans_by_kind"],
                "completed_total":completed,
                "completed_by_kind":summary["completed_plans_by_kind"],
                "invalidated_total":invalidated,
                "invalidated_by_kind":summary["invalidated_plans_by_kind"],
                "timed_out_total":timed_out,
                "timed_out_by_kind":summary["timed_out_plans_by_kind"],
                "blocked_active_drop_total":blocked_drops,
                "blocked_active_drops_by_kind":summary["blocked_active_drops_by_kind"],
                "terminal_residual_total":terminal_residual,
                "terminal_active_plan":plain(life_trace[-1]["body"]["active_plan"]),
            },
            "terminal_snapshot":final_obs,
            "daily":day_rows,
        },
        "life_trace":life_trace,
        "boundary":{
            "observer_changes_body_behavior":False,
            "new_plan_kind_added":False,
            "new_status_rule_added":False,
            "terminal_aware_logic_added":False,
            "priority_score_utility_added":False,
        },
    }

    Path("whole_farm_observation_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

    print("GUARD "+json.dumps(result["observer_guard"],separators=(",",":")))
    print("SUMMARY "+json.dumps({
        "turns":result["whole_life"]["turns"],
        "terminal_self":result["whole_life"]["terminal_self"],
        "cash":{
            "start":result["whole_life"]["cash"]["start"],
            "min":result["whole_life"]["cash"]["minimum"],
            "max":result["whole_life"]["cash"]["maximum"],
            "terminal":result["whole_life"]["cash"]["terminal"],
            "cash_return_events":result["whole_life"]["cash"]["positive_cash_transitions"],
            "total_cash_return":result["whole_life"]["cash"]["total_positive_cash_delta"],
            "total_cash_outflow":result["whole_life"]["cash"]["total_negative_cash_delta"],
        },
        "production":result["whole_life"]["production"],
        "time_use":result["whole_life"]["time_use"],
        "work_lifecycle":result["whole_life"]["work_lifecycle"],
        "terminal_snapshot":result["whole_life"]["terminal_snapshot"],
        "surface_transition_counts":result["whole_life"]["surface"]["transition_counts"],
    },separators=(",",":")))


if __name__=="__main__":
    main()
