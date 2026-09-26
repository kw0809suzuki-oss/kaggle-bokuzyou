#!/usr/bin/env python3
"""Selection Commitment Occupancy v0.

Question:
Did Candidate-uniform vs Kind-uniform actually allocate the finite 30-day
turn budget differently through Plan commitments?

This observer does not add or change any policy.
It replays the same fixed baseline and Kind-uniform runs and measures only:

    Selected Plan kind
      -> turns occupied by that selected Plan
      -> next fresh Selection (or terminal boundary)

Important boundary:
"turns until next fresh Selection" is not automatically attributed to the
selected Plan. We separately record:
- occupied_turns: turns on which the same selected Plan is active
- selection_gap_turns: wall-clock turns until the next fresh Selection
- nonactive_gap_turns: gap turns not occupied by that selected Plan

No downstream WEED / Harvest / Sell / Cash explanation is attempted here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import run_exclude_confirmed_blocked_ab_v0 as body
import run_selection_kind_uniform_ab_v0 as ab


def plain(x):
    return body.plain(x)


def selection_events(run):
    turns=run["turns"]
    selected_turns=[
        i for i,row in enumerate(turns)
        if row.get("selection_reason") is not None
    ]
    events=[]
    anomalies=[]

    for pos,i in enumerate(selected_turns):
        row=turns[i]
        active=row.get("active_before_action")
        if not active:
            anomalies.append({
                "turn":i,
                "reason":"fresh_selection_without_active_before_action",
                "selection_reason":row.get("selection_reason"),
            })
            continue

        sequence=active.get("sequence")
        kind=str(active.get("kind"))
        next_selection_turn=(
            selected_turns[pos+1]
            if pos+1<len(selected_turns)
            else None
        )
        boundary=next_selection_turn if next_selection_turn is not None else len(turns)

        active_indices=[]
        seen_inactive=False
        reappeared=False
        for k in range(i,boundary):
            a=turns[k].get("active_before_action")
            same=bool(a and a.get("sequence")==sequence)
            if same:
                active_indices.append(k)
                if seen_inactive:
                    reappeared=True
            else:
                seen_inactive=True

        if reappeared:
            anomalies.append({
                "turn":i,
                "sequence":sequence,
                "reason":"same_plan_sequence_reappeared_after_becoming_inactive",
            })

        occupied_turns=len(active_indices)
        selection_gap_turns=boundary-i
        nonactive_gap_turns=selection_gap_turns-occupied_turns

        events.append({
            "selection_turn":i,
            "selection_day_hour":[
                int(row["pre_state"]["day"]),
                int(row["pre_state"]["hour"]),
            ],
            "sequence":sequence,
            "kind":kind,
            "target":plain(active.get("target")),
            "selection_reason":row.get("selection_reason"),
            "occupied_turns":occupied_turns,
            "first_active_turn":active_indices[0] if active_indices else None,
            "last_active_turn":active_indices[-1] if active_indices else None,
            "next_fresh_selection_turn":next_selection_turn,
            "selection_gap_turns":selection_gap_turns,
            "nonactive_gap_turns":nonactive_gap_turns,
            "ends_at_terminal_boundary":next_selection_turn is None,
            "terminal_turn_boundary":len(turns),
        })

    direct_active_turns=sum(
        1 for row in turns if row.get("active_before_action") is not None
    )
    allocated_active_turns=sum(e["occupied_turns"] for e in events)

    guards={
        "every_selection_has_event":len(events)==len(selected_turns),
        "sum_event_occupied_turns_equals_direct_active_turns":(
            allocated_active_turns==direct_active_turns
        ),
        "no_sequence_reappearance":not any(
            a["reason"]=="same_plan_sequence_reappeared_after_becoming_inactive"
            for a in anomalies
        ),
    }
    guards["passed"]=all(guards.values())

    return events,anomalies,guards


def mean(values):
    return (sum(values)/len(values)) if values else None


def median(values):
    if not values:
        return None
    vals=sorted(values)
    n=len(vals)
    m=n//2
    if n%2:
        return vals[m]
    return (vals[m-1]+vals[m])/2


def summarize(run,events):
    total_turns=len(run["turns"])
    by_kind=defaultdict(list)
    for e in events:
        by_kind[e["kind"]].append(e)

    kind_rows={}
    for kind,rows in sorted(by_kind.items()):
        occ=[r["occupied_turns"] for r in rows]
        gaps=[r["selection_gap_turns"] for r in rows]
        nonactive=[r["nonactive_gap_turns"] for r in rows]
        total_occ=sum(occ)
        kind_rows[kind]={
            "selection_count":len(rows),
            "occupied_turns_total":total_occ,
            "occupied_turn_share_of_world":total_occ/total_turns if total_turns else None,
            "mean_occupied_turns_per_selection":mean(occ),
            "median_occupied_turns_per_selection":median(occ),
            "min_occupied_turns":min(occ),
            "max_occupied_turns":max(occ),
            "mean_selection_gap_turns":mean(gaps),
            "nonactive_gap_turns_total":sum(nonactive),
            "occupancy_distribution":{
                str(k):v for k,v in sorted(Counter(occ).items())
            },
        }

    all_occ=[e["occupied_turns"] for e in events]
    all_gaps=[e["selection_gap_turns"] for e in events]

    return {
        "terminal_self":run["summary"]["terminal_self_cash"],
        "turns":total_turns,
        "selection_count":len(events),
        "active_turns_total":sum(all_occ),
        "active_turn_share_of_world":(
            sum(all_occ)/total_turns if total_turns else None
        ),
        "mean_occupied_turns_per_selection":mean(all_occ),
        "median_occupied_turns_per_selection":median(all_occ),
        "mean_selection_gap_turns":mean(all_gaps),
        "nonactive_gap_turns_total":sum(
            e["nonactive_gap_turns"] for e in events
        ),
        "by_kind":kind_rows,
    }


def compare_kind_rows(b,p):
    kinds=sorted(set(b["by_kind"])|set(p["by_kind"]))
    out={}
    for kind in kinds:
        br=b["by_kind"].get(kind)
        pr=p["by_kind"].get(kind)
        out[kind]={
            "baseline":{
                "selection_count":br["selection_count"] if br else 0,
                "occupied_turns_total":br["occupied_turns_total"] if br else 0,
                "occupied_turn_share_of_world":(
                    br["occupied_turn_share_of_world"] if br else 0.0
                ),
                "mean_occupied_turns_per_selection":(
                    br["mean_occupied_turns_per_selection"] if br else None
                ),
            },
            "kind_uniform":{
                "selection_count":pr["selection_count"] if pr else 0,
                "occupied_turns_total":pr["occupied_turns_total"] if pr else 0,
                "occupied_turn_share_of_world":(
                    pr["occupied_turn_share_of_world"] if pr else 0.0
                ),
                "mean_occupied_turns_per_selection":(
                    pr["mean_occupied_turns_per_selection"] if pr else None
                ),
            },
            "delta":{
                "selection_count":(
                    (pr["selection_count"] if pr else 0)
                    -(br["selection_count"] if br else 0)
                ),
                "occupied_turns_total":(
                    (pr["occupied_turns_total"] if pr else 0)
                    -(br["occupied_turns_total"] if br else 0)
                ),
                "occupied_turn_share_of_world":(
                    (pr["occupied_turn_share_of_world"] if pr else 0.0)
                    -(br["occupied_turn_share_of_world"] if br else 0.0)
                ),
            },
        }
    return out


def main():
    # Same fixed policies and seeds as Selection Kind-Uniform A/B v0.
    baseline=body.run_policy("exclude_blocked")
    kind_uniform=ab.run_kind_uniform()

    b_events,b_anom,b_guard=selection_events(baseline)
    p_events,p_anom,p_guard=selection_events(kind_uniform)

    if not b_guard["passed"]:
        raise RuntimeError("baseline occupancy guard failed: "+json.dumps(b_guard))
    if not p_guard["passed"]:
        raise RuntimeError("kind-uniform occupancy guard failed: "+json.dumps(p_guard))

    b_summary=summarize(baseline,b_events)
    p_summary=summarize(kind_uniform,p_events)

    result={
        "schema":"selection-commitment-occupancy-v0",
        "question":"Did the two Selection distributions actually allocate the finite turn budget differently through Plan commitments?",
        "source":{
            "baseline":"Confirmed Circulation Body v0 / exclude_blocked",
            "p1":"Selection Kind-Uniform A/B v0 / kind_uniform",
            "environment_seed":ab.ENV_SEED,
            "policy_seed":ab.POLICY_SEED,
        },
        "measurement_definition":{
            "fresh_selection":"turn where selection_reason is not null",
            "occupied_turns":"number of turns before the next fresh Selection on which the same selected plan sequence is active_before_action",
            "selection_gap_turns":"wall-clock turn distance from this fresh Selection to the next fresh Selection, or terminal boundary",
            "nonactive_gap_turns":"selection_gap_turns minus occupied_turns",
        },
        "baseline":{
            "summary":b_summary,
            "events":b_events,
            "anomalies":b_anom,
            "guard":b_guard,
        },
        "kind_uniform":{
            "summary":p_summary,
            "events":p_events,
            "anomalies":p_anom,
            "guard":p_guard,
        },
        "comparison":{
            "whole_run":{
                "selection_count_delta":(
                    p_summary["selection_count"]-b_summary["selection_count"]
                ),
                "active_turns_delta":(
                    p_summary["active_turns_total"]-b_summary["active_turns_total"]
                ),
                "active_turn_share_delta":(
                    p_summary["active_turn_share_of_world"]
                    -b_summary["active_turn_share_of_world"]
                ),
                "mean_occupied_turns_per_selection_delta":(
                    p_summary["mean_occupied_turns_per_selection"]
                    -b_summary["mean_occupied_turns_per_selection"]
                ),
                "nonactive_gap_turns_delta":(
                    p_summary["nonactive_gap_turns_total"]
                    -b_summary["nonactive_gap_turns_total"]
                ),
            },
            "by_kind":compare_kind_rows(b_summary,p_summary),
        },
        "boundary":{
            "policy_changed":False,
            "generator_changed":False,
            "continuation_changed":False,
            "projector_changed":False,
            "downstream_causal_claim_made":False,
            "weed_harvest_sell_cash_not_explained_here":True,
            "single_fixed_world_pair_only":True,
        },
    }

    Path("selection_commitment_occupancy_v0.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8",
    )

    print("SUMMARY "+json.dumps({
        "baseline":b_summary,
        "kind_uniform":p_summary,
        "comparison":result["comparison"],
        "guards":{
            "baseline":b_guard,
            "kind_uniform":p_guard,
        },
    },separators=(",",":")))


if __name__=="__main__":
    main()
