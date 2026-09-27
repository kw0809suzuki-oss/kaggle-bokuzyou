#!/usr/bin/env python3
"""Observe the exact Official World transition that first splits opponent cash.

For each Fresh20 paired world, find the first turn whose opponent pre-state cash
already differs. The generating transition is the immediately previous turn.
Require the opponent ActionBundle on that generating transition to be identical,
then compare Baseline vs Lens Official World pre->post transitions.

Observation only. Frozen Body/Lens/Harness are unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kaggle_environments import make

import relationship_surface_body_v0 as promoted
import run_exclude_confirmed_blocked_ab_v0 as legacy
from one_shot_view_lens_harness_v0 import OneShotViewLensHarness
from plan_generator_entrance_v0 import bind_official_state
from run_one_shot_view_lens_fresh20_battle_replay_v0 import LENSES, SEEDS

def _plain(x):
    return promoted._plain(x)

def _flatten(v, path=()):
    out={}
    if isinstance(v, dict):
        for k in sorted(v, key=str):
            out.update(_flatten(v[k], path+(str(k),)))
    elif isinstance(v, list):
        for i,x in enumerate(v):
            out.update(_flatten(x, path+(str(i),)))
    else:
        out[".".join(path)] = v
    return out

def _transition(pre, post):
    a=_flatten(pre); z=_flatten(post)
    keys=set(a)|set(z)
    return {k:{"before":a.get(k),"after":z.get(k)}
            for k in sorted(keys) if a.get(k)!=z.get(k)}

def _scope(raw):
    # Opponent's observation: private belongs to that opponent; farms/market/town
    # are shared Official World surfaces.
    return {
        "farms": raw.get("farms"),
        "private": raw.get("private"),
        "market": raw.get("market"),
        "town": raw.get("town"),
    }

def _run(seed, lens=None):
    opp=legacy.load_opponent(f"cash_transition_{lens or 'baseline'}_{seed}")
    env=make("kaggriculture", configuration={"seed":seed}, debug=False)
    env.reset(num_agents=2)
    agent=(promoted.RelationshipSurfaceBody() if lens is None
           else OneShotViewLensHarness(lens))
    harness=None if lens is None else agent
    rows=[]
    while not env.done:
        s0=env._Environment__get_shared_state(0)
        s1=env._Environment__get_shared_state(1)
        pre0=bind_official_state(s0["observation"])
        pre1=bind_official_state(s1["observation"])
        a0=agent.act(s0["observation"])
        a1=_plain(opp.agent(s1["observation"]))
        env.step([_plain(a0),a1])
        post0=bind_official_state(env.state[0].observation)
        post1=bind_official_state(env.state[1].observation)
        if harness is not None:
            harness.observe_post(env.state[0].observation)
        rpre1=pre1.raw(); rpost1=post1.raw()
        p=int(rpre1["player"])
        rows.append({
            "turn":len(rows),
            "self_action":_plain(a0),
            "opponent_action":a1,
            "opponent_cash_pre":float(rpre1["farms"][p].get("money",0) or 0),
            "opponent_cash_post":float(rpost1["farms"][p].get("money",0) or 0),
            "opponent_pre":_scope(rpre1),
            "opponent_post":_scope(rpost1),
            "self_pre_hash":pre0.canonical_hash,
            "self_post_hash":post0.canonical_hash,
        })
    return rows

def _paired(seed,lens):
    b=_run(seed,None); v=_run(seed,lens)
    if len(b)!=len(v): raise RuntimeError("turn-count mismatch")

    cash_turn=None
    for i,(x,y) in enumerate(zip(b,v)):
        if x["opponent_cash_pre"] != y["opponent_cash_pre"]:
            cash_turn=i; break
    if cash_turn is None or cash_turn==0:
        raise RuntimeError(f"seed {seed}: no valid first cash divergence")
    g=cash_turn-1
    bx,vy=b[g],v[g]
    if bx["opponent_action"] != vy["opponent_action"]:
        raise RuntimeError(
            f"seed {seed}: opponent action differs on generating transition {g}"
        )
    if bx["opponent_cash_pre"] != vy["opponent_cash_pre"]:
        raise RuntimeError(f"seed {seed}: cash already differed before generating transition")
    if bx["opponent_cash_post"] == vy["opponent_cash_post"]:
        raise RuntimeError(f"seed {seed}: generating transition did not create cash split")

    bt=_transition(bx["opponent_pre"],bx["opponent_post"])
    vt=_transition(vy["opponent_pre"],vy["opponent_post"])
    keys=sorted(set(bt)|set(vt))
    delta_transition={}
    for k in keys:
        if bt.get(k)!=vt.get(k):
            delta_transition[k]={
                "baseline":bt.get(k),
                "lens":vt.get(k),
            }

    top=Counter()
    for k in delta_transition:
        top[k.split(".",1)[0]]+=1

    return {
        "seed":seed,
        "first_opponent_cash_divergence_turn":cash_turn,
        "generating_transition_turn":g,
        "same_opponent_action":True,
        "opponent_action":bx["opponent_action"],
        "baseline_self_action":bx["self_action"],
        "lens_self_action":vy["self_action"],
        "baseline_opponent_cash":{
            "before":bx["opponent_cash_pre"],"after":bx["opponent_cash_post"],
            "delta":bx["opponent_cash_post"]-bx["opponent_cash_pre"],
        },
        "lens_opponent_cash":{
            "before":vy["opponent_cash_pre"],"after":vy["opponent_cash_post"],
            "delta":vy["opponent_cash_post"]-vy["opponent_cash_pre"],
        },
        "transition_difference_count":len(delta_transition),
        "transition_difference_top_level_counts":dict(top),
        "transition_differences":delta_transition,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lens",required=True,choices=LENSES)
    args=ap.parse_args()
    rows=[_paired(seed,args.lens) for seed in SEEDS]

    path_counts=Counter()
    top_counts=Counter()
    for r in rows:
        for k in r["transition_differences"]:
            path_counts[k]+=1
            top_counts[k.split(".",1)[0]]+=1

    result={
        "schema":"one-shot-view-lens-first-opponent-cash-transition-v0",
        "question":"With opponent Action unchanged, what Official World transition difference first created opponent cash divergence?",
        "lens":args.lens,
        "seeds":SEEDS,
        "aggregate":{
            "seed_count":len(rows),
            "same_opponent_action_at_generating_transition":sum(r["same_opponent_action"] for r in rows),
            "top_level_difference_occurrences":dict(top_counts),
            "most_common_exact_paths":path_counts.most_common(30),
        },
        "per_seed":rows,
        "boundary":{
            "same_fresh20_worlds":True,
            "body_lens_harness_unchanged":True,
            "transition_is_turn_immediately_before_first_pre_cash_divergence":True,
            "observation_only":True,
            "no_mechanism_claim":True,
        },
    }
    out=Path(f"one_shot_view_lens_first_opponent_cash_transition_{args.lens}_v0.json")
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # concise log: keep exact-path counts and per-seed transition/action/cash;
    # full raw transition diffs remain in artifact.
    print("SUMMARY "+json.dumps({
        "lens":args.lens,
        "aggregate":result["aggregate"],
        "per_seed":[{
            "seed":r["seed"],
            "cash_turn":r["first_opponent_cash_divergence_turn"],
            "transition_turn":r["generating_transition_turn"],
            "opponent_action":r["opponent_action"],
            "baseline_self_action":r["baseline_self_action"],
            "lens_self_action":r["lens_self_action"],
            "baseline_cash":r["baseline_opponent_cash"],
            "lens_cash":r["lens_opponent_cash"],
            "top":r["transition_difference_top_level_counts"],
            "paths":list(r["transition_differences"].keys())[:20],
        } for r in rows]
    },separators=(",",":")))

if __name__=="__main__":
    main()
