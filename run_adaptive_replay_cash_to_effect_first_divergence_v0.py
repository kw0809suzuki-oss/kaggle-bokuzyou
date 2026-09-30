#!/usr/bin/env python3
"""Cash -> first realized self-effect observation v0.

Known boundary:
    Candidate self Cash first differs at step97 by +6 (seed04 over seed05).
    Prior transition trace first sees a new self-local non-Cash difference at
    step111: private.seeds.STRAWBERRY.

This probe observes only the immediately preceding transition 110 -> 111 and
checks whether the same requested action produces a different realized
resource effect.

No new Guard, Recovery, or policy change.
"""
from __future__ import annotations
import importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS={"positive":93803004,"negative":93803005}
TARGET_STEP=110


def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod


def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]


def count_plants(tiles):
    out={}
    for row in tiles:
        for tile in row:
            if isinstance(tile,dict) and tile.get("kind")=="PLANT":
                crop=tile.get("crop")
                out[crop]=out.get(crop,0)+1
    return out


def delta_dict(pre,post):
    keys=sorted(set(pre)|set(post))
    return {k:post.get(k,0)-pre.get(k,0) for k in keys if post.get(k,0)!=pre.get(k,0)}


def run(seed,label):
    model=load(MODEL,f"effect_model_{label}_{seed}_{os.getpid()}")
    opp=load(OPPONENT,f"effect_opp_{label}_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0=plain(shared(env,0))
        obs1=plain(shared(env,1))
        step=int(obs0.get("step",0) or 0)
        a0=plain(model.agent(obs0))
        a1=plain(opp.agent(obs1))

        if step!=TARGET_STEP:
            env.step([a0,a1])
            continue

        farm=obs0["farms"][0]
        pre={
            "cash":float(farm["money"]),
            "seeds":plain(obs0["private"]["seeds"]),
            "shed":plain(obs0["private"]["shed"]),
            "hands":len(farm.get("hands",[]) or []),
            "hires_today":int(farm.get("hires_today",0) or 0),
            "unlocked_quadrants":plain(farm.get("unlocked_quadrants",[])),
            "plants":count_plants(farm["tiles"]),
            "market_prices":plain(obs0["market"]["prices"]),
            "market_inventory":plain(obs0["market"]["inventory"]),
        }

        env.step([a0,a1])
        obs111=plain(shared(env,0))
        farm2=obs111["farms"][0]
        post={
            "cash":float(farm2["money"]),
            "seeds":plain(obs111["private"]["seeds"]),
            "shed":plain(obs111["private"]["shed"]),
            "hands":len(farm2.get("hands",[]) or []),
            "hires_today":int(farm2.get("hires_today",0) or 0),
            "unlocked_quadrants":plain(farm2.get("unlocked_quadrants",[])),
            "plants":count_plants(farm2["tiles"]),
            "market_prices":plain(obs111["market"]["prices"]),
            "market_inventory":plain(obs111["market"]["inventory"]),
        }

        return {
            "seed":seed,
            "step":TARGET_STEP,
            "subject_action":a0,
            "opponent_action":a1,
            "pre":pre,
            "post":post,
            "effect":{
                "cash_delta":post["cash"]-pre["cash"],
                "seed_delta":delta_dict(pre["seeds"],post["seeds"]),
                "shed_delta":delta_dict(pre["shed"],post["shed"]),
                "hands_delta":post["hands"]-pre["hands"],
                "hires_today_delta":post["hires_today"]-pre["hires_today"],
                "unlocked_added":[x for x in post["unlocked_quadrants"] if x not in pre["unlocked_quadrants"]],
                "plant_count_delta":delta_dict(pre["plants"],post["plants"]),
            }
        }
    raise RuntimeError("target step not reached")


def main():
    runs={label:run(seed,label) for label,seed in SEEDS.items()}
    p=runs["positive"]; n=runs["negative"]

    summary={
        "subject_action_equal":p["subject_action"]==n["subject_action"],
        "opponent_action_equal":p["opponent_action"]==n["opponent_action"],
        "positive_pre_cash":p["pre"]["cash"],
        "negative_pre_cash":n["pre"]["cash"],
        "pre_cash_difference_positive_minus_negative":p["pre"]["cash"]-n["pre"]["cash"],
        "positive_effect":p["effect"],
        "negative_effect":n["effect"],
        "seed_effect_equal":p["effect"]["seed_delta"]==n["effect"]["seed_delta"],
        "hands_effect_equal":p["effect"]["hands_delta"]==n["effect"]["hands_delta"],
        "land_effect_equal":p["effect"]["unlocked_added"]==n["effect"]["unlocked_added"],
        "plant_effect_equal":p["effect"]["plant_count_delta"]==n["effect"]["plant_count_delta"],
    }

    print("CASH_TO_EFFECT_110_111_SUMMARY "+json.dumps(summary,separators=(",",":")))
    print("CASH_TO_EFFECT_110_111_POSITIVE "+json.dumps(p,separators=(",",":")))
    print("CASH_TO_EFFECT_110_111_NEGATIVE "+json.dumps(n,separators=(",",":")))

    Path("adaptive_replay_cash_to_effect_first_divergence_v0.json").write_text(
        json.dumps({
            "schema":"adaptive-replay-cash-to-effect-first-divergence-v0",
            "question":"Does the pre-existing Cash gap first become a different realized self-resource effect in transition step110->111?",
            "world":"Seyamalam v21 / subject seat0",
            "summary":summary,
            "runs":runs,
            "boundary":{
                "target_transition":"step110 pre-state -> step111 post-state",
                "reason_for_target":"Prior trace first observed private.seeds.STRAWBERRY difference at step111.",
                "no_new_guard":True,
                "no_recovery":True,
                "no_terminal_causal_claim":True
            }
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )


if __name__=="__main__":
    main()
