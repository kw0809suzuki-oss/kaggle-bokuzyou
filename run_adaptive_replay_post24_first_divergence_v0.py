#!/usr/bin/env python3
"""Post-step24 First Divergence Probe v0.

Question:
    Starting from the confirmed identical observable step24 state, when do
    positive seed 93803004 and negative seed 93803005 first become observably
    different?

World:
    Seyamalam v21, subject seat0.

Compare separately:
    A) frozen DECEM Replay baseline
    B) Adaptive Replay Contract Runtime v0

Observe only:
    first World-state divergence
    first subject Action divergence
    first opponent Action divergence

No causal explanation and no new intervention.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODELS={
    "baseline": ROOT/"decem_replay_distilled_157026_v0.py",
    "candidate": ROOT/"adaptive_replay_contract_runtime_v0.py",
}
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
POSITIVE_SEED=93803004
NEGATIVE_SEED=93803005
START_STEP=24


def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x
    if hasattr(x,"items"):
        return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):
        return [plain(v) for v in x]
    return x


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"):
        mod.reset_agent()
    return mod


def shared(env,seat):
    return env._Environment__get_shared_state(seat)["observation"]


def recursive_diff(a,b,path=""):
    diffs=[]
    if type(a)!=type(b):
        return [{"path":path or "$","a":a,"b":b,"kind":"type"}]
    if isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            p=f"{path}.{k}" if path else str(k)
            if k not in a:
                diffs.append({"path":p,"a":"<missing>","b":b[k],"kind":"missing_a"})
            elif k not in b:
                diffs.append({"path":p,"a":a[k],"b":"<missing>","kind":"missing_b"})
            else:
                diffs.extend(recursive_diff(a[k],b[k],p))
        return diffs
    if isinstance(a,list):
        if len(a)!=len(b):
            diffs.append({"path":(path or "$")+".length","a":len(a),"b":len(b),"kind":"length"})
        for i,(x,y) in enumerate(zip(a,b)):
            diffs.extend(recursive_diff(x,y,f"{path}[{i}]"))
        return diffs
    if a!=b:
        diffs.append({"path":path or "$","a":a,"b":b,"kind":"value"})
    return diffs


def run(seed,model_path,tag):
    model=load(model_path,f"{tag}_model_{seed}_{os.getpid()}")
    opponent=load(OPPONENT,f"{tag}_opp_{seed}_{os.getpid()}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    trace=[]

    while not env.done:
        obs0=plain(shared(env,0))
        obs1=plain(shared(env,1))
        step=int(obs0.get("step",0) or 0)
        a0=plain(model.agent(obs0))
        a1=plain(opponent.agent(obs1))
        if step>=START_STEP:
            trace.append({
                "step":step,
                "world":obs0,
                "subject_action":a0,
                "opponent_action":a1,
            })
        env.step([a0,a1])

    final=plain(env.state[0].observation)
    return {
        "trace":trace,
        "terminal_self":float(final["farms"][0]["money"]),
        "terminal_opponent":float(final["farms"][1]["money"]),
    }


def first_divergence(pos_trace,neg_trace,field):
    by_step_pos={r["step"]:r[field] for r in pos_trace}
    by_step_neg={r["step"]:r[field] for r in neg_trace}
    for step in sorted(set(by_step_pos)&set(by_step_neg)):
        a=by_step_pos[step]
        b=by_step_neg[step]
        if a!=b:
            diffs=recursive_diff(a,b)
            return {
                "step":step,
                "diff_count":len(diffs),
                "sample_diffs":diffs[:100],
            }
    return None


def main():
    out={
        "schema":"adaptive-replay-post24-first-divergence-v0",
        "world":"Seyamalam v21 / subject seat0",
        "positive_seed":POSITIVE_SEED,
        "negative_seed":NEGATIVE_SEED,
        "models":{},
        "boundary":{
            "no_new_guard":True,
            "no_causal_explanation":True,
            "question":"When does hidden seed difference first become observable after identical step24 state?",
        },
    }

    for name,path in MODELS.items():
        pos=run(POSITIVE_SEED,path,f"{name}_pos")
        neg=run(NEGATIVE_SEED,path,f"{name}_neg")

        step24_pos=next(r for r in pos["trace"] if r["step"]==24)
        step24_neg=next(r for r in neg["trace"] if r["step"]==24)

        result={
            "step24_world_equal":step24_pos["world"]==step24_neg["world"],
            "step24_subject_action_equal":step24_pos["subject_action"]==step24_neg["subject_action"],
            "step24_opponent_action_equal":step24_pos["opponent_action"]==step24_neg["opponent_action"],
            "first_world_divergence":first_divergence(pos["trace"],neg["trace"],"world"),
            "first_subject_action_divergence":first_divergence(pos["trace"],neg["trace"],"subject_action"),
            "first_opponent_action_divergence":first_divergence(pos["trace"],neg["trace"],"opponent_action"),
            "positive_terminal_self":pos["terminal_self"],
            "negative_terminal_self":neg["terminal_self"],
            "terminal_self_difference_positive_minus_negative":pos["terminal_self"]-neg["terminal_self"],
        }
        out["models"][name]=result

        compact={
            "model":name,
            "step24_world_equal":result["step24_world_equal"],
            "step24_subject_action_equal":result["step24_subject_action_equal"],
            "step24_opponent_action_equal":result["step24_opponent_action_equal"],
            "first_world_divergence_step":None if result["first_world_divergence"] is None else result["first_world_divergence"]["step"],
            "first_world_diff_paths":[] if result["first_world_divergence"] is None else [d["path"] for d in result["first_world_divergence"]["sample_diffs"][:30]],
            "first_subject_action_divergence_step":None if result["first_subject_action_divergence"] is None else result["first_subject_action_divergence"]["step"],
            "first_opponent_action_divergence_step":None if result["first_opponent_action_divergence"] is None else result["first_opponent_action_divergence"]["step"],
            "positive_terminal_self":result["positive_terminal_self"],
            "negative_terminal_self":result["negative_terminal_self"],
            "terminal_self_difference_positive_minus_negative":result["terminal_self_difference_positive_minus_negative"],
        }
        print("POST24_FIRST_DIVERGENCE "+json.dumps(compact,separators=(",",":")))

    Path("adaptive_replay_post24_first_divergence_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )


if __name__=="__main__":
    main()
