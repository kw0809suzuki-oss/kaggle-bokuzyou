#!/usr/bin/env python3
"""Fixed10 baseline vs Price Board v0."""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "adaptive_circulation_runtime_v0.py"
CANDIDATE = ROOT / "adaptive_circulation_price_board_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def run_one(seed, model_path, tag):
    model = load(model_path, f"{tag}_{seed}")
    opp = load(OPPONENT, f"opp_{tag}_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))
        env.step([a0, a1])
    final = plain(env.state[0].observation)
    s = float(final["farms"][0]["money"])
    o = float(final["farms"][1]["money"])
    return {
        "terminal_self": s,
        "terminal_opponent": o,
        "margin": s-o,
        "trigger_count": int(getattr(model, "trigger_count", 0)),
        "event": plain(getattr(model, "last_event", None)),
    }


def main():
    rows=[]
    for seed in SEEDS:
        b=run_one(seed, BASELINE, "base")
        c=run_one(seed, CANDIDATE, "price")
        d=c["terminal_self"]-b["terminal_self"]
        row={"seed":seed,"baseline":b,"candidate":c,"delta_self":d}
        rows.append(row)
        ev=c.get("event") or {}
        pv=ev.get("price_view") or {}
        print("CASE "+json.dumps({
            "seed":seed,
            "choice":ev.get("price_choice"),
            "crop":pv.get("crop"),
            "price":(pv.get("prices") or {}).get(pv.get("crop")),
            "target_ratio":pv.get("target_ratio"),
            "mean_ratio":pv.get("mean_ratio"),
            "baseline":b["terminal_self"],
            "candidate":c["terminal_self"],
            "delta":d,
        },ensure_ascii=False,separators=(",",":")))

    ds=[r["delta_self"] for r in rows]
    choices={}
    for r in rows:
        ch=((r["candidate"].get("event") or {}).get("price_choice"))
        choices[ch]=choices.get(ch,0)+1

    summary={
        "n":len(rows),
        "triggered":sum(r["candidate"]["trigger_count"]>0 for r in rows),
        "choices":choices,
        "improved":sum(x>0 for x in ds),
        "worsened":sum(x<0 for x in ds),
        "same":sum(x==0 for x in ds),
        "mean_delta_self":statistics.mean(ds),
        "median_delta_self":statistics.median(ds),
        "min_delta_self":min(ds),
        "max_delta_self":max(ds),
        "baseline_mean":statistics.mean(r["baseline"]["terminal_self"] for r in rows),
        "candidate_mean":statistics.mean(r["candidate"]["terminal_self"] for r in rows),
    }
    out={
        "schema":"adaptive-circulation-price-board-v0",
        "objective":"terminal_self",
        "rule":"target normalized current price vs mean normalized current price across market products",
        "summary":summary,
        "cases":rows,
        "boundary":[
            "This is one fixed10 A/B at the already-observed first opportunity.",
            "No seed-specific threshold is fitted.",
            "A positive result would justify broader validation, not immediate promotion.",
            "A negative result rejects this rule as written; no rescue condition is added."
        ]
    }
    Path("adaptive_circulation_price_board_v0_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))


if __name__=="__main__":
    main()
