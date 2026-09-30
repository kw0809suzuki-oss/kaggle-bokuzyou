#!/usr/bin/env python3
"""Observe recent visible market-price history before step250.

Question:
Can simple predeclared price-motion features separate HARVEST>PASS (4)
from PASS>HARVEST (6) in the fixed10?

No classifier fitting. No compound rules. No intervention.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_PARAMS, PRODUCTS

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]
HARVEST_MINUS_PASS = {
    92802001: -7657, 92802002: -104, 92802003: -9578, 92802004: -50,
    92802005: -794, 92802006: 895, 92802007: 190, 92802008: -53,
    92802009: 20312, 92802010: 21568,
}


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


def rank_ratio(prices, target="WHEAT"):
    ratios = {}
    for item in PRODUCTS:
        if item in prices:
            ratios[item] = float(prices[item]) / float(MARKET_PARAMS[item]["base"])
    ordered = sorted(ratios.items(), key=lambda kv: (-kv[1], kv[0]))
    for i, (item, _) in enumerate(ordered, start=1):
        if item == target:
            return i
    return None


def run_history(seed):
    model = load(MODEL, f"hist_model_{seed}")
    opp = load(OPPONENT, f"hist_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    hist = []

    while not env.done:
        o0 = shared(env, 0)
        step = int(o0.get("step", 0) or 0)
        prices = plain((o0.get("market") or {}).get("prices") or {})
        if step <= 250:
            hist.append({
                "step": step,
                "wheat": float(prices.get("WHEAT", 0)),
                "rank": rank_ratio(prices, "WHEAT"),
            })
        if step == 250:
            return hist
        o1 = shared(env, 1)
        env.step([plain(model.agent(o0, env.configuration)), plain(opp.agent(o1))])
    raise RuntimeError(seed)


def at(hist, step):
    for r in hist:
        if r["step"] == step:
            return r
    return None


def features(hist):
    now = at(hist, 250)
    out = {
        "price_now": now["wheat"],
        "rank_now": now["rank"],
    }
    for lag in (1, 6, 12, 24, 48):
        prev = at(hist, 250-lag)
        if prev:
            out[f"delta_{lag}"] = now["wheat"] - prev["wheat"]
            out[f"rank_delta_{lag}"] = now["rank"] - prev["rank"]
    # recent direction counts, no regression fitting
    recent = [r["wheat"] for r in hist if 226 <= r["step"] <= 250]
    diffs = [b-a for a,b in zip(recent, recent[1:])]
    out["up_steps_24"] = sum(d > 0 for d in diffs)
    out["down_steps_24"] = sum(d < 0 for d in diffs)
    out["flat_steps_24"] = sum(d == 0 for d in diffs)
    out["range_24"] = max(recent)-min(recent) if recent else 0
    return out


def perfect_threshold(rows, key):
    vals=[]
    for r in rows:
        v=r["features"].get(key)
        if not isinstance(v,(int,float)):
            return []
        vals.append((float(v),r["label"],r["seed"]))
    uniq=sorted(set(v for v,_,_ in vals))
    ans=[]
    for a,b in zip(uniq,uniq[1:]):
        cut=(a+b)/2
        for orient in ("GT_HARVEST","LE_HARVEST"):
            good=True
            for v,lab,_ in vals:
                pred="HARVEST_GT_PASS" if ((v>cut) if orient=="GT_HARVEST" else (v<=cut)) else "PASS_GT_HARVEST"
                if pred!=lab:
                    good=False; break
            if good:
                ans.append({"feature":key,"threshold":cut,"orientation":orient})
    return ans


def main():
    rows=[]
    for seed in SEEDS:
        hist=run_history(seed)
        f=features(hist)
        label="HARVEST_GT_PASS" if HARVEST_MINUS_PASS[seed]>0 else "PASS_GT_HARVEST"
        row={"seed":seed,"label":label,"delta_harvest_minus_pass":HARVEST_MINUS_PASS[seed],"features":f}
        rows.append(row)
        print("CASE "+json.dumps(row,separators=(",",":")))

    keys=sorted(rows[0]["features"])
    thresholds=[]
    disjoint=[]
    pos=[r for r in rows if r["label"]=="HARVEST_GT_PASS"]
    neg=[r for r in rows if r["label"]=="PASS_GT_HARVEST"]

    for k in keys:
        thresholds.extend(perfect_threshold(rows,k))
        pv={r["features"][k] for r in pos}
        nv={r["features"][k] for r in neg}
        if pv.isdisjoint(nv):
            disjoint.append({
                "feature":k,
                "harvest_values":[r["features"][k] for r in pos],
                "pass_values":[r["features"][k] for r in neg],
            })

    summary={
        "n":len(rows),
        "features_checked":keys,
        "disjoint_feature_count":len(disjoint),
        "disjoint_features":disjoint,
        "perfect_threshold_count":len(thresholds),
        "perfect_thresholds":thresholds,
    }
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("adaptive_circulation_price_history_probe_v0_result.json").write_text(
        json.dumps({"schema":"adaptive-circulation-price-history-probe-v0","summary":summary,"cases":rows,
        "boundary":[
            "No compound classifier is fitted.",
            "A separating feature on fixed10 is observational, not causal.",
            "If no simple feature separates, do not add rescue conditions."
        ]},indent=2)+"\n",encoding="utf-8"
    )


if __name__=="__main__":
    main()
