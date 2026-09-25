import json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as self_agent
import seyamalam_v21 as opponent_agent

SEEDS = [7001, 7002, 7003]
SAMPLE_HOURS = {0, 3, 6, 9, 12, 15, 18, 21}
SEED_COST = {"WHEAT":10,"CARROT":20,"TOMATO":50,"STRAWBERRY":100,"MELON":80}
ANIMAL_COST = {"GOOSE":300,"COW":400,"SHEEP":500}
LAND_COST = {1:1000,2:2000,3:4000}

def fib(n):
    a,b=1,1
    for _ in range(max(0,n)):
        a,b=b,a+b
    return a

def tile_counts(farm):
    c=Counter()
    for row in farm.get("tiles",[]):
        for t in row:
            if t is None: c["EMPTY"] += 1
            elif isinstance(t,dict): c[t.get("kind","?")] += 1
    return c

def features(obs):
    pid=obs["player"]
    farm=obs["farms"][pid]
    private=obs["private"]
    c=tile_counts(farm)
    prices=obs["market"]["prices"]
    return {
        "day": obs["day"],
        "hour": obs["hour"],
        "money": float(farm.get("money",0)),
        "land": len(farm.get("unlocked_quadrants",[])),
        "hands": len(farm.get("hands",[])),
        "plants": c["PLANT"],
        "pastures": c["PASTURE"] + c["COOP"],
        "empty": c["EMPTY"],
        "weeds": c["WEED"],
        "seeds": sum(private.get("seeds",{}).values()),
        "p_wheat": prices.get("WHEAT",0),
        "p_strawberry": prices.get("STRAWBERRY",0),
        "p_melon": prices.get("MELON",0),
        "p_milk": prices.get("MILK",0),
        "p_wool": prices.get("WOOL",0),
        "p_egg": prices.get("EGG",0),
    }

def investment_spend(record):
    farm=record["farm"]
    n=farm.get("hires_today",0)
    lands=len(farm.get("unlocked_quadrants",[]))
    out=Counter()
    for o in record["action"].get("market",[]):
        if not o: continue
        op=o[0]
        if op=="HIRE":
            out["THROUGHPUT"] += fib(n); n += 1
        elif op=="BUY_LAND":
            out["SURFACE"] += LAND_COST.get(lands,0); lands += 1
        elif op=="BUY_SEED" and len(o)>=3:
            out["CROP_ENGINE"] += SEED_COST.get(o[1],0) * int(o[2])
        elif op=="BUY_ANIMAL" and len(o)>=3:
            out["ANIMAL_ENGINE"] += ANIMAL_COST.get(o[1],0) * int(o[2])
    return out

def collect(seed):
    records=[]
    def traced_opp(obs):
        action=opponent_agent.agent(obs)
        records.append({
            "obs_features": features(obs),
            "farm": obs["farms"][obs["player"]],
            "action": action,
        })
        return action
    env=make("kaggriculture",configuration={"seed":seed,"episodeSteps":265},debug=False)
    env.run([self_agent.agent,traced_opp])

    rows=[]
    for i,r in enumerate(records):
        f=r["obs_features"]
        if not (7 <= f["day"] <= 10 and f["hour"] in SAMPLE_HOURS):
            continue
        spend=Counter()
        for future in records[i:i+24]:
            spend.update(investment_spend(future))
        label = spend.most_common(1)[0][0] if spend else "HOLD"
        rows.append({"seed":seed,"features":f,"label":label,"spend":dict(spend)})
    return rows

FEATURES=[
    "day","hour","money","land","hands","plants","pastures","empty","weeds","seeds",
    "p_wheat","p_strawberry","p_melon","p_milk","p_wool","p_egg"
]

def scales(rows):
    mm={}
    for k in FEATURES:
        vals=[r["features"][k] for r in rows]
        lo,hi=min(vals),max(vals)
        mm[k]=(lo,hi)
    return mm

def dist(a,b,mm):
    s=0.0
    for k in FEATURES:
        lo,hi=mm[k]
        den=(hi-lo) or 1.0
        d=(a[k]-b[k])/den
        s += d*d
    return math.sqrt(s)

def predict(train, f, k=5):
    mm=scales(train)
    near=sorted(((dist(r["features"],f,mm),r["label"]) for r in train), key=lambda x:x[0])[:k]
    votes=defaultdict(float)
    for d,label in near:
        votes[label] += 1.0/(d+1e-6)
    return max(votes,key=votes.get)

def main():
    by_seed={s:collect(s) for s in SEEDS}
    fold_results=[]
    all_cases=[]
    for test_seed in SEEDS:
        train=[r for s,rows in by_seed.items() if s!=test_seed for r in rows]
        test=by_seed[test_seed]
        correct=0
        for r in test:
            pred=predict(train,r["features"])
            ok=pred==r["label"]
            correct += ok
            all_cases.append({
                "seed":test_seed,
                "day":r["features"]["day"],
                "hour":r["features"]["hour"],
                "predicted":pred,
                "observed":r["label"],
                "match":ok,
            })
        fold_results.append({
            "test_seed":test_seed,
            "cases":len(test),
            "correct":correct,
            "accuracy":correct/len(test) if test else 0,
        })

    total=sum(x["cases"] for x in fold_results)
    correct=sum(x["correct"] for x in fold_results)
    result={
        "probe":"phase_a_teacher_knn_v0",
        "scope":"Phase A only; environment truncated at Day 11",
        "lens":"raise productive asset value; labels are observed strong-opponent investment direction over next 24 turns",
        "boundary":"behavioral reference, not proof of optimality",
        "seeds":SEEDS,
        "folds":fold_results,
        "summary":{
            "cases":total,
            "correct":correct,
            "accuracy":correct/total if total else 0,
            "label_counts":dict(Counter(r["label"] for rows in by_seed.values() for r in rows)),
        },
        "cases":all_cases,
    }
    Path("phase_a_teacher_result_v0.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result["summary"],ensure_ascii=False))

if __name__=="__main__":
    main()
