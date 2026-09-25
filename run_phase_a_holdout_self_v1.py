import json, sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as self_agent
import seyamalam_v21 as opponent_agent
from run_phase_a_teacher_probe_v0 import collect, predict, features, investment_spend, SAMPLE_HOURS

TRAIN_SEEDS=[7001,7002,7003]
TEST_SEED=7004

def collect_pair(seed):
    self_records=[]
    opp_records=[]

    def traced_self(obs):
        action=self_agent.agent(obs)
        self_records.append({"features":features(obs),"action":action})
        return action

    def traced_opp(obs):
        action=opponent_agent.agent(obs)
        opp_records.append({
            "features":features(obs),
            "farm":obs["farms"][obs["player"]],
            "action":action,
        })
        return action

    env=make("kaggriculture",configuration={"seed":seed,"episodeSteps":265},debug=False)
    env.run([traced_self,traced_opp])

    opp_rows=[]
    for i,r in enumerate(opp_records):
        f=r["features"]
        if not (7<=f["day"]<=10 and f["hour"] in SAMPLE_HOURS):
            continue
        spend=Counter()
        for future in opp_records[i:i+24]:
            spend.update(investment_spend(future))
        label=spend.most_common(1)[0][0] if spend else "HOLD"
        opp_rows.append({"features":f,"label":label})

    self_rows=[
        r for r in self_records
        if 7<=r["features"]["day"]<=10 and r["features"]["hour"] in SAMPLE_HOURS
    ]
    return opp_rows,self_rows

def main():
    train=[r for s in TRAIN_SEEDS for r in collect(s)]
    test_opp,test_self=collect_pair(TEST_SEED)

    opp_cases=[]
    correct=0
    for r in test_opp:
        p=predict(train,r["features"])
        ok=p==r["label"]
        correct+=ok
        opp_cases.append({
            "day":r["features"]["day"],"hour":r["features"]["hour"],
            "predicted":p,"observed":r["label"],"match":ok,
        })

    self_cases=[]
    for r in test_self:
        f=r["features"]
        p=predict(train,f)
        self_cases.append({
            "day":f["day"],"hour":f["hour"],
            "money":f["money"],"land":f["land"],"hands":f["hands"],
            "plants":f["plants"],"pastures":f["pastures"],"empty":f["empty"],
            "direction":p,
        })

    result={
        "probe":"phase_a_holdout_and_self_shadow_v1",
        "train_seeds":TRAIN_SEEDS,
        "test_seed":TEST_SEED,
        "boundary":"model imitates observed strong-opponent investment direction; not yet Battle evidence",
        "opponent_holdout":{
            "cases":len(test_opp),
            "correct":correct,
            "accuracy":correct/len(test_opp) if test_opp else 0,
            "cases_detail":opp_cases,
        },
        "self_shadow":{
            "cases":len(self_cases),
            "direction_counts":dict(Counter(x["direction"] for x in self_cases)),
            "cases_detail":self_cases,
        }
    }
    Path("phase_a_holdout_self_result_v1.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({
        "holdout_accuracy":result["opponent_holdout"]["accuracy"],
        "self_direction_counts":result["self_shadow"]["direction_counts"]
    },ensure_ascii=False))

if __name__=="__main__":
    main()
