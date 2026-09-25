import sys, statistics
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as baseline
import phase_a_pair_hire_v2 as candidate
import seyamalam_v21 as opponent

SEEDS=list(range(7001,7011))

def run(agent, seed):
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([agent,opponent.agent])
    r=[float(s.reward) for s in env.state]
    return r[0],r[1],r[0]-r[1]

def main():
    rows=[]
    for seed in SEEDS:
        b=run(baseline.agent,seed)
        c=run(candidate.agent,seed)
        rows.append((seed,b,c))
        print({"seed":seed,"baseline_self":b[0],"candidate_self":c[0],"delta_self":c[0]-b[0],
               "baseline_margin":b[2],"candidate_margin":c[2],"delta_margin":c[2]-b[2]})
    bmean=statistics.mean(x[1][0] for x in rows)
    cmean=statistics.mean(x[2][0] for x in rows)
    bm=statistics.mean(x[1][2] for x in rows)
    cm=statistics.mean(x[2][2] for x in rows)
    print({"summary":True,"baseline_absolute_mean_self":bmean,"candidate_absolute_mean_self":cmean,
           "delta_self":cmean-bmean,"baseline_mean_margin":bm,"candidate_mean_margin":cm,
           "delta_margin":cm-bm,
           "improved":sum(x[2][0]>x[1][0] for x in rows),
           "worsened":sum(x[2][0]<x[1][0] for x in rows),
           "equal":sum(x[2][0]==x[1][0] for x in rows)})

if __name__=="__main__": main()
