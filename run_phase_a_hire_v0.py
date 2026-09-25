import sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import phase_a_hire_v0 as candidate
import seyamalam_v21 as opponent

def main():
    seed=7001
    env=make("kaggriculture",configuration={"seed":seed},debug=True)
    env.run([candidate.agent,opponent.agent])
    rewards=[float(s.reward) for s in env.state]
    print({"seed":seed,"self":rewards[0],"opponent":rewards[1],"margin":rewards[0]-rewards[1],"steps":len(env.steps)})

if __name__=="__main__":
    main()
