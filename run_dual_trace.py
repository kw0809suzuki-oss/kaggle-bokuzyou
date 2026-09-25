import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as existing_self
import seyamalam_v21 as opponent

records = []


def snapshot(label, obs, action):
    pid = int(obs["player"])
    records.append({
        "side": label,
        "player": pid,
        "day": obs.get("day"),
        "hour": obs.get("hour"),
        "own_farm": obs.get("farms", [])[pid],
        "other_farm": obs.get("farms", [])[1 - pid],
        "own_state": obs.get("private", {}),
        "market": obs.get("market", {}),
        "town": obs.get("town", {}),
        "action": action,
    })


def traced(label, fn):
    def agent(obs):
        action = fn(obs)
        snapshot(label, obs, action)
        return action
    return agent


def main():
    seed = 7001
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.run([traced("self", existing_self.agent), traced("opponent", opponent.agent)])

    rewards = [float(s.reward) for s in env.state]
    summary = {
        "seed": seed,
        "self": rewards[0],
        "opponent": rewards[1],
        "margin": rewards[0] - rewards[1],
        "steps": len(env.steps),
        "trace_records": len(records),
    }

    with open("dual_trace_seed7001.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
