import copy
import importlib.util
import json
import sys
from pathlib import Path
from collections import Counter

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as current_self

SEEDS = [7004, 7005, 7006]
SAMPLE_POINTS = {(7,0),(8,0),(9,0),(10,0),(10,23)}

def load_seyamalam(name):
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compact(obs, seat=0):
    farm = obs["farms"][seat]
    c = Counter()
    animals = Counter()
    for row in farm.get("tiles", []):
        for t in row:
            if t is None:
                c["EMPTY"] += 1
            elif t == "LOCKED":
                c["LOCKED"] += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k:
                    c[k] += 1
                a = t.get("animal")
                if a:
                    animals[a] += 1
    return {
        "day": obs["day"],
        "hour": obs["hour"],
        "money": float(farm.get("money", 0)),
        "land": len(farm.get("unlocked_quadrants", [])),
        "hands": len(farm.get("hands", [])),
        "plants": c["PLANT"],
        "pastures": c["PASTURE"],
        "coops": c["COOP"],
        "empty": c["EMPTY"],
        "weeds": c["WEED"],
        "animals": dict(animals),
        "animal_count": sum(animals.values()),
    }

def run_current(seed):
    opp = load_seyamalam(f"opp_current_{seed}")
    samples = []

    def traced_self(obs):
        if (obs["day"], obs["hour"]) in SAMPLE_POINTS:
            samples.append(compact(obs, 0))
        return current_self.agent(obs)

    env = make("kaggriculture", configuration={"seed":seed,"episodeSteps":265}, debug=False)
    env.run([traced_self, opp.agent])
    return samples

def run_seyamalam_self(seed):
    self_s = load_seyamalam(f"self_s_{seed}")
    opp_s = load_seyamalam(f"opp_s_{seed}")
    samples = []

    def traced_self(obs):
        if (obs["day"], obs["hour"]) in SAMPLE_POINTS:
            samples.append(compact(obs, 0))
        return self_s.agent(obs)

    env = make("kaggriculture", configuration={"seed":seed,"episodeSteps":265}, debug=False)
    env.run([traced_self, opp_s.agent])
    return samples

def by_key(rows):
    return {(r["day"],r["hour"]):r for r in rows}

def main():
    out_rows = []
    for seed in SEEDS:
        b = by_key(run_current(seed))
        s = by_key(run_seyamalam_self(seed))
        points = []
        for key in sorted(SAMPLE_POINTS):
            br = b.get(key)
            sr = s.get(key)
            points.append({
                "day": key[0],
                "hour": key[1],
                "current_self": br,
                "seyamalam_self": sr,
                "delta": None if not br or not sr else {
                    "money": sr["money"] - br["money"],
                    "land": sr["land"] - br["land"],
                    "hands": sr["hands"] - br["hands"],
                    "plants": sr["plants"] - br["plants"],
                    "pastures": sr["pastures"] - br["pastures"],
                    "coops": sr["coops"] - br["coops"],
                    "animal_count": sr["animal_count"] - br["animal_count"],
                    "empty": sr["empty"] - br["empty"],
                }
            })
        out_rows.append({"seed":seed,"points":points})

    result = {
        "probe":"seyamalam_on_self_seat_state_trajectory_v0",
        "scope":"Short official-world run through Day10. Same seeds and same seat1 policy; seat0 is Current Self vs independent Seyamalam instance.",
        "boundary":"Observation only. No interpretation or policy extraction.",
        "seeds":SEEDS,
        "rows":out_rows,
    }

    Path("seyamalam_on_self_seat_state_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    for row in out_rows:
        print(json.dumps({
            "seed": row["seed"],
            "points": [
                {
                    "day": p["day"],
                    "hour": p["hour"],
                    "delta": p["delta"],
                }
                for p in row["points"]
            ]
        }, ensure_ascii=False))

if __name__=="__main__":
    main()
