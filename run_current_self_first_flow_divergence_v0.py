import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as current_self

SEED = 7004
FIELDS = [
    "money","land","hands","plants","pastures","coops","animal_count",
    "crop_yield_units","animal_yield_units","shed_sellable"
]
SELLABLE = ("WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","MILK","WOOL","EGG","FERTILIZER")

def load_seyamalam(name):
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def snapshot(obs):
    me = obs["farms"][0]
    private = obs["private"]
    c = Counter()
    animals = Counter()
    crop_yield_units = 0.0
    animal_yield_units = 0.0

    for row in me.get("tiles", []):
        for t in row:
            if t is None:
                c["EMPTY"] += 1
            elif t == "LOCKED":
                c["LOCKED"] += 1
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind:
                    c[kind] += 1
                y = float(t.get("yield_units", 0) or 0)
                if kind == "PLANT":
                    crop_yield_units += y
                elif kind in ("PASTURE","COOP"):
                    a = t.get("animal")
                    if a:
                        animals[a] += 1
                    animal_yield_units += y

    shed = private.get("shed", {})
    shed_sellable = sum(float(shed.get(k, 0) or 0) for k in SELLABLE)

    return {
        "day": obs["day"],
        "hour": obs["hour"],
        "money": float(me.get("money", 0)),
        "land": len(me.get("unlocked_quadrants", [])),
        "hands": len(me.get("hands", [])),
        "plants": c["PLANT"],
        "pastures": c["PASTURE"],
        "coops": c["COOP"],
        "animal_count": sum(animals.values()),
        "crop_yield_units": crop_yield_units,
        "animal_yield_units": animal_yield_units,
        "shed_sellable": shed_sellable,
    }

def normalize_action(action):
    return {
        "farmer": copy.deepcopy(action.get("farmer", ["PASS"])),
        "hands": copy.deepcopy(action.get("hands", [])),
        "market": copy.deepcopy(action.get("market", [])),
    }

def run(agent_kind):
    opp = load_seyamalam(f"opp_{agent_kind}")
    self_s = load_seyamalam(f"self_{agent_kind}") if agent_kind == "seyamalam" else None
    rows = []

    def traced(obs):
        action = self_s.agent(obs) if self_s else current_self.agent(obs)
        rows.append({
            "state": snapshot(obs),
            "action": normalize_action(action),
        })
        return action

    env = make("kaggriculture", configuration={"seed":SEED}, debug=False)
    env.run([traced, opp.agent])
    return rows

def main():
    cur = run("current")
    sey = run("seyamalam")
    n = min(len(cur), len(sey))

    first_state_diff = None
    first_action_diff = None

    for i in range(n):
        cs, ss = cur[i]["state"], sey[i]["state"]

        if first_action_diff is None and cur[i]["action"] != sey[i]["action"]:
            first_action_diff = {
                "turn": i,
                "day": cs["day"],
                "hour": cs["hour"],
                "current_action": cur[i]["action"],
                "seyamalam_action": sey[i]["action"],
                "current_state": cs,
                "seyamalam_state": ss,
            }

        diffs = {k: ss[k] - cs[k] for k in FIELDS if ss[k] != cs[k]}
        if first_state_diff is None and diffs:
            prev = None
            if i > 0:
                prev = {
                    "turn": i-1,
                    "day": cur[i-1]["state"]["day"],
                    "hour": cur[i-1]["state"]["hour"],
                    "current_state": cur[i-1]["state"],
                    "seyamalam_state": sey[i-1]["state"],
                    "current_action": cur[i-1]["action"],
                    "seyamalam_action": sey[i-1]["action"],
                }
            first_state_diff = {
                "turn": i,
                "day": cs["day"],
                "hour": cs["hour"],
                "diffs": diffs,
                "current_state": cs,
                "seyamalam_state": ss,
                "previous_turn": prev,
            }
            break

    result = {
        "probe": "current_self_first_flow_divergence_vs_seyamalam_v0",
        "seed": SEED,
        "scope": "Same seed, same seat1 Seyamalam policy, seat0 Current Self vs independent Seyamalam. Full episode observed from turn 0.",
        "flow_fields": FIELDS,
        "boundary": "Mechanical first-difference detection only. No causal interpretation.",
        "first_action_diff": first_action_diff,
        "first_state_diff": first_state_diff,
    }

    Path("current_self_first_flow_divergence_vs_seyamalam_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
