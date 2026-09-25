import copy
import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as current_self
import seyamalam_v21 as live_opponent

SAMPLES = {
    (7, 0), (7, 6),
    (8, 0), (8, 6),
    (9, 0), (9, 6),
    (10, 0), (10, 6),
}
SEED = 7004


def load_shadow():
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location("seyamalam_shadow_v21", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compact_state(obs):
    p = obs["player"]
    me = obs["farms"][p]
    counts = {"PLANT": 0, "PASTURE": 0, "COOP": 0, "EMPTY": 0, "WEED": 0, "LOCKED": 0}
    for row in me.get("tiles", []):
        for t in row:
            if t is None:
                counts["EMPTY"] += 1
            elif t == "LOCKED":
                counts["LOCKED"] += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k in counts:
                    counts[k] += 1
    return {
        "day": obs["day"],
        "hour": obs["hour"],
        "money": me.get("money", 0),
        "land": len(me.get("unlocked_quadrants", [])),
        "hands": len(me.get("hands", [])),
        "plants": counts["PLANT"],
        "pastures": counts["PASTURE"],
        "coops": counts["COOP"],
        "empty": counts["EMPTY"],
        "weeds": counts["WEED"],
        "shed": copy.deepcopy(obs.get("private", {}).get("shed", {})),
        "seeds": copy.deepcopy(obs.get("private", {}).get("seeds", {})),
    }


def normalize(action):
    return {
        "farmer": action.get("farmer", ["PASS"]),
        "hands": action.get("hands", []),
        "market": action.get("market", []),
    }


def market_ops(action):
    return [o[0] for o in action.get("market", []) if o]


def main():
    shadow = load_shadow()
    rows = []

    def traced_self(obs):
        shadow_action = normalize(shadow.agent(copy.deepcopy(obs)))
        self_action = normalize(current_self.agent(copy.deepcopy(obs)))

        if (obs["day"], obs["hour"]) in SAMPLES:
            rows.append({
                "state": compact_state(obs),
                "current_self_action": self_action,
                "seyamalam_on_same_self_state": shadow_action,
                "same_action": self_action == shadow_action,
                "current_market_ops": market_ops(self_action),
                "seyamalam_market_ops": market_ops(shadow_action),
            })
        return self_action

    env = make("kaggriculture", configuration={"seed": SEED, "episodeSteps": 265}, debug=False)
    env.run([traced_self, live_opponent.agent])

    result = {
        "probe": "same_self_state_current_vs_seyamalam_v0",
        "seed": SEED,
        "scope": "One real Battle trajectory. Shadow Seyamalam receives the exact seat-0 Self observation every turn from the start; only 8 Phase-A states are recorded.",
        "boundary": "Observation only. Shadow actions are not executed and do not alter the Battle.",
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "same_actions": sum(1 for r in rows if r["same_action"]),
            "different_actions": sum(1 for r in rows if not r["same_action"]),
        },
    }
    Path("same_self_state_seyamalam_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result["summary"], ensure_ascii=False))
    for r in rows:
        s = r["state"]
        print(json.dumps({
            "day": s["day"],
            "hour": s["hour"],
            "money": s["money"],
            "land": s["land"],
            "hands": s["hands"],
            "current_market_ops": r["current_market_ops"],
            "seyamalam_market_ops": r["seyamalam_market_ops"],
            "same_action": r["same_action"],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
