import json
import sys
from collections import Counter
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as baseline
import seyamalam_v21 as opponent
import phase_a_teacher_minimal_action_v2 as candidate

SEEDS = [7004, 7005, 7006]


def counts(farm):
    c = Counter()
    for row in farm.get("tiles", []):
        for tile in row:
            if tile is None:
                c["empty"] += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "PLANT":
                    c["plants"] += 1
                elif kind in ("PASTURE", "COOP"):
                    c["animal_assets"] += 1
                elif kind == "WEED":
                    c["weeds"] += 1
    return c


def run(agent_fn, seed):
    last = {}

    def traced_self(obs):
        nonlocal last
        last = obs
        return agent_fn(obs)

    env = make("kaggriculture", configuration={"seed": seed,"episodeSteps":265},debug=False)
    env.run([traced_self, opponent.agent])

    farm = last["farms"][0]
    c = counts(farm)
    return {
        "money": float(farm.get("money", 0)),
        "land": len(farm.get("unlocked_quadrants", [])),
        "plants": c["plants"],
        "animal_assets": c["animal_assets"],
        "productive_assets": c["plants"] + c["animal_assets"],
        "empty": c["empty"],
    }


def main():
    rows = []
    for seed in SEEDS:
        b = run(baseline.agent, seed)
        candidate.reset_stats()
        c = run(candidate.agent, seed)
        rows.append({
            "seed": seed,
            "baseline": b,
            "candidate": c,
            "connection": candidate.get_stats(),
        })

    result = {
        "probe": "phase_a_teacher_minimal_action_v2",
        "scope": "Phase A only; seeds 7004-7006; Day7-10 sampled decisions",
        "intervention": "When Self lacks the Teacher-predicted investment class, add one currently affordable minimal action; concrete crop/animal type comes from nearest Teacher state, quantity forced to one.",
        "boundary": "Productive-asset effect only. Not terminal Battle evidence.",
        "rows": rows,
        "summary": {
            "delta_productive_assets": sum(r["candidate"]["productive_assets"] - r["baseline"]["productive_assets"] for r in rows),
            "delta_plants": sum(r["candidate"]["plants"] - r["baseline"]["plants"] for r in rows),
            "delta_animal_assets": sum(r["candidate"]["animal_assets"] - r["baseline"]["animal_assets"] for r in rows),
            "delta_land": sum(r["candidate"]["land"] - r["baseline"]["land"] for r in rows),
            "delta_money": sum(r["candidate"]["money"] - r["baseline"]["money"] for r in rows),
            "decision_points": sum(r["connection"]["decision_points"] for r in rows),
            "direction_missing_from_self": sum(r["connection"]["direction_missing_from_self"] for r in rows),
            "teacher_exemplar_available": sum(r["connection"]["teacher_exemplar_available"] for r in rows),
            "feasible_minimal_action": sum(r["connection"]["feasible_minimal_action"] for r in rows),
            "injected": sum(r["connection"]["injected"] for r in rows),
        },
    }
    Path("phase_a_teacher_minimal_action_result_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result["summary"], ensure_ascii=False))

if __name__ == "__main__":
    main()
