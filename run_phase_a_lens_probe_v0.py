import json
from collections import Counter
from pathlib import Path

from phase_a_lens_model_v0 import evaluate

ROOT = Path(__file__).resolve().parent

def main():
    packet = json.loads((ROOT / "phase_a_blind_cases_v0.json").read_text())
    choices = []
    for case in packet["cases"]:
        result = evaluate(case)
        choices.append(result["direction"])
        print(json.dumps({
            "id": case["id"],
            "direction": result["direction"],
            "scores": result["scores"],
            "best_crop_proxy": result["best_crop_proxy"],
            "best_animal_proxy": result["best_animal_proxy"],
        }, ensure_ascii=False))
    counts = Counter(choices)
    print(json.dumps({
        "summary": True,
        "cases": len(choices),
        "unique_directions": sorted(counts),
        "counts": dict(counts),
        "case_invariant": len(counts) == 1
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
