import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SELF_DIR = ROOT / "selfsrc"
OPP_PATH = ROOT / "opponents" / "seyamalam_v21.py"
SEEDS = [7004,7005,7006,7007,7008]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)
    return module


def run_one(seed, variant):
    current = load_module(SELF_DIR / "whole_flow_control_agent.py", f"current_{variant}_{seed}")
    opponent = load_module(OPP_PATH, f"opponent_{variant}_{seed}")

    if variant == "baseline":
        seat0_agent = current.agent
    else:
        replacement = load_module(OPP_PATH, f"replacement_{variant}_{seed}")
        used = {"done": False}

        def seat0_agent(obs):
            if not used["done"] and obs["day"] == 0 and obs["hour"] == 0:
                _ = current.agent(obs)  # preserve Current Self's h0 call/state update; discard output
                used["done"] = True
                return replacement.agent(obs)
            return current.agent(obs)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    steps = env.run([seat0_agent, opponent.agent])
    terminal_self = float(steps[-1][0]["reward"])
    terminal_opp = float(steps[-1][1]["reward"])
    return {
        "seed": seed,
        "variant": variant,
        "terminal_self": terminal_self,
        "terminal_opponent": terminal_opp,
    }


def child_main(seed, variant):
    print(json.dumps(run_one(seed, variant), ensure_ascii=False))


def parent_main():
    rows = []
    for seed in SEEDS:
        pair = {}
        for variant in ("baseline", "candidate"):
            proc = subprocess.run(
                [sys.executable, __file__, "--child", "--seed", str(seed), "--variant", variant],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            pair[variant] = result

        delta = pair["candidate"]["terminal_self"] - pair["baseline"]["terminal_self"]
        rows.append({
            "seed": seed,
            "baseline_terminal_self": pair["baseline"]["terminal_self"],
            "candidate_terminal_self": pair["candidate"]["terminal_self"],
            "paired_delta": delta,
        })

    baseline_mean = sum(r["baseline_terminal_self"] for r in rows) / len(rows)
    candidate_mean = sum(r["candidate_terminal_self"] for r in rows) / len(rows)
    mean_delta = candidate_mean - baseline_mean
    improved = sum(r["paired_delta"] > 0 for r in rows)
    worsened = sum(r["paired_delta"] < 0 for r in rows)
    same = sum(r["paired_delta"] == 0 for r in rows)

    result = {
        "probe": "day0_h0_single_replacement_terminal_v0",
        "question": "Does replacing only the first observed divergence action (Day0 h0) with Seyamalam's action move terminal self?",
        "design": {
            "baseline": "Day0 h0 Current Self action; Current Self thereafter.",
            "candidate": "At Day0 h0, call Current Self and discard its output; execute fresh Seyamalam action on the exact same observation; Current Self thereafter.",
            "opponent": "Fresh Seyamalam instance.",
            "process_isolation": "Each seed/variant runs in a fresh Python subprocess.",
            "seeds": SEEDS,
        },
        "boundary": "Terminal response only. No downstream-state interpretation in this probe.",
        "rows": rows,
        "summary": {
            "baseline_absolute_mean_self": baseline_mean,
            "candidate_absolute_mean_self": candidate_mean,
            "delta": mean_delta,
            "improved_seeds": improved,
            "worsened_seeds": worsened,
            "same_seeds": same,
        },
    }

    Path("day0_h0_single_replacement_terminal_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--variant", choices=["baseline","candidate"])
    args = p.parse_args()
    if args.child:
        child_main(args.seed, args.variant)
    else:
        parent_main()
