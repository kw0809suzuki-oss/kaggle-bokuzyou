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

        def seat0_agent(obs):
            if obs["day"] == 0:
                _ = current.agent(obs)  # preserve Current Self call/state update; discard output
                return replacement.agent(obs)
            return current.agent(obs)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    steps = env.run([seat0_agent, opponent.agent])
    return {
        "seed": seed,
        "variant": variant,
        "terminal_self": float(steps[-1][0]["reward"]),
        "terminal_opponent": float(steps[-1][1]["reward"]),
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
                check=True, capture_output=True, text=True,
            )
            pair[variant] = json.loads(proc.stdout.strip().splitlines()[-1])

        b = pair["baseline"]["terminal_self"]
        c = pair["candidate"]["terminal_self"]
        rows.append({
            "seed": seed,
            "baseline_terminal_self": b,
            "candidate_terminal_self": c,
            "paired_delta": c - b,
        })

    bm = sum(r["baseline_terminal_self"] for r in rows) / len(rows)
    cm = sum(r["candidate_terminal_self"] for r in rows) / len(rows)

    result = {
        "probe": "day0_full_prefix_replacement_terminal_v0",
        "question": "Does replacing only Day0 (24 turns) with Seyamalam actions move terminal self?",
        "design": {
            "baseline": "Current Self throughout.",
            "candidate": "During Day0, call Current Self each turn and discard output; execute a fresh Seyamalam instance instead. Resume Current Self from Day1.",
            "opponent": "Fresh Seyamalam instance.",
            "process_isolation": "Each seed/variant runs in a fresh Python subprocess.",
            "seeds": SEEDS
        },
        "boundary": "Terminal response only. No intermediate-state interpretation.",
        "rows": rows,
        "summary": {
            "baseline_absolute_mean_self": bm,
            "candidate_absolute_mean_self": cm,
            "delta": cm - bm,
            "improved_seeds": sum(r["paired_delta"] > 0 for r in rows),
            "worsened_seeds": sum(r["paired_delta"] < 0 for r in rows),
            "same_seeds": sum(r["paired_delta"] == 0 for r in rows),
        }
    }

    Path("day0_full_prefix_replacement_terminal_result_v0.json").write_text(
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
