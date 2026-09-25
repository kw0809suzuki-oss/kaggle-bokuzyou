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
    opponent = load_module(OPP_PATH, f"opponent_{variant}_{seed}")

    if variant == "full_seyamalam":
        seat0 = load_module(OPP_PATH, f"seat0_full_{seed}")
        seat0_agent = seat0.agent
    else:
        current = load_module(SELF_DIR / "whole_flow_control_agent.py", f"current_handoff_{seed}")
        prefix = load_module(OPP_PATH, f"prefix_handoff_{seed}")

        def seat0_agent(obs):
            if obs["day"] == 0:
                _ = current.agent(obs)
                return prefix.agent(obs)
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
        for variant in ("full_seyamalam", "handoff_day1"):
            proc = subprocess.run(
                [sys.executable, __file__, "--child", "--seed", str(seed), "--variant", variant],
                check=True,
                capture_output=True,
                text=True,
            )
            pair[variant] = json.loads(proc.stdout.strip().splitlines()[-1])

        full = pair["full_seyamalam"]["terminal_self"]
        handoff = pair["handoff_day1"]["terminal_self"]
        rows.append({
            "seed": seed,
            "full_seyamalam_terminal_self": full,
            "handoff_day1_terminal_self": handoff,
            "paired_delta_handoff_minus_full": handoff - full,
        })

    full_mean = sum(r["full_seyamalam_terminal_self"] for r in rows) / len(rows)
    handoff_mean = sum(r["handoff_day1_terminal_self"] for r in rows) / len(rows)

    result = {
        "probe": "seyamalam_vs_day1_handoff_terminal_v0",
        "question": "From the same Day0 Seyamalam-generated trajectory, does switching judgment/action generation to Current Self from Day1 change terminal self versus keeping Seyamalam closed-loop?",
        "design": {
            "baseline": "Fresh Seyamalam controls seat0 for the full battle.",
            "candidate": "Fresh Seyamalam controls seat0 on Day0 only; Current Self is also called and discarded during Day0; Current Self controls seat0 from Day1 onward.",
            "opponent": "Fresh independent Seyamalam instance.",
            "process_isolation": "Each seed/variant runs in a fresh Python subprocess.",
            "seeds": SEEDS
        },
        "boundary": "Terminal response only. No intermediate-state interpretation.",
        "rows": rows,
        "summary": {
            "full_seyamalam_absolute_mean_self": full_mean,
            "handoff_day1_absolute_mean_self": handoff_mean,
            "delta_handoff_minus_full": handoff_mean - full_mean,
            "handoff_higher_seeds": sum(r["paired_delta_handoff_minus_full"] > 0 for r in rows),
            "handoff_lower_seeds": sum(r["paired_delta_handoff_minus_full"] < 0 for r in rows),
            "same_seeds": sum(r["paired_delta_handoff_minus_full"] == 0 for r in rows),
        }
    }

    Path("seyamalam_vs_day1_handoff_terminal_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--child", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--variant", choices=["full_seyamalam","handoff_day1"])
    args = p.parse_args()
    if args.child:
        child_main(args.seed, args.variant)
    else:
        parent_main()
