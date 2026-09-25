#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SELF_ROOT = ROOT / "selfsrc"
OPP_ROOT = ROOT / "opponents"
sys.path.insert(0, str(SELF_ROOT))
sys.path.insert(0, str(OPP_ROOT))

import seyamalam_v21 as opponent
import wr02_same_tile_plant_deconfliction_v0 as wr02
import strong_origin_body
import strong_origin_early_target_reservation_v0 as reserved_origin

SEEDS = list(range(7001, 7011))


def candidate_agent(obs):
    old = strong_origin_body.strong_origin
    strong_origin_body.strong_origin = reserved_origin
    try:
        return wr02.agent(obs)
    finally:
        strong_origin_body.strong_origin = old


def run_one(model, seed):
    if model == "baseline":
        agent = wr02.agent
    elif model == "candidate":
        agent = candidate_agent
    else:
        raise ValueError(model)

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.run([agent, opponent.agent])
    rewards = [float(s.reward) for s in env.state]
    payload = {
        "model": model,
        "seed": seed,
        "self": rewards[0],
        "opponent": rewards[1],
        "margin": rewards[0] - rewards[1],
        "steps": len(env.steps),
    }
    print(json.dumps(payload, separators=(",", ":")))


def child(model, seed):
    cp = subprocess.run(
        [sys.executable, __file__, "--model", model, "--seed", str(seed)],
        check=True, capture_output=True, text=True,
    )
    line = cp.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "candidate"])
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    if args.model:
        run_one(args.model, args.seed)
        return

    rows = []
    for seed in SEEDS:
        b = child("baseline", seed)
        c = child("candidate", seed)
        row = {
            "seed": seed,
            "baseline_self": b["self"],
            "candidate_self": c["self"],
            "delta_self": c["self"] - b["self"],
            "baseline_margin": b["margin"],
            "candidate_margin": c["margin"],
            "delta_margin": c["margin"] - b["margin"],
        }
        rows.append(row)
        print(json.dumps(row, separators=(",", ":")))

    out = {
        "schema": "nr01-early-target-reservation-official-v0",
        "baseline": "Baseline+WR-02",
        "candidate": "Baseline+WR-02 + Day0-4 target reservation",
        "world": "official kaggle_environments kaggriculture",
        "seeds": SEEDS,
        "summary": {
            "baseline_mean_self": statistics.mean(r["baseline_self"] for r in rows),
            "candidate_mean_self": statistics.mean(r["candidate_self"] for r in rows),
            "delta_mean_self": statistics.mean(r["delta_self"] for r in rows),
            "baseline_median_self": statistics.median(r["baseline_self"] for r in rows),
            "candidate_median_self": statistics.median(r["candidate_self"] for r in rows),
            "baseline_mean_margin": statistics.mean(r["baseline_margin"] for r in rows),
            "candidate_mean_margin": statistics.mean(r["candidate_margin"] for r in rows),
            "delta_mean_margin": statistics.mean(r["delta_margin"] for r in rows),
            "improved": sum(r["delta_self"] > 0 for r in rows),
            "worsened": sum(r["delta_self"] < 0 for r in rows),
            "equal": sum(r["delta_self"] == 0 for r in rows),
            "baseline_min_self": min(r["baseline_self"] for r in rows),
            "baseline_max_self": max(r["baseline_self"] for r in rows),
            "candidate_min_self": min(r["candidate_self"] for r in rows),
            "candidate_max_self": max(r["candidate_self"] for r in rows),
        },
        "rows": rows,
        "boundary": [
            "Only the Strong Origin Day0-4 unit target reservation differs.",
            "WR-02 remains active in both arms.",
            "Each model/seed is executed in a fresh subprocess.",
            "Terminal rewards are read directly from official env.state rewards."
        ],
    }
    Path("nr01_early_target_reservation_official_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(out["summary"], separators=(",", ":")))


if __name__ == "__main__":
    main()
