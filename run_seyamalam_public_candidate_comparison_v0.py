#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEED = 92801101
SOURCE_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"

CANDIDATES = [
    ("V7_PUBLIC_V18", "candidate_v7_public_v18.py"),
    ("V8_MARKET_ORDER", "candidate_v8_market_order.py"),
    ("V19_WOOL_FLOOR", "candidate_v19_wool_floor.py"),
    ("V20_LATE_ABSTAIN", "candidate_v20_late_abstain.py"),
    ("V21_CAPITAL_LATCH", "candidate_v21_capital_latch.py"),
]


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    raise TypeError(type(x).__name__)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def run_one(label, candidate_file):
    candidate = load_module(
        ROOT / "teacher_candidates" / candidate_file,
        f"self_{label.lower()}",
    )
    opponent = load_module(
        ROOT / "teacher_candidates" / "main_v21.py",
        f"opp_{label.lower()}",
    )

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    daily = {}
    turn = 0
    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)

        day = int(obs0["day"])
        if day not in daily:
            daily[day] = {
                "day": day,
                "self_cash": float(obs0["farms"][0]["money"]),
                "opponent_cash": float(obs1["farms"][1]["money"]),
            }

        a0 = plain(candidate.agent(obs0))
        a1 = plain(opponent.agent(obs1))
        env.step([a0, a1])
        turn += 1

    final = env.state[0].observation
    return {
        "label": label,
        "candidate_file": candidate_file,
        "self_terminal": float(final["farms"][0]["money"]),
        "opponent_terminal": float(final["farms"][1]["money"]),
        "margin": float(final["farms"][0]["money"]) - float(final["farms"][1]["money"]),
        "turns": turn,
        "daily_start_cash": [daily[d] for d in sorted(daily)],
    }


def main():
    rows = [run_one(label, fname) for label, fname in CANDIDATES]
    rows.sort(key=lambda x: x["self_terminal"], reverse=True)

    result = {
        "schema": "seyamalam-public-candidate-comparison-v0",
        "seed": SEED,
        "provenance": {
            "source_commit": SOURCE_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
            "opponent": "main.py / current V21 at same pinned source commit",
        },
        "results": rows,
    }

    Path("seyamalam_public_candidate_comparison_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for row in rows:
        print("RESULT " + json.dumps({
            "label": row["label"],
            "self": row["self_terminal"],
            "opponent": row["opponent_terminal"],
            "margin": row["margin"],
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
