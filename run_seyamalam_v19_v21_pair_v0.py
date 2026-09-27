#!/usr/bin/env python3
import importlib.util
import json
import os
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SOURCE_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"


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


def run_match(seed, self_file, tag):
    self_mod = load_module(ROOT / "teacher_candidates" / self_file, f"self_{tag}_{seed}")
    opp_mod = load_module(ROOT / "teacher_candidates" / "main_v21.py", f"opp_{tag}_{seed}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)
        env.step([plain(self_mod.agent(obs0)), plain(opp_mod.agent(obs1))])

    final = env.state[0].observation
    return {
        "self_terminal": float(final["farms"][0]["money"]),
        "opponent_terminal": float(final["farms"][1]["money"]),
        "margin": float(final["farms"][0]["money"]) - float(final["farms"][1]["money"]),
    }


def main():
    seed = int(os.environ["SEED"])

    v19 = run_match(seed, "candidate_v19_wool_floor.py", "v19")
    v21 = run_match(seed, "candidate_v21_capital_latch.py", "v21")

    result = {
        "schema": "seyamalam-v19-v21-paired-seed-v0",
        "seed": seed,
        "provenance": {
            "source_commit": SOURCE_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
            "opponent": "main.py V21 from same pinned source commit",
        },
        "v19": v19,
        "v21": v21,
        "paired_self_delta_v19_minus_v21": v19["self_terminal"] - v21["self_terminal"],
    }

    out = Path(f"seyamalam_v19_v21_pair_seed{seed}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PAIR " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
