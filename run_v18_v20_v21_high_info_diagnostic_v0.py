#!/usr/bin/env python3
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
V18_COMMIT = "69c64aebfba7cf54287fb9a58f06ce1b0eff06e4"
V20_V21_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"


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


def animal_counts(farm):
    out = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal"):
                out[str(tile.get("animal"))] += 1
    return dict(out)


def public_snapshot(obs):
    f0 = obs["farms"][0]
    f1 = obs["farms"][1]
    return {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "self_cash": float(f0.get("money", 0) or 0),
        "opponent_cash": float(f1.get("money", 0) or 0),
        "opponent_minus_self_cash": float(f1.get("money", 0) or 0) - float(f0.get("money", 0) or 0),
        "self_animals": animal_counts(f0),
        "opponent_animals": animal_counts(f1),
        "self_hands": len(f0.get("hands", []) or []),
        "opponent_hands": len(f1.get("hands", []) or []),
    }


def run_match(seed, self_file, tag):
    self_mod = load_module(ROOT / "teacher_candidates" / self_file, f"self_{tag}_{seed}")
    opp_mod = load_module(ROOT / "teacher_candidates" / "main_v21.py", f"opp_{tag}_{seed}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    step577 = None
    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)
        step = int(obs0.get("step", 0) or 0)
        if step == 577 and step577 is None:
            step577 = public_snapshot(obs0)
        env.step([plain(self_mod.agent(obs0)), plain(opp_mod.agent(obs1))])

    final = env.state[0].observation
    return {
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "margin": float(final["farms"][0]["money"]) - float(final["farms"][1]["money"]),
        "step577": step577,
    }


def main():
    seed = int(os.environ["SEED"])

    v18 = run_match(seed, "v18_release_main.py", "v18")
    v20 = run_match(seed, "candidate_v20_late_abstain.py", "v20")
    v21 = run_match(seed, "candidate_v21_capital_latch.py", "v21")

    result = {
        "schema": "v18-v20-v21-high-info-diagnostic-v0",
        "seed": seed,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "v18_commit": V18_COMMIT,
            "v20_v21_commit": V20_V21_COMMIT,
            "opponent": "main.py V21",
            "seat": "self seat 0",
        },
        "v18": v18,
        "v20": v20,
        "v21": v21,
        "delta_v20_minus_v18": v20["terminal_self"] - v18["terminal_self"],
        "delta_v21_minus_v18": v21["terminal_self"] - v18["terminal_self"],
    }

    out = Path(f"v18_v20_v21_high_info_seed{seed}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("HIGHINFO " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
