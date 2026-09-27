#!/usr/bin/env python3
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent

OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
FROZEN_COMMIT = "b29f8b849e65ab56790f0e2352df0863b736752a"
ASTRA_V0_COMMIT = "55c043e034c68568a6398b11cc126e7802e5bb15"
ASTRA_V1_COMMIT = "35baab3fa5720e897623bb3b959926b7c9d8dbbd"
OPPONENT_SOURCE_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"

MODELS = (
    ("frozen", ROOT / "relationship_surface_body_v0.py"),
    ("v1", ROOT / "astra_flow_terminal_model_v1.py"),
    ("v0", ROOT / "astra_flow_terminal_model_v0.py"),
)
OPPONENT_PATH = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def run_one(seed: int, label: str, model_path: Path):
    model = load_module(model_path, f"three_model_{label}_{seed}")
    opponent = load_module(OPPONENT_PATH, f"three_model_opponent_{label}_{seed}")

    if hasattr(model, "reset_agent"):
        model.reset_agent()
    if hasattr(opponent, "reset_agent"):
        opponent.reset_agent()

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    turns = []
    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)

        action0 = plain(model.agent(obs0))
        action1 = plain(opponent.agent(obs1))

        turns.append({
            "step": int(obs0.get("step", len(turns))),
            "observation": plain(obs0),
            "self_action": action0,
            "opponent_action": action1,
        })
        env.step([action0, action1])

    final0 = plain(env.state[0].observation)
    final1 = plain(env.state[1].observation)
    self_terminal = float(final0["farms"][0]["money"])
    opponent_terminal = float(final0["farms"][1]["money"])

    return {
        "label": label,
        "model_file": model_path.name,
        "turn_count": len(turns),
        "terminal_self": self_terminal,
        "terminal_opponent": opponent_terminal,
        "margin": self_terminal - opponent_terminal,
        "turns": turns,
        "final_self_observation": final0,
        "final_opponent_observation": final1,
    }


def main():
    seed = int(os.environ["SEED"])

    runs = []
    for label, model_path in MODELS:
        runs.append(run_one(seed, label, model_path))

    full = {
        "schema": "astra-flow-three-model-observation-v0",
        "seed": seed,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "frozen_commit": FROZEN_COMMIT,
            "astra_v0_commit": ASTRA_V0_COMMIT,
            "astra_v1_commit": ASTRA_V1_COMMIT,
            "opponent_source_commit": OPPONENT_SOURCE_COMMIT,
            "opponent_file": "astra_flow_vendor/seyamalam_v21.py",
            "seat": "self seat 0",
            "github_sha": os.environ.get("GITHUB_SHA"),
        },
        "runs": runs,
    }

    summary = {
        "schema": "astra-flow-three-model-observation-v0-summary",
        "seed": seed,
        "provenance": full["provenance"],
        "results": [
            {
                "label": r["label"],
                "model_file": r["model_file"],
                "turn_count": r["turn_count"],
                "terminal_self": r["terminal_self"],
                "terminal_opponent": r["terminal_opponent"],
                "margin": r["margin"],
            }
            for r in runs
        ],
    }

    full_path = Path(f"astra_flow_three_model_seed{seed}.json.gz")
    with gzip.open(full_path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(full, fh, ensure_ascii=False, separators=(",", ":"))

    summary_path = Path(f"astra_flow_three_model_seed{seed}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("THREEMODEL " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
