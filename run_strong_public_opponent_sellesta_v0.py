#!/usr/bin/env python3
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ["SEED"])

SELLESTA_COMMIT = "0a5ce7ef83211d6df5824a4e7e17f3d4a40c22e7"
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
INDEPENDENT_COMMIT = "eff2cf1ae6eb03809d215248f9b05b79a2b4dfe9"
V21_SOURCE_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"

AGENT_PATHS = {
    "independent": ROOT / "astra_flow_independent_distilled_v0.py",
    "v21": ROOT / "astra_flow_vendor" / "seyamalam_v21.py",
    "sellesta": ROOT / "opponents" / "sellesta_main_pinned.py",
}
PAIRINGS = [
    ("independent", "sellesta"),
    ("independent", "v21"),
    ("sellesta", "v21"),
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
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def call_agent(mod, obs):
    fn = getattr(mod, "agent")
    try:
        return plain(fn(obs))
    except TypeError:
        return plain(fn(obs, None))


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def farm_summary(obs, seat):
    farm = obs["farms"][seat]
    plants = 0
    animals = 0
    crops = {}
    animal_types = {}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                plants += 1
                crop = str(tile.get("crop"))
                crops[crop] = crops.get(crop, 0) + 1
            animal = tile.get("animal")
            if animal:
                animals += 1
                animal = str(animal)
                animal_types[animal] = animal_types.get(animal, 0) + 1
    return {
        "cash": float(farm.get("money", 0) or 0),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "hands": len(farm.get("hands", []) or []),
        "plants": plants,
        "crops": crops,
        "animals": animals,
        "animal_types": animal_types,
    }


def run_game(a_label, b_label, a_seat):
    game_id = f"{a_label}-vs-{b_label}-a{a_seat}"
    a_mod = load_module(AGENT_PATHS[a_label], f"{game_id}_a_{SEED}".replace("-", "_"))
    b_mod = load_module(AGENT_PATHS[b_label], f"{game_id}_b_{SEED}".replace("-", "_"))

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    seat_mod = [None, None]
    seat_label = [None, None]
    seat_mod[a_seat] = a_mod
    seat_label[a_seat] = a_label
    b_seat = 1 - a_seat
    seat_mod[b_seat] = b_mod
    seat_label[b_seat] = b_label

    turns = []
    while not env.done:
        obs0 = plain(shared_obs(env, 0))
        obs1 = plain(shared_obs(env, 1))
        observations = [obs0, obs1]
        actions = [
            call_agent(seat_mod[0], observations[0]),
            call_agent(seat_mod[1], observations[1]),
        ]
        turns.append({
            "step": int(obs0.get("step", len(turns))),
            "day": int(obs0.get("day", 0)),
            "hour": int(obs0.get("hour", 0)),
            "seat_labels": list(seat_label),
            "observation_seat0": obs0,
            "observation_seat1": obs1,
            "action_seat0": actions[0],
            "action_seat1": actions[1],
        })
        env.step(actions)

    final0 = plain(env.state[0].observation)
    cash0 = float(final0["farms"][0]["money"])
    cash1 = float(final0["farms"][1]["money"])

    a_cash = cash0 if a_seat == 0 else cash1
    b_cash = cash1 if a_seat == 0 else cash0
    result = "win" if a_cash > b_cash else ("loss" if a_cash < b_cash else "tie")

    summary = {
        "game_id": game_id,
        "a_label": a_label,
        "b_label": b_label,
        "a_seat": a_seat,
        "turn_count": len(turns),
        "a_terminal": a_cash,
        "b_terminal": b_cash,
        "a_margin": a_cash - b_cash,
        "a_result": result,
        "seat0_terminal": cash0,
        "seat1_terminal": cash1,
        "final_a": farm_summary(final0, a_seat),
        "final_b": farm_summary(final0, b_seat),
    }

    with gzip.open(f"strong_public_{SEED}_{game_id}.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump({
            "schema": "strong-public-opponent-sellesta-v0-full",
            "seed": SEED,
            "provenance": {
                "official_commit": OFFICIAL_COMMIT,
                "independent_commit": INDEPENDENT_COMMIT,
                "v21_source_commit": V21_SOURCE_COMMIT,
                "sellesta_commit": SELLESTA_COMMIT,
            },
            "summary": summary,
            "turns": turns,
        }, fh, ensure_ascii=False, separators=(",", ":"))

    return summary


def main():
    results = []
    for a_label, b_label in PAIRINGS:
        for a_seat in (0, 1):
            results.append(run_game(a_label, b_label, a_seat))

    out = {
        "schema": "strong-public-opponent-sellesta-v0-summary",
        "seed": SEED,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "independent_commit": INDEPENDENT_COMMIT,
            "v21_source_commit": V21_SOURCE_COMMIT,
            "sellesta_commit": SELLESTA_COMMIT,
        },
        "results": results,
    }
    Path(f"strong_public_{SEED}_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("STRONGPUBLIC " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
