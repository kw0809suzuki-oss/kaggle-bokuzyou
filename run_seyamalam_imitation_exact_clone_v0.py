#!/usr/bin/env python3
import importlib.util
import json
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEED = 92801101
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
TEACHER_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"


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


def side_snapshot(obs, seat):
    farm = obs["farms"][seat]
    plants = Counter()
    animals = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    plants[str(tile.get("crop"))] += 1
                if "animal" in tile and tile.get("animal"):
                    animals[str(tile.get("animal"))] += 1
    return {
        "day": int(obs["day"]),
        "hour": int(obs["hour"]),
        "cash": float(farm.get("money", 0) or 0),
        "hands": len(farm.get("hands", []) or []),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "plants": dict(plants),
        "plants_total": int(sum(plants.values())),
        "animals": dict(animals),
        "animals_total": int(sum(animals.values())),
    }


def main():
    teacher_self = load_module(ROOT / "opponents" / "seyamalam_self.py", "seyamalam_self_clone")
    teacher_opp = load_module(ROOT / "opponents" / "seyamalam_opp.py", "seyamalam_opponent_clone")

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    daily = {}
    thresholds = [5000, 10000, 20000, 50000, 75000, 100000, 125000, 150000]
    self_thresholds = {}
    opp_thresholds = {}

    turn = 0
    while not env.done:
        obs0 = env._Environment__get_shared_state(0)["observation"]
        obs1 = env._Environment__get_shared_state(1)["observation"]

        s0 = side_snapshot(obs0, 0)
        s1 = side_snapshot(obs1, 1)

        day = s0["day"]
        if day not in daily:
            daily[day] = {"self_start": s0, "opponent_start": s1}
        daily[day]["self_last_pre"] = s0
        daily[day]["opponent_last_pre"] = s1

        for t in thresholds:
            if t not in self_thresholds and s0["cash"] >= t:
                self_thresholds[t] = {"turn": turn, "day": s0["day"], "hour": s0["hour"], "cash": s0["cash"]}
            if t not in opp_thresholds and s1["cash"] >= t:
                opp_thresholds[t] = {"turn": turn, "day": s1["day"], "hour": s1["hour"], "cash": s1["cash"]}

        a0 = plain(teacher_self.agent(obs0))
        a1 = plain(teacher_opp.agent(obs1))
        env.step([a0, a1])
        turn += 1

    final_obs0 = env.state[0].observation
    final_self = float(final_obs0["farms"][0]["money"])
    final_opp = float(final_obs0["farms"][1]["money"])

    rows = []
    for day in sorted(daily):
        row = daily[day]
        next_day = day + 1
        self_end = daily[next_day]["self_start"]["cash"] if next_day in daily else final_self
        opp_end = daily[next_day]["opponent_start"]["cash"] if next_day in daily else final_opp
        rows.append({
            "day": day,
            "self_start_cash": row["self_start"]["cash"],
            "self_end_cash": self_end,
            "self_end_pre": row["self_last_pre"],
            "opponent_start_cash": row["opponent_start"]["cash"],
            "opponent_end_cash": opp_end,
            "opponent_end_pre": row["opponent_last_pre"],
        })

    result = {
        "schema": "seyamalam-imitation-exact-clone-v0",
        "seed": SEED,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "teacher_commit": TEACHER_COMMIT,
            "self_agent": "exact pinned Seyamalam source loaded independently",
            "opponent_agent": "exact pinned Seyamalam source loaded independently",
        },
        "terminal": {
            "self": final_self,
            "opponent": final_opp,
            "margin": final_self - final_opp,
        },
        "self_cash_thresholds": {str(k): v for k, v in self_thresholds.items()},
        "opponent_cash_thresholds": {str(k): v for k, v in opp_thresholds.items()},
        "daily": rows,
    }

    Path("seyamalam_imitation_exact_clone_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("TERMINAL " + json.dumps(result["terminal"], separators=(",", ":")))
    print("SELF_THRESHOLDS " + json.dumps(result["self_cash_thresholds"], separators=(",", ":")))
    for row in rows:
        if row["day"] in [0, 5, 7, 10, 12, 15, 20, 25, 26, 29]:
            print("DAY " + json.dumps(row, separators=(",", ":")))


if __name__ == "__main__":
    main()
