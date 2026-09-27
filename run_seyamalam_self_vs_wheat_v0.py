#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

from kaggle_environments import make
from cash_return_wheat_v0 import make_candidate

ROOT = Path(__file__).resolve().parent
SEED = 92801101
TEACHER_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
WHEAT_COMMIT = "5278b25325d119ed71372c137e418d4a17fb37b2"
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


def load_teacher():
    path = ROOT / "opponents" / "seyamalam_self.py"
    spec = importlib.util.spec_from_file_location("seyamalam_self_vs_wheat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def main():
    teacher = load_teacher()
    wheat = make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    daily = {}
    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)
        day = int(obs0["day"])
        if day not in daily:
            daily[day] = {
                "teacher_start_cash": float(obs0["farms"][0]["money"]),
                "wheat_start_cash": float(obs1["farms"][1]["money"]),
            }
        a0 = plain(teacher.agent(obs0))
        a1 = plain(wheat.act(obs1))
        env.step([a0, a1])

    final = env.state[0].observation
    teacher_terminal = float(final["farms"][0]["money"])
    wheat_terminal = float(final["farms"][1]["money"])

    result = {
        "schema": "seyamalam-self-vs-wheat-v0",
        "seed": SEED,
        "provenance": {
            "teacher_commit": TEACHER_COMMIT,
            "wheat_commit": WHEAT_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
        },
        "terminal": {
            "teacher_self": teacher_terminal,
            "wheat_opponent": wheat_terminal,
            "margin": teacher_terminal - wheat_terminal,
        },
        "daily_start_cash": [{"day": d, **daily[d]} for d in sorted(daily)],
    }

    Path("seyamalam_self_vs_wheat_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("TERMINAL " + json.dumps(result["terminal"], separators=(",", ":")))
    for d in [0,5,7,10,12,15,20,25,29]:
        if d in daily:
            print("DAY " + json.dumps({"day": d, **daily[d]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
