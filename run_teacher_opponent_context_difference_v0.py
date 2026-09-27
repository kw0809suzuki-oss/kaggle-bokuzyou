#!/usr/bin/env python3
import importlib.util
import json
from collections import Counter
from pathlib import Path

from kaggle_environments import make

from seyamalam_teacher_body_v0 import make_teacher_body
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_wheat():
    return make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def farm_summary(obs, seat):
    farm = obs["farms"][seat]
    plants = Counter()
    animals = Counter()
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    plants[str(tile.get("crop"))] += 1
                if tile.get("animal"):
                    animals[str(tile.get("animal"))] += 1
    return {
        "cash": float(farm.get("money", 0) or 0),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "hands": len(farm.get("hands", []) or []),
        "plants": dict(plants),
        "animals": dict(animals),
    }


def market_summary(obs):
    market = obs.get("market", {}) or {}
    return {
        "prices": plain(market.get("prices", {}) or {}),
        "inventory": plain(market.get("inventory", {}) or {}),
    }


def first_diff(a, b):
    if a == b:
        return None
    return {"a": a, "b": b}


def main():
    self_mod_a = load_module(ROOT / "opponents" / "teacher_self_a.py", "teacher_self_a")
    self_mod_b = load_module(ROOT / "opponents" / "teacher_self_b.py", "teacher_self_b")
    opp_teacher_mod = load_module(ROOT / "opponents" / "teacher_opp_b.py", "teacher_opp_b")

    self_a = make_teacher_body(self_mod_a)  # teacher vs WHEAT
    self_b = make_teacher_body(self_mod_b)  # teacher vs teacher
    opp_a = make_wheat()
    opp_b = make_teacher_body(opp_teacher_mod)

    env_a = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env_b = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env_a.reset(num_agents=2)
    env_b.reset(num_agents=2)

    first_market_diff = None
    first_self_state_diff = None
    first_self_action_diff = None
    first_opponent_action_diff = None
    daily = {}

    turn = 0
    while not env_a.done and not env_b.done:
        obs_a0 = shared_obs(env_a, 0)
        obs_a1 = shared_obs(env_a, 1)
        obs_b0 = shared_obs(env_b, 0)
        obs_b1 = shared_obs(env_b, 1)

        clock = {"turn": turn, "day": int(obs_a0["day"]), "hour": int(obs_a0["hour"])}

        ma = market_summary(obs_a0)
        mb = market_summary(obs_b0)
        sa = farm_summary(obs_a0, 0)
        sb = farm_summary(obs_b0, 0)

        if first_market_diff is None and ma != mb:
            first_market_diff = {**clock, "teacher_vs_wheat": ma, "teacher_vs_teacher": mb}
        if first_self_state_diff is None and sa != sb:
            first_self_state_diff = {**clock, "teacher_vs_wheat": sa, "teacher_vs_teacher": sb}

        a_self = plain(self_a.act(obs_a0))
        b_self = plain(self_b.act(obs_b0))
        a_opp = plain(opp_a.act(obs_a1))
        b_opp = plain(opp_b.act(obs_b1))

        if first_self_action_diff is None and a_self != b_self:
            first_self_action_diff = {
                **clock,
                "teacher_vs_wheat": a_self,
                "teacher_vs_teacher": b_self,
                "state_teacher_vs_wheat": sa,
                "state_teacher_vs_teacher": sb,
            }
        if first_opponent_action_diff is None and a_opp != b_opp:
            first_opponent_action_diff = {
                **clock,
                "wheat_opponent": a_opp,
                "teacher_opponent": b_opp,
            }

        day = int(obs_a0["day"])
        if day not in daily:
            daily[day] = {
                "day": day,
                "teacher_vs_wheat_start": sa,
                "teacher_vs_teacher_start": sb,
            }
        daily[day]["teacher_vs_wheat_last_pre"] = sa
        daily[day]["teacher_vs_teacher_last_pre"] = sb

        env_a.step([a_self, a_opp])
        env_b.step([b_self, b_opp])
        turn += 1

    final_a = env_a.state[0].observation
    final_b = env_b.state[0].observation

    result = {
        "schema": "teacher-opponent-context-difference-v0",
        "seed": SEED,
        "provenance": {
            "teacher_commit": TEACHER_COMMIT,
            "wheat_commit": WHEAT_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
        },
        "terminal": {
            "teacher_vs_wheat": float(final_a["farms"][0]["money"]),
            "teacher_vs_teacher": float(final_b["farms"][0]["money"]),
            "difference": float(final_a["farms"][0]["money"]) - float(final_b["farms"][0]["money"]),
        },
        "first_opponent_action_diff": first_opponent_action_diff,
        "first_market_diff": first_market_diff,
        "first_self_state_diff": first_self_state_diff,
        "first_self_action_diff": first_self_action_diff,
        "daily": [daily[d] for d in sorted(daily)],
    }

    Path("teacher_opponent_context_difference_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TERMINAL " + json.dumps(result["terminal"], separators=(",", ":")))
    print("FIRST_OPP_ACTION " + json.dumps(first_opponent_action_diff, separators=(",", ":")))
    print("FIRST_MARKET " + json.dumps(first_market_diff, separators=(",", ":")))
    print("FIRST_SELF_STATE " + json.dumps(first_self_state_diff, separators=(",", ":")))
    print("FIRST_SELF_ACTION " + json.dumps(first_self_action_diff, separators=(",", ":")))
    for d in [0,1,3,5,7,10,12,15,20,25,29]:
        print("DAY " + json.dumps(daily[d], separators=(",", ":")))


if __name__ == "__main__":
    main()
