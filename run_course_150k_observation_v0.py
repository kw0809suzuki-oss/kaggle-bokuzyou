#!/usr/bin/env python3
import gzip
import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make
from cash_return_wheat_v0 import make_candidate

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
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def assets(obs, seat=0):
    farm = obs["farms"][seat]
    crops = {}
    animals = {}
    plants = 0
    animal_count = 0
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
                animal_count += 1
                animals[str(animal)] = animals.get(str(animal), 0) + 1
    return {
        "cash": float(farm.get("money", 0) or 0),
        "land": len(farm.get("unlocked_quadrants", []) or []),
        "hands": len(farm.get("hands", []) or []),
        "plants": plants,
        "crops": crops,
        "animals": animal_count,
        "animal_types": animals,
    }


def make_wheat():
    return make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )


def run(label):
    if label == "teacher":
        model = load_module(ROOT / "opponents" / "seyamalam_self.py", "teacher_150k")
        self_agent = model.agent
    elif label == "independent":
        model = load_module(ROOT / "astra_flow_independent_distilled_v0.py", "independent_150k")
        if hasattr(model, "reset_agent"):
            model.reset_agent()
        self_agent = model.agent
    else:
        raise ValueError(label)

    wheat = make_wheat()
    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    turns = []
    daily = {}
    action_counts = {}
    sell_units = {}

    while not env.done:
        obs0 = plain(shared_obs(env, 0))
        obs1 = plain(shared_obs(env, 1))
        step = int(obs0["step"])
        day = int(obs0["day"])
        a0 = plain(self_agent(obs0))
        a1 = plain(wheat.act(obs1))

        if day not in daily:
            daily[day] = {
                "day": day,
                "start": assets(obs0, 0),
            }

        actor_actions = [a0.get("farmer", ["PASS"]), *(a0.get("hands", []) or [])]
        for act in actor_actions:
            op = act[0] if act else "PASS"
            action_counts[op] = action_counts.get(op, 0) + 1
        for order in a0.get("market", []) or []:
            op = order[0] if order else "UNKNOWN"
            action_counts["MARKET_" + op] = action_counts.get("MARKET_" + op, 0) + 1
            if op == "SELL" and len(order) >= 3:
                item = str(order[1])
                sell_units[item] = sell_units.get(item, 0) + max(0, int(order[2] or 0))

        turns.append({
            "step": step,
            "day": day,
            "hour": int(obs0["hour"]),
            "self": assets(obs0, 0),
            "opponent": assets(obs0, 1),
            "shed": plain(obs0.get("private", {}).get("shed", {})),
            "seeds": plain(obs0.get("private", {}).get("seeds", {})),
            "market": plain(obs0.get("market", {})),
            "self_action": a0,
            "opponent_action": a1,
            "observation": obs0,
        })

        daily[day]["last_pre_action"] = assets(obs0, 0)
        env.step([a0, a1])

    final = plain(env.state[0].observation)
    final_self = assets(final, 0)
    final_opp = assets(final, 1)
    return {
        "label": label,
        "terminal_self": final_self["cash"],
        "terminal_opponent": final_opp["cash"],
        "margin": final_self["cash"] - final_opp["cash"],
        "final_self": final_self,
        "action_counts": action_counts,
        "requested_sell_units": sell_units,
        "daily": [daily[d] for d in sorted(daily)],
        "turns": turns,
    }


def main():
    teacher = run("teacher")
    independent = run("independent")

    mismatch_count = 0
    first_mismatch = None
    for ta, ib in zip(teacher["turns"], independent["turns"]):
        if ta["self_action"] != ib["self_action"]:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "step": ta["step"],
                    "day": ta["day"],
                    "hour": ta["hour"],
                    "teacher_action": ta["self_action"],
                    "independent_action": ib["self_action"],
                    "teacher_self": ta["self"],
                    "independent_self": ib["self"],
                    "teacher_market": ta["market"],
                    "independent_market": ib["market"],
                }

    summary = {
        "schema": "course-150k-observation-v0",
        "seed": SEED,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "teacher_commit": TEACHER_COMMIT,
            "teacher_source": "opponents/seyamalam_self.py",
            "independent_source": "astra_flow_independent_distilled_v0.py",
            "wheat_source": "cash_return_wheat_v0.py",
        },
        "terminal": {
            "teacher": teacher["terminal_self"],
            "independent": independent["terminal_self"],
            "difference_independent_minus_teacher": independent["terminal_self"] - teacher["terminal_self"],
        },
        "action_difference": {
            "count": mismatch_count,
            "first": first_mismatch,
        },
        "teacher_daily": teacher["daily"],
        "independent_daily": independent["daily"],
        "teacher_action_counts": teacher["action_counts"],
        "independent_action_counts": independent["action_counts"],
        "teacher_requested_sell_units": teacher["requested_sell_units"],
        "independent_requested_sell_units": independent["requested_sell_units"],
    }

    Path("course_150k_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with gzip.open("course_150k_full.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump({
            "schema": "course-150k-observation-v0-full",
            "summary": summary,
            "teacher": teacher,
            "independent": independent,
        }, fh, ensure_ascii=False, separators=(",", ":"))

    print("COURSE150K " + json.dumps({
        "teacher": teacher["terminal_self"],
        "independent": independent["terminal_self"],
        "delta": independent["terminal_self"] - teacher["terminal_self"],
        "action_mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
    }, separators=(",", ":")))

    for d in [0, 5, 7, 10, 12, 15, 20, 25, 29]:
        t = teacher["daily"][d]["start"]
        i = independent["daily"][d]["start"]
        print("DAY150K " + json.dumps({
            "day": d,
            "teacher": t,
            "independent": i,
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()

# rerun after restoring cash_return_wheat_v0.py
