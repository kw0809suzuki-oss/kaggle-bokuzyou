#!/usr/bin/env python3
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

from cash_return_wheat_v0 import make_candidate
from plan_generator_entrance_v0 import bind_official_state

ROOT = Path(__file__).resolve().parent
SEED = 92801101
CANDIDATE_CODE_COMMIT = "5278b25325d119ed71372c137e418d4a17fb37b2"
OPPONENT_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
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


def load_opponent():
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location("seyamalam_v21_course", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def tile_summary(farm):
    plants = Counter()
    animals = Counter()
    yields = Counter()
    locked = 0
    empty = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile is None:
                empty += 1
            elif tile == "LOCKED":
                locked += 1
            elif isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    crop = str(tile.get("crop"))
                    plants[crop] += 1
                    yields[crop] += float(tile.get("yield_units", 0) or 0)
                elif "animal" in tile:
                    animals[str(tile.get("animal"))] += 1
    return {
        "plants": dict(plants),
        "plants_total": sum(plants.values()),
        "animals": dict(animals),
        "animals_total": sum(animals.values()),
        "yield_units": dict(yields),
        "yield_units_total": sum(yields.values()),
        "locked_tiles": locked,
        "empty_tiles": empty,
    }


def side_snapshot(obs, seat):
    raw = bind_official_state(obs).raw()
    farm = raw["farms"][seat]
    private = raw["private"]
    t = tile_summary(farm)
    shed = {str(k): float(v) for k, v in (private.get("shed", {}) or {}).items() if v}
    seeds = {str(k): float(v) for k, v in (private.get("seeds", {}) or {}).items() if v}
    carried = Counter()
    for inv in private.get("inventories", []) or []:
        for k, v in (inv or {}).items():
            if v:
                carried[str(k)] += float(v)
    return {
        "day": int(raw["day"]),
        "hour": int(raw["hour"]),
        "cash": float(farm.get("money", 0) or 0),
        "unlocked_quadrants": len(farm.get("unlocked_quadrants", []) or []),
        "hands": len(farm.get("hands", []) or []),
        "plants": t["plants"],
        "plants_total": t["plants_total"],
        "animals": t["animals"],
        "animals_total": t["animals_total"],
        "yield_units": t["yield_units"],
        "yield_units_total": t["yield_units_total"],
        "shed": shed,
        "shed_total": sum(shed.values()),
        "seeds": seeds,
        "seeds_total": sum(seeds.values()),
        "carried": dict(carried),
        "carried_total": sum(carried.values()),
        "locked_tiles": t["locked_tiles"],
        "empty_tiles": t["empty_tiles"],
    }


def action_stats():
    return {
        "market_ops": Counter(),
        "buy_seed": Counter(),
        "buy_animal": Counter(),
        "buy_product": Counter(),
        "sell_requested": Counter(),
        "hire": 0,
        "buy_land": 0,
        "unit_ops": Counter(),
    }


def add_action_stats(stats, bundle):
    for unit in [bundle.get("farmer"), *(bundle.get("hands", []) or [])]:
        if isinstance(unit, list) and unit:
            stats["unit_ops"][str(unit[0])] += 1

    for op in bundle.get("market", []) or []:
        if not isinstance(op, list) or not op:
            continue
        typ = str(op[0])
        stats["market_ops"][typ] += 1
        if typ == "HIRE":
            stats["hire"] += 1
        elif typ == "BUY_LAND":
            stats["buy_land"] += 1
        elif len(op) >= 2:
            item = str(op[1])
            qty = int(op[2]) if len(op) >= 3 and isinstance(op[2], (int, float)) else 1
            if typ == "BUY_SEED":
                stats["buy_seed"][item] += qty
            elif typ == "BUY_ANIMAL":
                stats["buy_animal"][item] += qty
            elif typ == "BUY_PRODUCT":
                stats["buy_product"][item] += qty
            elif typ == "SELL":
                stats["sell_requested"][item] += qty


def simplify_stats(stats):
    return {
        "market_ops": dict(stats["market_ops"]),
        "buy_seed": dict(stats["buy_seed"]),
        "buy_animal": dict(stats["buy_animal"]),
        "buy_product": dict(stats["buy_product"]),
        "sell_requested": dict(stats["sell_requested"]),
        "hire": int(stats["hire"]),
        "buy_land": int(stats["buy_land"]),
        "unit_ops": dict(stats["unit_ops"]),
    }


def main():
    candidate = make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )
    opponent = load_opponent()

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    daily = defaultdict(lambda: {
        "self_first": None,
        "self_last": None,
        "opponent_first": None,
        "opponent_last": None,
        "opponent_max_hands": 0,
        "opponent_min_cash": None,
        "opponent_max_cash": None,
        "opponent_stats": action_stats(),
    })
    land_events = []
    cash_thresholds = {}
    thresholds = [5000, 10000, 20000, 50000, 75000, 100000, 125000, 150000]

    turn = 0
    while not env.done:
        obs0 = shared_obs(env, 0)
        obs1 = shared_obs(env, 1)
        s0 = side_snapshot(obs0, 0)
        s1 = side_snapshot(obs1, 1)
        day = s1["day"]

        d = daily[day]
        if d["self_first"] is None:
            d["self_first"] = s0
            d["opponent_first"] = s1
        d["self_last"] = s0
        d["opponent_last"] = s1
        d["opponent_max_hands"] = max(d["opponent_max_hands"], s1["hands"])
        d["opponent_min_cash"] = s1["cash"] if d["opponent_min_cash"] is None else min(d["opponent_min_cash"], s1["cash"])
        d["opponent_max_cash"] = s1["cash"] if d["opponent_max_cash"] is None else max(d["opponent_max_cash"], s1["cash"])

        for threshold in thresholds:
            if threshold not in cash_thresholds and s1["cash"] >= threshold:
                cash_thresholds[threshold] = {
                    "turn": turn,
                    "day": s1["day"],
                    "hour": s1["hour"],
                    "cash": s1["cash"],
                }

        a0 = plain(candidate.act(obs0))
        a1 = plain(opponent.agent(obs1))
        add_action_stats(d["opponent_stats"], a1)

        for op in a1.get("market", []) or []:
            if isinstance(op, list) and op and op[0] == "BUY_LAND":
                land_events.append({
                    "turn": turn,
                    "day": s1["day"],
                    "hour": s1["hour"],
                    "cash_before": s1["cash"],
                    "unlocked_before": s1["unlocked_quadrants"],
                })

        env.step([a0, a1])
        turn += 1

    terminal_raw = bind_official_state(env.state[0].observation).raw()
    terminal_self = float(terminal_raw["farms"][0]["money"])
    terminal_opponent = float(terminal_raw["farms"][1]["money"])

    rows = []
    previous_terminal_cash = None
    for day in sorted(daily):
        d = daily[day]
        start = d["opponent_first"]["cash"]
        # The next day's first cash is the true end-of-day cash after the final action.
        next_day = day + 1
        if next_day in daily:
            end_cash = daily[next_day]["opponent_first"]["cash"]
        else:
            end_cash = terminal_opponent
        rows.append({
            "day": day,
            "self_start_cash": d["self_first"]["cash"],
            "opponent_start_cash": start,
            "opponent_end_cash": end_cash,
            "opponent_net_cash": end_cash - start,
            "opponent_min_cash": d["opponent_min_cash"],
            "opponent_max_cash": d["opponent_max_cash"],
            "opponent_unlocked_quadrants_end_pre": d["opponent_last"]["unlocked_quadrants"],
            "opponent_max_hands": d["opponent_max_hands"],
            "opponent_plants_end_pre": d["opponent_last"]["plants"],
            "opponent_plants_total_end_pre": d["opponent_last"]["plants_total"],
            "opponent_animals_end_pre": d["opponent_last"]["animals"],
            "opponent_animals_total_end_pre": d["opponent_last"]["animals_total"],
            "opponent_yield_units_end_pre": d["opponent_last"]["yield_units"],
            "opponent_shed_end_pre": d["opponent_last"]["shed"],
            "opponent_action_stats": simplify_stats(d["opponent_stats"]),
        })

    result = {
        "schema": "opponent-course-observation-v0",
        "seed": SEED,
        "provenance": {
            "candidate_code_commit": CANDIDATE_CODE_COMMIT,
            "opponent_commit": OPPONENT_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
        },
        "terminal": {
            "self": terminal_self,
            "opponent": terminal_opponent,
            "margin": terminal_self - terminal_opponent,
        },
        "opponent_cash_thresholds": {str(k): v for k, v in cash_thresholds.items()},
        "opponent_land_events": land_events,
        "daily": rows,
    }

    Path("opponent_course_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TERMINAL " + json.dumps(result["terminal"], separators=(",", ":")))
    print("LAND " + json.dumps(land_events, separators=(",", ":")))
    print("THRESHOLDS " + json.dumps(result["opponent_cash_thresholds"], separators=(",", ":")))
    for row in rows:
        print("DAY " + json.dumps(row, separators=(",", ":")))


if __name__ == "__main__":
    main()
