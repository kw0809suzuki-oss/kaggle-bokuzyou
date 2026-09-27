#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ["SEED"])

MODELS = {
    "baseline": ROOT / "astra_flow_independent_distilled_v0.py",
    "candidate": ROOT / "independent_day0_melon12_v0.py",
}
SELLESTA = ROOT / "opponents" / "sellesta_main_pinned.py"

_ORIG_COMMIT_UNIT = kg._commit_unit
_ORIG_APPLY_UNIT_ACTION = kg._apply_unit_action
_CTX = {
    "step": None,
    "farm_ids": {},
    "private_ids": {},
    "events": [],
}


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


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def crop_count(farm, crop):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
                n += 1
    return n


def _inventory_total(private, item):
    total = 0
    for inv in private.get("inventories", []) or []:
        total += int(inv.get(item, 0) or 0)
    return total


def _logged_commit_unit(op, item, price, farm, private, market, shed_capacity=100):
    seat = _CTX["farm_ids"].get(id(farm))
    ok = _ORIG_COMMIT_UNIT(op, item, price, farm, private, market, shed_capacity)
    if ok and seat is not None and op == "SELL" and item == "MELON":
        _CTX["events"].append({
            "kind": "MELON_SELL",
            "step": int(_CTX["step"]),
            "seat": int(seat),
            "units": 1,
            "cash": float(price),
        })
    return ok


def _logged_apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
    seat = _CTX["private_ids"].get(id(private))
    before = _inventory_total(private, "MELON") if seat is not None else 0
    _ORIG_APPLY_UNIT_ACTION(
        farm, private, idx, action, board_size, day, turns_per_day, shed_capacity
    )
    if (
        seat is not None
        and isinstance(action, list)
        and action
        and action[0] == "HARVEST"
    ):
        after = _inventory_total(private, "MELON")
        delta = after - before
        if delta > 0:
            _CTX["events"].append({
                "kind": "MELON_HARVEST",
                "step": int(_CTX["step"]),
                "seat": int(seat),
                "units": int(delta),
            })


kg._commit_unit = _logged_commit_unit
kg._apply_unit_action = _logged_apply_unit_action


def configure_recorder(env, step=None):
    farms = env.state[0].observation.farms
    privates = [env.state[i].observation.private for i in range(2)]
    _CTX["farm_ids"] = {id(farms[i]): i for i in range(2)}
    _CTX["private_ids"] = {id(privates[i]): i for i in range(2)}
    _CTX["step"] = step
    _CTX["events"] = []


def reachability(a_seat):
    subject = load_module(
        MODELS["candidate"], f"reach_candidate_{SEED}_{a_seat}"
    )
    opponent = load_module(
        SELLESTA, f"reach_sellesta_{SEED}_{a_seat}"
    )

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    mods = [None, None]
    mods[a_seat] = subject
    mods[1 - a_seat] = opponent

    while True:
        obs0 = plain(shared_obs(env, 0))
        if int(obs0.get("day", 0)) >= 1:
            break
        actions = [
            plain(mods[0].agent(plain(shared_obs(env, 0)))),
            plain(mods[1].agent(plain(shared_obs(env, 1)))),
        ]
        env.step(actions)

    final = plain(env.state[0].observation)
    farm = final["farms"][a_seat]
    return {
        "candidate_seat": a_seat,
        "day": int(final["day"]),
        "hour": int(final["hour"]),
        "melon_plants": crop_count(farm, "MELON"),
        "wheat_plants": crop_count(farm, "WHEAT"),
        "cash": float(farm["money"]),
    }


def run_game(label, a_seat):
    subject = load_module(
        MODELS[label], f"{label}_{SEED}_{a_seat}"
    )
    opponent = load_module(
        SELLESTA, f"sellesta_{label}_{SEED}_{a_seat}"
    )

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)
    configure_recorder(env)

    mods = [None, None]
    mods[a_seat] = subject
    mods[1 - a_seat] = opponent

    day0_snapshot = None
    turns = 0

    while not env.done:
        obs0 = plain(shared_obs(env, 0))
        _CTX["step"] = int(obs0.get("step", turns))
        actions = [
            plain(mods[0].agent(plain(shared_obs(env, 0)))),
            plain(mods[1].agent(plain(shared_obs(env, 1)))),
        ]
        env.step(actions)
        turns += 1

        if day0_snapshot is None:
            post = plain(env.state[0].observation)
            if int(post.get("day", 0)) == 1 and int(post.get("hour", 0)) == 0:
                farm = post["farms"][a_seat]
                day0_snapshot = {
                    "melon_plants": crop_count(farm, "MELON"),
                    "wheat_plants": crop_count(farm, "WHEAT"),
                    "cash": float(farm["money"]),
                }

    final = plain(env.state[0].observation)
    self_cash = float(final["farms"][a_seat]["money"])
    opp_cash = float(final["farms"][1 - a_seat]["money"])

    events = [e for e in _CTX["events"] if e["seat"] == a_seat]
    harvest = [e for e in events if e["kind"] == "MELON_HARVEST"]
    sells = [e for e in events if e["kind"] == "MELON_SELL"]
    sell_units = sum(e["units"] for e in sells)
    sell_revenue = sum(e["cash"] for e in sells)

    return {
        "label": label,
        "seed": SEED,
        "seat": a_seat,
        "turns": turns,
        "terminal_self": self_cash,
        "terminal_sellesta": opp_cash,
        "margin": self_cash - opp_cash,
        "day0_end": day0_snapshot,
        "melon_harvest_units": sum(e["units"] for e in harvest),
        "melon_sell_units": sell_units,
        "melon_sell_revenue": sell_revenue,
        "melon_avg_sell_price": (sell_revenue / sell_units) if sell_units else None,
        "first_melon_sell_step": min((e["step"] for e in sells), default=None),
        "last_melon_sell_step": max((e["step"] for e in sells), default=None),
    }


def main():
    reach = [reachability(0), reachability(1)]
    reached = all(r["melon_plants"] == 12 for r in reach)

    out = {
        "schema": "independent-day0-melon12-v0",
        "seed": SEED,
        "reachability": reach,
        "reachability_pass": reached,
        "results": [],
        "paired": [],
    }

    if not reached:
        Path(f"melon12_seed{SEED}_summary.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("MELON12 " + json.dumps(out, separators=(",", ":")))
        raise SystemExit("Reachability failed: candidate did not produce 12 MELON plants on Day0")

    results = []
    for seat in (0, 1):
        results.append(run_game("baseline", seat))
        results.append(run_game("candidate", seat))
    out["results"] = results

    for seat in (0, 1):
        base = next(r for r in results if r["label"] == "baseline" and r["seat"] == seat)
        cand = next(r for r in results if r["label"] == "candidate" and r["seat"] == seat)
        out["paired"].append({
            "seat": seat,
            "terminal_self_delta": cand["terminal_self"] - base["terminal_self"],
            "margin_delta": cand["margin"] - base["margin"],
            "melon_harvest_delta": cand["melon_harvest_units"] - base["melon_harvest_units"],
            "melon_sell_units_delta": cand["melon_sell_units"] - base["melon_sell_units"],
            "melon_sell_revenue_delta": cand["melon_sell_revenue"] - base["melon_sell_revenue"],
        })

    Path(f"melon12_seed{SEED}_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("MELON12 " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
