#!/usr/bin/env python3
import importlib
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
V18_COMMIT = "69c64aebfba7cf54287fb9a58f06ce1b0eff06e4"
V19_V21_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
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


def get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@contextmanager
def capture_market_ledger():
    engine = importlib.import_module("kaggle_environments.envs.kaggriculture.kaggriculture")
    original_process = engine._process_market
    original_commit = engine._commit_unit
    events = []
    current = {}

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        player = current.get("players", {}).get(id(farm))
        result = original_commit(op, item, price, farm, private, market, shed_capacity)
        if result and op == "SELL" and player is not None:
            events.append({
                "step": int(current["step"]),
                "runtime_seat": int(player),
                "product": str(item),
                "price": float(price),
                "revenue": float(price),
            })
        return result

    def process(state, env):
        observation = state[0].observation
        farms = observation.farms
        current["step"] = int(get(observation, "step", 0) or 0)
        current["players"] = {id(farm): index for index, farm in enumerate(farms)}
        return original_process(state, env)

    engine._commit_unit = commit
    engine._process_market = process
    try:
        yield events
    finally:
        engine._process_market = original_process
        engine._commit_unit = original_commit


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def wool_sell_requested(action):
    total = 0
    orders = []
    for order in action.get("market", []) or []:
        if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL" and order[1] == "WOOL":
            qty = max(0, int(order[2]))
            total += qty
            orders.append(order)
    return total, orders


def snapshot(obs):
    farm0 = obs["farms"][0]
    farm1 = obs["farms"][1]
    market = obs.get("market", {}) or {}
    private = obs.get("private", {}) or {}
    return {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "self_cash": float(farm0.get("money", 0) or 0),
        "opponent_cash": float(farm1.get("money", 0) or 0),
        "wool_market_inventory": int((market.get("inventory", {}) or {}).get("WOOL", 0) or 0),
        "wool_market_price": float((market.get("prices", {}) or {}).get("WOOL", 0) or 0),
        "self_shed_wool": int((private.get("shed", {}) or {}).get("WOOL", 0) or 0),
    }


def run_policy(seed, self_file, tag):
    self_mod = load_module(ROOT / "teacher_candidates" / self_file, f"self_{tag}_{seed}")
    opp_mod = load_module(ROOT / "teacher_candidates" / "main_v21.py", f"opp_{tag}_{seed}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    steps = []
    requested_total = 0
    with capture_market_ledger() as ledger:
        while not env.done:
            obs0 = shared_obs(env, 0)
            obs1 = shared_obs(env, 1)
            a0 = plain(self_mod.agent(obs0))
            a1 = plain(opp_mod.agent(obs1))
            req, orders = wool_sell_requested(a0)
            requested_total += req
            row = snapshot(obs0)
            row["self_action"] = a0
            row["requested_wool_sell_units"] = req
            row["requested_wool_orders"] = orders
            steps.append(row)
            env.step([a0, a1])

        events = list(ledger)

    final = env.state[0].observation
    self_wool_events = [e for e in events if e["runtime_seat"] == 0 and e["product"] == "WOOL"]
    opp_wool_events = [e for e in events if e["runtime_seat"] == 1 and e["product"] == "WOOL"]

    return {
        "terminal_self": float(final["farms"][0]["money"]),
        "terminal_opponent": float(final["farms"][1]["money"]),
        "requested_wool_sell_units": requested_total,
        "executed_wool_units": len(self_wool_events),
        "realized_wool_revenue": sum(e["revenue"] for e in self_wool_events),
        "opponent_executed_wool_units": len(opp_wool_events),
        "opponent_realized_wool_revenue": sum(e["revenue"] for e in opp_wool_events),
        "wool_events": self_wool_events,
        "opponent_wool_events": opp_wool_events,
        "steps": steps,
    }


def first_diff(v18_steps, v19_steps, field):
    for a, b in zip(v18_steps, v19_steps):
        if a[field] != b[field]:
            return {
                "step": a["step"],
                "day": a["day"],
                "hour": a["hour"],
                "v18": a[field],
                "v19": b[field],
            }
    return None


def first_action_diff(v18_steps, v19_steps):
    for a, b in zip(v18_steps, v19_steps):
        if a["self_action"] != b["self_action"]:
            return {
                "step": a["step"],
                "day": a["day"],
                "hour": a["hour"],
                "v18_action": a["self_action"],
                "v19_action": b["self_action"],
                "v18_requested_wool": a["requested_wool_sell_units"],
                "v19_requested_wool": b["requested_wool_sell_units"],
                "v18_wool_price": a["wool_market_price"],
                "v19_wool_price": b["wool_market_price"],
                "v18_wool_inventory": a["wool_market_inventory"],
                "v19_wool_inventory": b["wool_market_inventory"],
                "v18_self_cash": a["self_cash"],
                "v19_self_cash": b["self_cash"],
            }
    return None


def differing_requested_steps(v18_steps, v19_steps):
    out = []
    for a, b in zip(v18_steps, v19_steps):
        if a["requested_wool_sell_units"] != b["requested_wool_sell_units"]:
            out.append({
                "step": a["step"],
                "day": a["day"],
                "hour": a["hour"],
                "v18_requested": a["requested_wool_sell_units"],
                "v19_requested": b["requested_wool_sell_units"],
                "v18_price": a["wool_market_price"],
                "v19_price": b["wool_market_price"],
                "v18_market_inventory": a["wool_market_inventory"],
                "v19_market_inventory": b["wool_market_inventory"],
                "v18_self_cash": a["self_cash"],
                "v19_self_cash": b["self_cash"],
            })
    return out


def main():
    seed = int(os.environ["SEED"])
    v18 = run_policy(seed, "v18_release_main.py", "v18")
    v19 = run_policy(seed, "candidate_v19_wool_floor.py", "v19")

    result = {
        "schema": "v18-v19-wool-sale-attribution-v0",
        "seed": seed,
        "provenance": {
            "v18_commit": V18_COMMIT,
            "v19_v21_commit": V19_V21_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
            "opponent": "V21 main.py",
            "seat": "self seat 0",
        },
        "terminal": {
            "v18": v18["terminal_self"],
            "v19": v19["terminal_self"],
            "delta_v19_minus_v18": v19["terminal_self"] - v18["terminal_self"],
        },
        "self_wool": {
            "requested_units_v18": v18["requested_wool_sell_units"],
            "requested_units_v19": v19["requested_wool_sell_units"],
            "executed_units_v18": v18["executed_wool_units"],
            "executed_units_v19": v19["executed_wool_units"],
            "executed_units_delta_v19_minus_v18": v19["executed_wool_units"] - v18["executed_wool_units"],
            "realized_revenue_v18": v18["realized_wool_revenue"],
            "realized_revenue_v19": v19["realized_wool_revenue"],
            "realized_revenue_delta_v19_minus_v18": v19["realized_wool_revenue"] - v18["realized_wool_revenue"],
        },
        "opponent_wool": {
            "executed_units_v18_world": v18["opponent_executed_wool_units"],
            "executed_units_v19_world": v19["opponent_executed_wool_units"],
            "realized_revenue_v18_world": v18["opponent_realized_wool_revenue"],
            "realized_revenue_v19_world": v19["opponent_realized_wool_revenue"],
            "realized_revenue_delta_v19_minus_v18": v19["opponent_realized_wool_revenue"] - v18["opponent_realized_wool_revenue"],
        },
        "first_self_action_diff": first_action_diff(v18["steps"], v19["steps"]),
        "first_market_inventory_diff": first_diff(v18["steps"], v19["steps"], "wool_market_inventory"),
        "first_market_price_diff": first_diff(v18["steps"], v19["steps"], "wool_market_price"),
        "first_self_cash_diff": first_diff(v18["steps"], v19["steps"], "self_cash"),
        "first_opponent_cash_diff": first_diff(v18["steps"], v19["steps"], "opponent_cash"),
        "requested_wool_diff_steps": differing_requested_steps(v18["steps"], v19["steps"]),
        "v18_wool_events": v18["wool_events"],
        "v19_wool_events": v19["wool_events"],
    }

    out = Path(f"v18_v19_wool_sale_attribution_seed{seed}.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ATTRIBUTION " + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
