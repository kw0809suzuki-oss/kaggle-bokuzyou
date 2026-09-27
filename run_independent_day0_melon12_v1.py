#!/usr/bin/env python3
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as kg

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ["SEED"])

OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
INDEPENDENT_COMMIT = "eff2cf1ae6eb03809d215248f9b05b79a2b4dfe9"
SELLESTA_COMMIT = "0a5ce7ef83211d6df5824a4e7e17f3d4a40c22e7"

MODELS = {
    "baseline": ROOT / "astra_flow_independent_distilled_v0.py",
    "candidate_v1": ROOT / "independent_day0_melon12_v1.py",
}
SELLESTA = ROOT / "opponents" / "sellesta_main_pinned.py"

_ORIG_COMMIT_UNIT = kg._commit_unit
_ORIG_APPLY_UNIT_ACTION = kg._apply_unit_action
_CTX = {"step": -1, "farm_ids": {}, "private_ids": {}, "events": []}


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


def public_state(obs):
    out = plain(obs)
    out.pop("private", None)
    out.pop("player", None)
    return out


def unit_position(farm, idx):
    if idx == 0:
        return list(farm.get("farmer", []))
    hands = farm.get("hands", []) or []
    return list(hands[idx - 1]) if idx - 1 < len(hands) else None


def melon_tiles(farm):
    out = []
    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
                item = {"x": x, "y": y}
                item.update(plain(tile))
                out.append(item)
    return out


def item_inventory(private, item):
    total = int((private.get("shed", {}) or {}).get(item, 0) or 0)
    for inv in private.get("inventories", []) or []:
        total += int(inv.get(item, 0) or 0)
    return total


def _record_commit_unit(op, item, price, farm, private, market, shed_capacity=100):
    seat = _CTX["farm_ids"].get(id(farm))
    ok = _ORIG_COMMIT_UNIT(op, item, price, farm, private, market, shed_capacity)
    if ok and seat is not None:
        _CTX["events"].append({
            "kind": "MARKET_COMMIT",
            "step": int(_CTX["step"]),
            "seat": int(seat),
            "op": str(op),
            "item": str(item),
            "units": 1,
            "unit_price": float(price),
        })
    return ok


def _record_apply_unit_action(farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100):
    seat = _CTX["private_ids"].get(id(private))
    pos = unit_position(farm, idx)
    before_tile = None
    before_item = None
    before_inv = None

    if seat is not None and pos and len(pos) == 2:
        x, y = pos
        tile = farm["tiles"][y][x]
        before_tile = plain(tile)
        if isinstance(action, list) and action:
            if action[0] == "HARVEST" and isinstance(tile, dict):
                if tile.get("kind") == "PLANT":
                    before_item = tile.get("crop")
                elif "animal" in tile:
                    before_item = kg.ANIMALS[tile["animal"]]["product"]
                if before_item:
                    before_inv = item_inventory(private, before_item)

    _ORIG_APPLY_UNIT_ACTION(
        farm, private, idx, action, board_size, day, turns_per_day, shed_capacity
    )

    if seat is None or not pos or not isinstance(action, list) or not action:
        return

    x, y = pos
    after_tile = plain(farm["tiles"][y][x])
    op = action[0]

    if op == "PLANT" and len(action) >= 2:
        crop = action[1]
        if (
            before_tile is None
            and isinstance(after_tile, dict)
            and after_tile.get("kind") == "PLANT"
            and after_tile.get("crop") == crop
        ):
            _CTX["events"].append({
                "kind": "PLANT_COMMIT",
                "step": int(_CTX["step"]),
                "seat": int(seat),
                "unit_index": int(idx),
                "crop": str(crop),
                "x": int(x),
                "y": int(y),
            })

    if op == "HARVEST" and before_item and before_inv is not None:
        delta = item_inventory(private, before_item) - before_inv
        if delta > 0:
            _CTX["events"].append({
                "kind": "HARVEST_COMMIT",
                "step": int(_CTX["step"]),
                "seat": int(seat),
                "unit_index": int(idx),
                "item": str(before_item),
                "units": int(delta),
                "x": int(x),
                "y": int(y),
            })


kg._commit_unit = _record_commit_unit
kg._apply_unit_action = _record_apply_unit_action


def configure_recorder(env):
    farms = env.state[0].observation.farms
    privates = [env.state[i].observation.private for i in range(2)]
    _CTX["farm_ids"] = {id(farms[i]): i for i in range(2)}
    _CTX["private_ids"] = {id(privates[i]): i for i in range(2)}
    _CTX["events"] = []
    _CTX["step"] = -1


def snapshot(env):
    o0 = plain(env.state[0].observation)
    o1 = plain(env.state[1].observation)
    return {
        "public": public_state(o0),
        "private_seat0": plain(o0.get("private", {}) or {}),
        "private_seat1": plain(o1.get("private", {}) or {}),
    }


def build_view(turns, subject_seat):
    day0_end = None
    for t in turns:
        post = t["post"]["public"]
        if int(post.get("day", -1)) == 1 and int(post.get("hour", -1)) == 0:
            farm = post["farms"][subject_seat]
            day0_end = {
                "cash": float(farm["money"]),
                "melon_tiles": melon_tiles(farm),
                "wheat_tiles": [
                    {"x": x, "y": y, **plain(tile)}
                    for y, row in enumerate(farm.get("tiles", []) or [])
                    for x, tile in enumerate(row)
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT"
                ],
            }
            break

    day0_coords = []
    if day0_end is not None:
        day0_coords = [[m["x"], m["y"]] for m in day0_end["melon_tiles"]]

    life = {}
    for x, y in day0_coords:
        key = f"{x},{y}"
        life[key] = {
            "x": x,
            "y": y,
            "max_yield_units": 0,
            "first_missing_step": None,
            "harvest_units": 0,
        }

    for t in turns:
        post = t["post"]["public"]
        farm = post["farms"][subject_seat]
        step = int(post.get("step", t["step"] + 1))
        for key, rec in life.items():
            x, y = rec["x"], rec["y"]
            tile = farm["tiles"][y][x]
            is_same_melon = (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "MELON"
            )
            if is_same_melon:
                rec["max_yield_units"] = max(rec["max_yield_units"], int(tile.get("yield_units", 0) or 0))
            elif rec["first_missing_step"] is None:
                rec["first_missing_step"] = step

        for e in t.get("world_execution_events", []):
            if (
                e.get("kind") == "HARVEST_COMMIT"
                and e.get("seat") == subject_seat
                and e.get("item") == "MELON"
            ):
                key = f"{e['x']},{e['y']}"
                if key in life:
                    life[key]["harvest_units"] += int(e["units"])

    market_events = [
        e
        for t in turns
        for e in t.get("world_execution_events", [])
        if e.get("kind") == "MARKET_COMMIT"
        and e.get("seat") == subject_seat
        and e.get("item") == "MELON"
    ]
    melon_sells = [e for e in market_events if e.get("op") == "SELL"]
    sell_units = sum(int(e["units"]) for e in melon_sells)
    sell_revenue = sum(float(e["unit_price"]) * int(e["units"]) for e in melon_sells)

    harvest_events = [
        e
        for t in turns
        for e in t.get("world_execution_events", [])
        if e.get("kind") == "HARVEST_COMMIT"
        and e.get("seat") == subject_seat
        and e.get("item") == "MELON"
    ]

    final = turns[-1]["post"]["public"]
    self_cash = float(final["farms"][subject_seat]["money"])
    opp_cash = float(final["farms"][1 - subject_seat]["money"])

    return {
        "reachability": {
            "day0_melon_count": len(day0_coords),
            "day0_melon_coords": day0_coords,
            "pass_12": len(day0_coords) == 12,
            "day0_wheat_count": len(day0_end["wheat_tiles"]) if day0_end else None,
            "day0_cash": day0_end["cash"] if day0_end else None,
        },
        "melon_life": list(life.values()),
        "melon_harvest_units": sum(int(e["units"]) for e in harvest_events),
        "melon_sell_units": sell_units,
        "melon_sell_revenue": sell_revenue,
        "melon_avg_sell_price": (sell_revenue / sell_units) if sell_units else None,
        "terminal_self": self_cash,
        "terminal_sellesta": opp_cash,
        "margin": self_cash - opp_cash,
    }


def run_game(label, subject_seat):
    subject = load_module(MODELS[label], f"{label}_{SEED}_{subject_seat}")
    sellesta = load_module(SELLESTA, f"sellesta_{label}_{SEED}_{subject_seat}")

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)
    configure_recorder(env)

    mods = [None, None]
    mods[subject_seat] = subject
    mods[1 - subject_seat] = sellesta

    turns = []
    while not env.done:
        pre0 = plain(shared_obs(env, 0))
        pre1 = plain(shared_obs(env, 1))
        pre = snapshot(env)
        step = int(pre["public"].get("step", len(turns)))
        _CTX["step"] = step
        _CTX["events"] = []

        actions = [
            plain(mods[0].agent(pre0)),
            plain(mods[1].agent(pre1)),
        ]
        env.step(actions)
        post = snapshot(env)

        candidate_private_pre = pre[f"private_seat{subject_seat}"]
        candidate_private_post = post[f"private_seat{subject_seat}"]
        candidate_farm_pre = pre["public"]["farms"][subject_seat]
        candidate_farm_post = post["public"]["farms"][subject_seat]

        turns.append({
            "step": step,
            "day": int(pre["public"].get("day", 0)),
            "hour": int(pre["public"].get("hour", 0)),
            "subject_label": label,
            "subject_seat": subject_seat,
            "pre": pre,
            "action_seat0": actions[0],
            "action_seat1": actions[1],
            "post": post,
            "candidate_private_pre": candidate_private_pre,
            "candidate_private_post": candidate_private_post,
            "market_pre": plain(pre["public"].get("market", {})),
            "market_post": plain(post["public"].get("market", {})),
            "candidate_melon_tiles_pre": melon_tiles(candidate_farm_pre),
            "candidate_melon_tiles_post": melon_tiles(candidate_farm_post),
            "world_execution_events": plain(_CTX["events"]),
        })

    battle_id = f"{label}-vs-sellesta-seat{subject_seat}-seed{SEED}"
    provenance = {
        "official_commit": OFFICIAL_COMMIT,
        "independent_commit": INDEPENDENT_COMMIT,
        "sellesta_commit": SELLESTA_COMMIT,
        "experiment_git_sha": os.environ.get("GITHUB_SHA"),
        "seed": SEED,
        "subject_label": label,
        "subject_seat": subject_seat,
    }

    battle = {
        "schema": "kaggriculture-battle-evidence-v1",
        "battle_id": battle_id,
        "provenance": provenance,
        "turn_count": len(turns),
        "turns": turns,
    }

    battle_path = Path(f"melon12_v1_{SEED}_{label}_seat{subject_seat}_battle.json.gz")
    with gzip.open(battle_path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(battle, fh, ensure_ascii=False, separators=(",", ":"))

    # Views are computed only after the full 719-turn Battle has been saved.
    view = build_view(turns, subject_seat)
    return {
        "battle_id": battle_id,
        "battle_file": battle_path.name,
        "turn_count": len(turns),
        "view": view,
    }


def main():
    results = []
    for seat in (0, 1):
        results.append(run_game("baseline", seat))
        results.append(run_game("candidate_v1", seat))

    paired = []
    for seat in (0, 1):
        base = next(r for r in results if r["battle_id"].startswith("baseline-") and f"seat{seat}-" in r["battle_id"])
        cand = next(r for r in results if r["battle_id"].startswith("candidate_v1-") and f"seat{seat}-" in r["battle_id"])
        bv = base["view"]
        cv = cand["view"]
        paired.append({
            "seat": seat,
            "reachability_baseline": bv["reachability"],
            "reachability_candidate_v1": cv["reachability"],
            "melon_harvest_delta": cv["melon_harvest_units"] - bv["melon_harvest_units"],
            "melon_sell_units_delta": cv["melon_sell_units"] - bv["melon_sell_units"],
            "melon_sell_revenue_delta": cv["melon_sell_revenue"] - bv["melon_sell_revenue"],
            "terminal_self_delta": cv["terminal_self"] - bv["terminal_self"],
            "margin_delta": cv["margin"] - bv["margin"],
        })

    out = {
        "schema": "independent-day0-melon12-v1-summary",
        "seed": SEED,
        "note": "Battle files are Evidence. Views and paired judgments are derived after full completion.",
        "results": results,
        "paired": paired,
    }
    Path(f"melon12_v1_seed{SEED}_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("MELON12_V1 " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
