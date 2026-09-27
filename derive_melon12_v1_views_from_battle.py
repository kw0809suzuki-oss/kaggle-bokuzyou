#!/usr/bin/env python3
"""Derive MELON12 v1 Views from saved Battle evidence.

Input is one or more kaggriculture-battle-evidence-v1 .json.gz files.
No Battle is rerun. The script derives only from saved pre-State, Action,
post-State and private inventory.
"""
import argparse
import gzip
import json
from pathlib import Path


def inv_total(private, item):
    return int((private.get("shed", {}) or {}).get(item, 0) or 0) + sum(
        int((inv or {}).get(item, 0) or 0)
        for inv in (private.get("inventories", []) or [])
    )


def actions_for(turn, seat):
    action = turn[f"action_seat{seat}"]
    return [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]


def positions(farm):
    return [farm.get("farmer"), *(farm.get("hands", []) or [])]


def melon_harvests(turn, seat, coord_filter=None):
    farm = turn["pre"]["public"]["farms"][seat]
    out = []
    for action, pos in zip(actions_for(turn, seat), positions(farm)):
        if not pos or not isinstance(action, list) or not action or action[0] != "HARVEST":
            continue
        x, y = pos
        if coord_filter is not None and (x, y) not in coord_filter:
            continue
        tile = farm["tiles"][y][x]
        if not (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "MELON"
            and int(tile.get("yield_units", 0) or 0) > 0
        ):
            continue
        if int(turn["day"]) - int(tile.get("planted_day", 0)) < 10:
            continue
        out.append({
            "step": int(turn["step"]),
            "day": int(turn["day"]),
            "hour": int(turn["hour"]),
            "x": int(x),
            "y": int(y),
            "units": int(tile["yield_units"]),
        })
    return out


def derive(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        battle = json.load(fh)

    turns = battle["turns"]
    seat = int(battle["provenance"]["subject_seat"])

    day0_coords = []
    start_index = 0
    day0_cash = None
    day0_wheat = None
    for i, turn in enumerate(turns):
        post = turn["post"]["public"]
        if int(post.get("day", -1)) == 1 and int(post.get("hour", -1)) == 0:
            farm = post["farms"][seat]
            for y, row in enumerate(farm["tiles"]):
                for x, tile in enumerate(row):
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        if tile.get("crop") == "MELON":
                            day0_coords.append((x, y))
            day0_wheat = sum(
                1
                for row in farm["tiles"]
                for tile in row
                if isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "WHEAT"
            )
            day0_cash = float(farm["money"])
            start_index = i + 1
            break

    coordset = set(day0_coords)
    max_yield = {coord: 0 for coord in day0_coords}
    first_exit = {coord: None for coord in day0_coords}

    for turn in turns[start_index:]:
        farm = turn["post"]["public"]["farms"][seat]
        post_step = int(turn["post"]["public"].get("step", turn["step"] + 1))
        for coord in day0_coords:
            x, y = coord
            tile = farm["tiles"][y][x]
            same = (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == "MELON"
            )
            if same:
                max_yield[coord] = max(
                    max_yield[coord], int(tile.get("yield_units", 0) or 0)
                )
            elif first_exit[coord] is None:
                first_exit[coord] = {
                    "step": post_step,
                    "day": int(turn["post"]["public"].get("day", 0)),
                    "hour": int(turn["post"]["public"].get("hour", 0)),
                    "post_tile": tile,
                }

    all_harvest_events = []
    origin_harvest_events = []
    sold_units = 0
    unexplained_inventory_losses = []

    for turn in turns:
        harvest_events = melon_harvests(turn, seat)
        origin_events = melon_harvests(turn, seat, coordset)
        all_harvest_events.extend(harvest_events)
        origin_harvest_events.extend(origin_events)
        harvested_this_turn = sum(e["units"] for e in harvest_events)

        market = turn[f"action_seat{seat}"].get("market", []) or []
        requested_sell = sum(
            int(order[2])
            for order in market
            if isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
            and order[1] == "MELON"
        )

        pre_total = inv_total(turn["candidate_private_pre"], "MELON")
        post_total = inv_total(turn["candidate_private_post"], "MELON")
        inventory_drop_after_harvest = pre_total + harvested_this_turn - post_total

        if requested_sell:
            if 0 <= inventory_drop_after_harvest <= requested_sell:
                sold_units += inventory_drop_after_harvest
            else:
                unexplained_inventory_losses.append({
                    "step": int(turn["step"]),
                    "requested_sell": requested_sell,
                    "pre_total": pre_total,
                    "harvested": harvested_this_turn,
                    "post_total": post_total,
                    "inferred_drop": inventory_drop_after_harvest,
                })
        elif inventory_drop_after_harvest > 0:
            unexplained_inventory_losses.append({
                "step": int(turn["step"]),
                "requested_sell": 0,
                "pre_total": pre_total,
                "harvested": harvested_this_turn,
                "post_total": post_total,
                "inferred_drop": inventory_drop_after_harvest,
            })

    final = turns[-1]["post"]["public"]
    self_cash = float(final["farms"][seat]["money"])
    opp_cash = float(final["farms"][1 - seat]["money"])

    life = []
    harvested_by_coord = {}
    for event in origin_harvest_events:
        coord = (event["x"], event["y"])
        harvested_by_coord[coord] = harvested_by_coord.get(coord, 0) + event["units"]
    for coord in day0_coords:
        life.append({
            "x": coord[0],
            "y": coord[1],
            "max_yield_units": max_yield[coord],
            "harvest_units": harvested_by_coord.get(coord, 0),
            "first_exit": first_exit[coord],
        })

    return {
        "battle_id": battle["battle_id"],
        "battle_file": Path(path).name,
        "turn_count": len(turns),
        "view": {
            "reachability": {
                "day0_melon_count": len(day0_coords),
                "day0_melon_coords": [list(c) for c in day0_coords],
                "pass_12": len(day0_coords) == 12,
                "day0_wheat_count": day0_wheat,
                "day0_cash": day0_cash,
            },
            "day0_melon_life": life,
            "day0_melon_max6_count": sum(v >= 6 for v in max_yield.values()),
            "day0_melon_harvested_coord_count": len(harvested_by_coord),
            "day0_melon_harvest_units": sum(e["units"] for e in origin_harvest_events),
            "all_melon_harvest_units": sum(e["units"] for e in all_harvest_events),
            "all_melon_sell_units": sold_units,
            "end_melon_inventory": inv_total(turns[-1]["candidate_private_post"], "MELON"),
            "unexplained_melon_inventory_losses": unexplained_inventory_losses,
            "terminal_self": self_cash,
            "terminal_sellesta": opp_cash,
            "margin": self_cash - opp_cash,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("battle_files", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    out = {
        "schema": "independent-day0-melon12-v1-derived-views",
        "source": "saved Battle evidence only; no rerun",
        "results": [derive(path) for path in args.battle_files],
    }
    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
