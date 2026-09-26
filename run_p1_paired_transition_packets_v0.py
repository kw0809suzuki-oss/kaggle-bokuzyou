#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_world

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as whole_flow
import seyamalam_v21 as opponent

SEEDS = list(range(7001, 7011))
EXPANSION_OPS = {"BUY_LAND", "BUY_SEED", "BUY_ANIMAL"}

_EXEC_LOG = []
_FARM_ID_TO_PLAYER = {}
_CURRENT_STEP = None

_ORIG_COMMIT = official_world._commit_unit
_ORIG_BUY_LAND = official_world._do_buy_land
_ORIG_HIRE = official_world._do_hire


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__dict__"):
        return {str(k): plain(v) for k, v in vars(x).items()}
    return repr(x)


def stable_hash(obj):
    raw = json.dumps(plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def player_for_farm(farm):
    return _FARM_ID_TO_PLAYER.get(id(farm), -1)


def observed_commit(op, item, price, farm, private, market, shed_capacity=100):
    before = {
        "money": float(farm.get("money", 0) or 0),
        "seed": int((private.get("seeds", {}) or {}).get(item, 0) or 0),
        "shed": int((private.get("shed", {}) or {}).get(item, 0) or 0),
        "market_inventory": int((market.get("inventory", {}) or {}).get(item, 0) or 0),
    }
    ok = _ORIG_COMMIT(op, item, price, farm, private, market, shed_capacity)
    after = {
        "money": float(farm.get("money", 0) or 0),
        "seed": int((private.get("seeds", {}) or {}).get(item, 0) or 0),
        "shed": int((private.get("shed", {}) or {}).get(item, 0) or 0),
        "market_inventory": int((market.get("inventory", {}) or {}).get(item, 0) or 0),
    }
    _EXEC_LOG.append({
        "step": _CURRENT_STEP,
        "player": player_for_farm(farm),
        "kind": "market_unit",
        "op": op,
        "item": item,
        "quoted_price": float(price),
        "success": bool(ok),
        "before": before,
        "after": after,
    })
    return ok


def observed_buy_land(farm, board_size):
    before = {
        "money": float(farm.get("money", 0) or 0),
        "land": list(farm.get("unlocked_quadrants", []) or []),
    }
    out = _ORIG_BUY_LAND(farm, board_size)
    after = {
        "money": float(farm.get("money", 0) or 0),
        "land": list(farm.get("unlocked_quadrants", []) or []),
    }
    _EXEC_LOG.append({
        "step": _CURRENT_STEP,
        "player": player_for_farm(farm),
        "kind": "atomic_market",
        "op": "BUY_LAND",
        "success": before != after,
        "before": before,
        "after": after,
    })
    return out


def observed_hire(farm, private, board_size, mult=official_world.FARM_HAND_COST_MULT):
    before = {
        "money": float(farm.get("money", 0) or 0),
        "hands": plain(farm.get("hands", []) or []),
    }
    out = _ORIG_HIRE(farm, private, board_size, mult)
    after = {
        "money": float(farm.get("money", 0) or 0),
        "hands": plain(farm.get("hands", []) or []),
    }
    _EXEC_LOG.append({
        "step": _CURRENT_STEP,
        "player": player_for_farm(farm),
        "kind": "atomic_market",
        "op": "HIRE",
        "success": before != after,
        "before": before,
        "after": after,
    })
    return out


official_world._commit_unit = observed_commit
official_world._do_buy_land = observed_buy_land
official_world._do_hire = observed_hire


def is_expansion(order):
    if not isinstance(order, (list, tuple)) or not order:
        return False
    op = order[0]
    if op in EXPANSION_OPS:
        return True
    return op == "BUY_PRODUCT" and len(order) > 1 and order[1] == "COW"


def p1_filter(obs, native_action):
    day = int(obs.get("day", 0) or 0)
    if day < 12 or not isinstance(native_action, dict):
        return copy.deepcopy(native_action), []
    market = list(native_action.get("market", []) or [])
    removed = [copy.deepcopy(o) for o in market if is_expansion(o)]
    if not removed:
        return copy.deepcopy(native_action), []
    revised = copy.deepcopy(native_action)
    revised["market"] = [copy.deepcopy(o) for o in market if not is_expansion(o)]
    return revised, removed


def summarize_farm(farm, private):
    plants = {}
    placed_animals = {}
    structures = {}
    empty = 0
    weeds = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if tile is None:
                empty += 1
                continue
            if tile == "LOCKED":
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crop = tile.get("crop")
                plants[crop] = plants.get(crop, 0) + 1
            elif kind == "WEED":
                weeds += 1
            if "animal" in tile:
                animal = tile.get("animal")
                placed_animals[animal] = placed_animals.get(animal, 0) + 1
            if kind in ("COOP", "PASTURE"):
                structures[kind] = structures.get(kind, 0) + 1

    shed = private.get("shed", {}) or {}
    return {
        "cash": float(farm.get("money", 0) or 0),
        "land": list(farm.get("unlocked_quadrants", []) or []),
        "land_count": len(farm.get("unlocked_quadrants", []) or []),
        "hands": plain(farm.get("hands", []) or []),
        "hands_count": len(farm.get("hands", []) or []),
        "farmer": plain(farm.get("farmer")),
        "seeds": {k: int(v or 0) for k, v in (private.get("seeds", {}) or {}).items()},
        "shed": {k: int(v or 0) for k, v in shed.items() if int(v or 0) != 0},
        "inventories": plain(private.get("inventories", []) or []),
        "plants": plants,
        "placed_animals": placed_animals,
        "shed_animals": {k: int(shed.get(k, 0) or 0) for k in ("GOOSE", "COW", "SHEEP")},
        "structures": structures,
        "empty_tiles": empty,
        "weeds": weeds,
    }


def snapshot(env):
    obs0 = env.state[0].observation
    obs1 = env.state[1].observation
    farms = obs0.farms
    core = {
        "day": int(obs0.day),
        "hour": int(obs0.hour),
        "self": summarize_farm(farms[0], obs0.private),
        "opponent": summarize_farm(farms[1], obs1.private),
        "market": plain(obs0.market),
        "town": plain(obs0.town),
    }
    fingerprint_source = {
        "day": int(obs0.day),
        "hour": int(obs0.hour),
        "farms": plain(farms),
        "private0": plain(obs0.private),
        "private1": plain(obs1.private),
        "market": plain(obs0.market),
        "town": plain(obs0.town),
    }
    core["world_hash"] = stable_hash(fingerprint_source)
    return core


def run_arm(arm, seed):
    global _CURRENT_STEP, _FARM_ID_TO_PLAYER
    whole_flow.reset_telemetry()
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)
    trace = []
    step = 0

    while not env.done:
        pre = snapshot(env)
        obs_self = env.state[0].observation
        obs_opp = env.state[1].observation

        native_self = plain(whole_flow.agent(obs_self))
        if arm == "p1":
            submitted_self, removed = p1_filter(obs_self, native_self)
        else:
            submitted_self = copy.deepcopy(native_self)
            _, removed = p1_filter(obs_self, native_self)  # hypothetical only, for pairing
        submitted_opp = plain(opponent.agent(obs_opp))

        _EXEC_LOG.clear()
        _CURRENT_STEP = step
        _FARM_ID_TO_PLAYER = {
            id(env.state[0].observation.farms[0]): 0,
            id(env.state[0].observation.farms[1]): 1,
        }

        env.step([submitted_self, submitted_opp])
        post = snapshot(env)
        world_exec = [copy.deepcopy(e) for e in _EXEC_LOG]

        trace.append({
            "step": step,
            "day": pre["day"],
            "hour": pre["hour"],
            "pre": pre,
            "native_self_action": native_self,
            "submitted_self_action": plain(submitted_self),
            "removed_by_p1_rule": plain(removed),
            "opponent_action": submitted_opp,
            "official_world_execution": world_exec,
            "post": post,
        })
        step += 1

    result = {
        "arm": arm,
        "seed": seed,
        "terminal_self": float(env.state[0].reward),
        "terminal_opponent": float(env.state[1].reward),
        "terminal_margin": float(env.state[0].reward) - float(env.state[1].reward),
        "steps": len(trace),
        "trace": trace,
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


def child(arm, seed):
    cp = subprocess.run(
        [sys.executable, __file__, "--arm", arm, "--seed", str(seed)],
        check=True, capture_output=True, text=True
    )
    return json.loads(cp.stdout.strip().splitlines()[-1])


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metric_delta(p1, base):
    out = {
        "cash": p1["cash"] - base["cash"],
        "land_count": p1["land_count"] - base["land_count"],
        "hands_count": p1["hands_count"] - base["hands_count"],
        "empty_tiles": p1["empty_tiles"] - base["empty_tiles"],
        "weeds": p1["weeds"] - base["weeds"],
    }
    for group in ("seeds", "plants", "placed_animals", "shed_animals", "structures"):
        keys = sorted(set(base.get(group, {})) | set(p1.get(group, {})))
        out[group] = {k: p1.get(group, {}).get(k, 0) - base.get(group, {}).get(k, 0) for k in keys}
    return out


def self_exec(events):
    return [e for e in events if e.get("player") == 0]


def build_packet(base, p1):
    bt = base["trace"]
    pt = p1["trace"]
    n = min(len(bt), len(pt))

    first_action = None
    for i in range(n):
        if action_key(bt[i]["submitted_self_action"]) != action_key(pt[i]["submitted_self_action"]):
            first_action = i
            break

    first_world = None
    start = first_action if first_action is not None else 0
    for i in range(start, n):
        if bt[i]["post"]["world_hash"] != pt[i]["post"]["world_hash"]:
            first_world = i
            break

    first_followup = None
    if first_world is not None:
        for i in range(first_world + 1, n):
            if action_key(bt[i]["native_self_action"]) != action_key(pt[i]["native_self_action"]):
                first_followup = i
                break

    packet = {
        "seed": base["seed"],
        "terminal": {
            "baseline_self": base["terminal_self"],
            "p1_self": p1["terminal_self"],
            "delta_self": p1["terminal_self"] - base["terminal_self"],
            "baseline_opponent": base["terminal_opponent"],
            "p1_opponent": p1["terminal_opponent"],
            "baseline_margin": base["terminal_margin"],
            "p1_margin": p1["terminal_margin"],
        },
        "first_action_divergence": None,
        "first_world_divergence": None,
        "first_downstream_current_divergence": None,
        "followup_3_turns_after_world_divergence": [],
    }

    if first_action is not None:
        b = bt[first_action]
        p = pt[first_action]
        packet["first_action_divergence"] = {
            "step": first_action,
            "day": b["day"],
            "hour": b["hour"],
            "pre_state_equal": b["pre"]["world_hash"] == p["pre"]["world_hash"],
            "baseline_pre_state": b["pre"],
            "p1_pre_state": p["pre"],
            "baseline_native_action": b["native_self_action"],
            "baseline_submitted_action": b["submitted_self_action"],
            "p1_native_action": p["native_self_action"],
            "p1_submitted_action": p["submitted_self_action"],
            "p1_removed_action": p["removed_by_p1_rule"],
            "baseline_official_world_execution": self_exec(b["official_world_execution"]),
            "p1_official_world_execution": self_exec(p["official_world_execution"]),
            "baseline_post_state": b["post"],
            "p1_post_state": p["post"],
            "immediate_self_state_delta_p1_minus_baseline": metric_delta(p["post"]["self"], b["post"]["self"]),
        }

    if first_world is not None:
        b = bt[first_world]
        p = pt[first_world]
        packet["first_world_divergence"] = {
            "step": first_world,
            "day": b["day"],
            "hour": b["hour"],
            "pre_state_equal": b["pre"]["world_hash"] == p["pre"]["world_hash"],
            "baseline_action": b["submitted_self_action"],
            "p1_action": p["submitted_self_action"],
            "baseline_official_world_execution": self_exec(b["official_world_execution"]),
            "p1_official_world_execution": self_exec(p["official_world_execution"]),
            "baseline_post_state": b["post"],
            "p1_post_state": p["post"],
            "self_state_delta_p1_minus_baseline": metric_delta(p["post"]["self"], b["post"]["self"]),
        }
        for j in range(first_world + 1, min(first_world + 4, n)):
            packet["followup_3_turns_after_world_divergence"].append({
                "step": j,
                "baseline_pre_state": bt[j]["pre"],
                "p1_pre_state": pt[j]["pre"],
                "baseline_native_action": bt[j]["native_self_action"],
                "p1_native_action": pt[j]["native_self_action"],
                "baseline_submitted_action": bt[j]["submitted_self_action"],
                "p1_submitted_action": pt[j]["submitted_self_action"],
                "baseline_post_state": bt[j]["post"],
                "p1_post_state": pt[j]["post"],
            })

    if first_followup is not None:
        b = bt[first_followup]
        p = pt[first_followup]
        packet["first_downstream_current_divergence"] = {
            "step": first_followup,
            "day": b["day"],
            "hour": b["hour"],
            "baseline_pre_state": b["pre"],
            "p1_pre_state": p["pre"],
            "pre_state_delta_p1_minus_baseline": metric_delta(p["pre"]["self"], b["pre"]["self"]),
            "baseline_native_action": b["native_self_action"],
            "p1_native_action": p["native_self_action"],
            "baseline_submitted_action": b["submitted_self_action"],
            "p1_submitted_action": p["submitted_self_action"],
            "baseline_official_world_execution": self_exec(b["official_world_execution"]),
            "p1_official_world_execution": self_exec(p["official_world_execution"]),
            "baseline_post_state": b["post"],
            "p1_post_state": p["post"],
        }

    packet["observation_complete"] = all([
        packet["first_action_divergence"] is not None,
        packet["first_action_divergence"] is not None and
            packet["first_action_divergence"]["baseline_pre_state"] is not None and
            packet["first_action_divergence"]["baseline_post_state"] is not None,
        packet["first_world_divergence"] is not None,
        packet["first_action_divergence"] is not None and
            packet["first_action_divergence"]["baseline_official_world_execution"] is not None and
            packet["first_action_divergence"]["p1_official_world_execution"] is not None,
        packet["first_downstream_current_divergence"] is not None,
    ])
    return packet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "p1"])
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    if args.arm:
        run_arm(args.arm, args.seed)
        return

    packets = []
    for seed in SEEDS:
        base = child("baseline", seed)
        p1 = child("p1", seed)
        packet = build_packet(base, p1)
        packets.append(packet)
        print("PACKET " + json.dumps({
            "seed": seed,
            "first_action_step": None if packet["first_action_divergence"] is None else packet["first_action_divergence"]["step"],
            "first_world_step": None if packet["first_world_divergence"] is None else packet["first_world_divergence"]["step"],
            "first_followup_step": None if packet["first_downstream_current_divergence"] is None else packet["first_downstream_current_divergence"]["step"],
            "delta_self": packet["terminal"]["delta_self"],
            "complete": packet["observation_complete"],
        }, separators=(",", ":")))

    output = {
        "schema": "p1-paired-transition-packets-v0",
        "purpose": "paired transition observation after positive terminal screening",
        "baseline": "whole_flow_control_agent",
        "p1": "whole_flow + D12 expansion closure",
        "world": "official Kaggriculture interpreter; execution hooks wrap official functions without changing return values",
        "opponent": "Seyamalam pinned 8b8c421e...",
        "seeds": SEEDS,
        "observation_completion_conditions": [
            "Action difference observed",
            "Pre-State and Post-State observed",
            "Official World execution result observed",
            "Baseline and P1 observed symmetrically",
            "Current follow-up from the next State connected",
        ],
        "packets": packets,
    }
    Path("p1_paired_transition_packets_v0.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps({
        "packets": len(packets),
        "complete": sum(p["observation_complete"] for p in packets),
        "improved": sum(p["terminal"]["delta_self"] > 0 for p in packets),
        "worsened": sum(p["terminal"]["delta_self"] < 0 for p in packets),
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
