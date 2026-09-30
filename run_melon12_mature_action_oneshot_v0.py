#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEED = 92801801
TARGET = (0, 1)
SUBJECT_SEAT = 0
STOP_AFTER_TRIGGER = 120

CANDIDATE = ROOT / "independent_day0_melon12_v1.py"
SELLESTA = ROOT / "opponents" / "sellesta_main_pinned.py"


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
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def snapshot(env):
    return {
        "seat0": plain(env.state[0].observation),
        "seat1": plain(env.state[1].observation),
    }


def normalize_for_hash(x):
    x = deepcopy(x)
    for seat in ("seat0", "seat1"):
        obs = x[seat]
        obs.pop("remainingOverageTime", None)
    return x


def fingerprint(x):
    raw = json.dumps(normalize_for_hash(x), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def same_target(tile):
    return (
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == "MELON"
        and int(tile.get("planted_day", -1)) == 0
    )


def run(label, intervene):
    candidate = load_module(CANDIDATE, f"candidate_{label}")
    sellesta = load_module(SELLESTA, f"sellesta_{label}")

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    trigger = None
    trace = []
    result = None

    while not env.done:
        pre_shared0 = plain(shared_obs(env, 0))
        pre_shared1 = plain(shared_obs(env, 1))
        pre = snapshot(env)
        step = int(pre["seat0"].get("step", len(trace)))

        actions = [
            plain(candidate.agent(pre_shared0)),
            plain(sellesta.agent(pre_shared1)),
        ]

        farm = pre["seat0"]["farms"][0]
        x, y = TARGET
        tile = farm["tiles"][y][x]

        if trigger is None and same_target(tile) and int(tile.get("yield_units", 0) or 0) >= 6:
            occupants = []
            if list(farm.get("farmer", [])) == [x, y]:
                occupants.append(("farmer", 0))
            for i, pos in enumerate(farm.get("hands", []) or []):
                if list(pos) == [x, y]:
                    occupants.append(("hand", i))

            chosen = None
            for who, idx in occupants:
                a = actions[0]["farmer"] if who == "farmer" else actions[0]["hands"][idx]
                if a == ["NORTH"]:
                    chosen = (who, idx, deepcopy(a))
                    break

            trigger = {
                "step": step,
                "state_hash": fingerprint(pre),
                "target_tile": deepcopy(tile),
                "occupants": occupants,
                "original_action": None if chosen is None else chosen[2],
                "unit": None if chosen is None else {"kind": chosen[0], "index": chosen[1]},
                "injected_action": ["EAST"] if intervene and chosen is not None else None,
                "guard_pass": chosen is not None,
            }

            if intervene and chosen is not None:
                who, idx, _ = chosen
                if who == "farmer":
                    actions[0]["farmer"] = ["EAST"]
                else:
                    actions[0]["hands"][idx] = ["EAST"]

        target_pre = deepcopy(tile) if same_target(tile) else None
        env.step(actions)
        post = snapshot(env)
        post_tile = post["seat0"]["farms"][0]["tiles"][y][x]

        row = {
            "step": step,
            "target_pre": target_pre,
            "action_seat0": deepcopy(actions[0]),
            "target_post": deepcopy(post_tile),
        }

        if trigger is not None and step >= trigger["step"]:
            trace.append(row)

            if target_pre is not None and not same_target(post_tile):
                harvested = False
                unit = trigger["unit"]
                if unit is not None:
                    if unit["kind"] == "farmer":
                        applied = actions[0]["farmer"]
                        inv_pre = pre["seat0"]["private"]["inventories"][0].get("MELON", 0)
                        inv_post = post["seat0"]["private"]["inventories"][0].get("MELON", 0)
                    else:
                        idx = unit["index"]
                        applied = actions[0]["hands"][idx]
                        inv_pre = pre["seat0"]["private"]["inventories"][idx + 1].get("MELON", 0)
                        inv_post = post["seat0"]["private"]["inventories"][idx + 1].get("MELON", 0)
                    harvested = applied == ["HARVEST"] and inv_post > inv_pre

                result = {
                    "resolved_step": step,
                    "resolved_as": "HARVEST" if harvested else "NON_HARVEST_DISAPPEARANCE",
                    "post_tile": deepcopy(post_tile),
                }
                break

            if step - trigger["step"] >= STOP_AFTER_TRIGGER:
                result = {
                    "resolved_step": step,
                    "resolved_as": "SHORT_WINDOW_NO_RESOLUTION",
                    "post_tile": deepcopy(post_tile),
                }
                break

    return {
        "label": label,
        "intervene": intervene,
        "trigger": trigger,
        "result": result,
        "trace": trace,
    }


def main():
    control = run("control", False)
    probe = run("probe", True)

    state_equal = (
        control["trigger"] is not None
        and probe["trigger"] is not None
        and control["trigger"]["state_hash"] == probe["trigger"]["state_hash"]
    )

    out = {
        "schema": "melon12-mature-action-oneshot-v0",
        "seed": SEED,
        "target": list(TARGET),
        "state_equal_at_trigger": state_equal,
        "control_trigger": control["trigger"],
        "probe_trigger": probe["trigger"],
        "control_result": control["result"],
        "probe_result": probe["result"],
    }

    if probe["result"] and probe["result"]["resolved_as"] == "HARVEST":
        out["probe_trace_until_harvest"] = probe["trace"]

    Path("melon12_mature_action_oneshot_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
