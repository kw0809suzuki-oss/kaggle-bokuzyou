#!/usr/bin/env python3
"""Trace the first MOVE-on-mature candidate for four turns.

Observation only. Determine whether actor11's step252 EAST is an idle move
or part of a continuing route/work sequence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "adaptive_circulation_runtime_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

SEEDS = [
    92802001, 92802002, 92802003, 92802004, 92802005,
    92802006, 92802007, 92802008, 92802009, 92802010,
]
START = 252
END = 257
ACTOR = 11


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


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def positions(farm):
    return [plain(farm.get("farmer"))] + [plain(x) for x in (farm.get("hands", []) or [])]


def unit_actions(bundle, count):
    out = [plain(bundle.get("farmer", ["PASS"]))]
    out.extend(plain(bundle.get("hands", []) or []))
    while len(out) < count:
        out.append(["PASS"])
    return out[:count]


def tile_at(farm, pos):
    x, y = int(pos[0]), int(pos[1])
    tiles = farm.get("tiles", []) or []
    if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
        return plain(tiles[y][x])
    return None


def inv(private, actor):
    xs = private.get("inventories", []) or []
    if actor < len(xs) and isinstance(xs[actor], dict):
        return {k: float(v) for k, v in xs[actor].items() if isinstance(v, (int, float)) and v}
    return {}


def run(seed):
    model = load(MODEL, f"route_model_{seed}")
    opp = load(OPPONENT, f"route_opp_{seed}")
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    trace = []
    while not env.done:
        o0 = shared(env, 0)
        o1 = shared(env, 1)
        a0 = plain(model.agent(o0, env.configuration))
        a1 = plain(opp.agent(o1))
        step = int(o0["step"])

        if START <= step <= END:
            farm = o0["farms"][0]
            pos = positions(farm)
            acts = unit_actions(a0, len(pos))
            actor_pos = pos[ACTOR]
            trace.append({
                "step": step,
                "day": int(o0["day"]),
                "hour": int(o0["hour"]),
                "position": actor_pos,
                "tile": tile_at(farm, actor_pos),
                "inventory": inv(o0["private"], ACTOR),
                "action": acts[ACTOR],
            })

        env.step([a0, a1])
        if step >= END:
            break

    return {"seed": seed, "trace": trace}


def main():
    rows = [run(seed) for seed in SEEDS]
    out = {
        "schema": "adaptive-circulation-first-move-route-trace-v0",
        "actor_index": ACTOR,
        "start_step": START,
        "end_step": END,
        "cases": rows,
        "boundary": "Observation only. MOVE is not classified as spare unless the trace supports that interpretation.",
    }
    Path("adaptive_circulation_first_move_route_trace_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print("TRACE " + json.dumps(row, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
