#!/usr/bin/env python3
import importlib.util
import json
import sys
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "independent_day0_melon12_v0.py"
SELLESTA = ROOT / "opponents" / "sellesta_main_pinned.py"
SEED = 92801801


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


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def crop_count(farm, crop):
    n = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop:
                n += 1
    return n


def plant_requests(action, crop):
    unit_actions = [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]
    return [a for a in unit_actions if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] == crop]


def main():
    candidate = load(CANDIDATE, "candidate_diag")
    sellesta = load(SELLESTA, "sellesta_diag")
    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    captures = []
    while not env.done:
        obs0 = plain(shared_obs(env, 0))
        obs1 = plain(shared_obs(env, 1))
        a0 = plain(candidate.agent(obs0))
        a1 = plain(sellesta.agent(obs1))

        day = int(obs0.get("day", 0))
        hour = int(obs0.get("hour", 0))
        if day == 0 and hour in (20, 21, 22, 23):
            priv = obs0.get("private", {}) or {}
            farm = obs0["farms"][0]
            captures.append({
                "phase": "pre",
                "day": day,
                "hour": hour,
                "step": int(obs0.get("step", 0)),
                "melon_seeds": int((priv.get("seeds", {}) or {}).get("MELON", 0) or 0),
                "wheat_seeds": int((priv.get("seeds", {}) or {}).get("WHEAT", 0) or 0),
                "melon_plants": crop_count(farm, "MELON"),
                "wheat_plants": crop_count(farm, "WHEAT"),
                "melon_plant_requests": len(plant_requests(a0, "MELON")),
                "wheat_plant_requests": len(plant_requests(a0, "WHEAT")),
                "action": a0,
            })

        env.step([a0, a1])

        post = plain(env.state[0].observation)
        if day == 0 and hour in (20, 21, 22, 23):
            priv = post.get("private", {}) or {}
            farm = post["farms"][0]
            captures.append({
                "phase": "post",
                "day": int(post.get("day", 0)),
                "hour": int(post.get("hour", 0)),
                "step": int(post.get("step", 0)),
                "source_hour": hour,
                "melon_seeds": int((priv.get("seeds", {}) or {}).get("MELON", 0) or 0),
                "wheat_seeds": int((priv.get("seeds", {}) or {}).get("WHEAT", 0) or 0),
                "melon_plants": crop_count(farm, "MELON"),
                "wheat_plants": crop_count(farm, "WHEAT"),
            })

        if int(post.get("day", 0)) >= 1:
            break

    out = {"seed": SEED, "captures": captures}
    Path("melon12_h22_diagnostic.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("MELON12_DIAG " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
