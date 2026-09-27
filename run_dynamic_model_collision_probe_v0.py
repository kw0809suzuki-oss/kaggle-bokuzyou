#!/usr/bin/env python3
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEEDS = [92802801, 92802802, 92802803]

OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
INDEPENDENT_COMMIT = "eff2cf1ae6eb03809d215248f9b05b79a2b4dfe9"
SELLESTA_COMMIT = "0a5ce7ef83211d6df5824a4e7e17f3d4a40c22e7"
MODEL_COMMIT = "8136bf799412b38f612ca28c503aa77940f654f7"

INDEPENDENT_PATH = ROOT / "astra_flow_independent_distilled_v0.py"
SELLESTA_PATH = ROOT / "opponents" / "sellesta_main_pinned.py"


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


def stable(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(x):
    return hashlib.sha256(stable(x).encode("utf-8")).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def call_agent(mod, obs):
    fn = getattr(mod, "agent")
    try:
        return plain(fn(obs))
    except TypeError:
        return plain(fn(obs, None))


def shared_obs(env, seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])


def strip_execution(farm):
    return {
        "money": farm.get("money"),
        "tiles": farm.get("tiles"),
        "unlocked_quadrants": farm.get("unlocked_quadrants"),
    }


def model_view(obs, self_seat):
    other = 1 - self_seat
    farms = obs["farms"]
    private = obs.get("private") or {}
    return {
        "time": {
            "step": obs.get("step"),
            "day": obs.get("day"),
            "hour": obs.get("hour"),
            "remaining_turns": 719 - int(obs.get("step", 0)),
        },
        "cash": {
            "self": farms[self_seat].get("money"),
            "opponent": farms[other].get("money"),
        },
        "resource_placement": {
            "self_farm": strip_execution(farms[self_seat]),
            "opponent_farm": strip_execution(farms[other]),
            "self_shed": private.get("shed"),
            "self_seeds": private.get("seeds"),
        },
        "execution_means": {
            "self_farmer": farms[self_seat].get("farmer"),
            "self_hands": farms[self_seat].get("hands"),
            "self_hires_today": farms[self_seat].get("hires_today"),
            "self_inventories": private.get("inventories"),
            "opponent_farmer": farms[other].get("farmer"),
            "opponent_hands": farms[other].get("hands"),
            "opponent_hires_today": farms[other].get("hires_today"),
            "opponent_inventories": "UNKNOWN",
        },
        "shared_market": {
            "market": obs.get("market"),
            "town": obs.get("town"),
        },
    }


def normalized_actions(actions, self_seat):
    return {
        "self": actions[self_seat],
        "opponent": actions[1 - self_seat],
    }


def run_battle(seed, self_seat):
    tag = f"seed{seed}-selfseat{self_seat}"
    independent = load_module(INDEPENDENT_PATH, f"ind_{tag}")
    sellesta = load_module(SELLESTA_PATH, f"sell_{tag}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    mods = [None, None]
    labels = [None, None]
    mods[self_seat] = independent
    labels[self_seat] = "independent"
    mods[1 - self_seat] = sellesta
    labels[1 - self_seat] = "sellesta"

    turns = []
    while not env.done:
        pre = [shared_obs(env, 0), shared_obs(env, 1)]
        actions = [call_agent(mods[0], pre[0]), call_agent(mods[1], pre[1])]
        focal_pre = pre[self_seat]
        view_pre = model_view(focal_pre, self_seat)

        env.step(actions)

        post = [shared_obs(env, 0), shared_obs(env, 1)]
        focal_post = post[self_seat]
        view_post = model_view(focal_post, self_seat)

        turns.append({
            "step": int(focal_pre.get("step", len(turns))),
            "day": int(focal_pre.get("day", 0)),
            "hour": int(focal_pre.get("hour", 0)),
            "seed": seed,
            "self_seat": self_seat,
            "seat_labels": labels,
            "pre_observation": focal_pre,
            "actions_normalized": normalized_actions(actions, self_seat),
            "post_observation": focal_post,
            "model_view_pre": view_pre,
            "model_view_post": view_post,
            "pre_signature": digest(view_pre),
            "action_signature": digest(normalized_actions(actions, self_seat)),
            "post_signature": digest(view_post),
            "eligible_for_collision": int(focal_pre.get("hour", 0)) != 23,
        })

    final = shared_obs(env, self_seat)
    self_cash = float(final["farms"][self_seat]["money"])
    opp_cash = float(final["farms"][1 - self_seat]["money"])
    out = {
        "schema": "dynamic-model-collision-probe-v0-battle",
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "independent_commit": INDEPENDENT_COMMIT,
            "sellesta_commit": SELLESTA_COMMIT,
            "model_commit": MODEL_COMMIT,
        },
        "seed": seed,
        "self_seat": self_seat,
        "turn_count": len(turns),
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "terminal_margin": self_cash - opp_cash,
        "turns": turns,
    }
    path = ROOT / f"dynamic_model_probe_{seed}_seat{self_seat}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    return out


def derive_collision_view(battles):
    groups = defaultdict(list)
    for battle in battles:
        for t in battle["turns"]:
            if not t["eligible_for_collision"]:
                continue
            key = (t["pre_signature"], t["action_signature"])
            groups[key].append({
                "seed": t["seed"],
                "self_seat": t["self_seat"],
                "step": t["step"],
                "day": t["day"],
                "hour": t["hour"],
                "post_signature": t["post_signature"],
                "model_view_pre": t["model_view_pre"],
                "actions_normalized": t["actions_normalized"],
                "model_view_post": t["model_view_post"],
            })

    repeated = []
    collisions = []
    for (pre_sig, action_sig), rows in groups.items():
        if len(rows) < 2:
            continue
        post_sigs = sorted({r["post_signature"] for r in rows})
        item = {
            "pre_signature": pre_sig,
            "action_signature": action_sig,
            "occurrences": len(rows),
            "post_signature_count": len(post_sigs),
            "rows": rows,
        }
        repeated.append(item)
        if len(post_sigs) > 1:
            collisions.append(item)

    repeated.sort(key=lambda x: min((r["step"], r["seed"], r["self_seat"]) for r in x["rows"]))
    collisions.sort(key=lambda x: min((r["step"], r["seed"], r["self_seat"]) for r in x["rows"]))

    return {
        "schema": "dynamic-model-collision-probe-v0-view",
        "probe_definition": {
            "model_commit": MODEL_COMMIT,
            "exact_match_only": True,
            "excluded_transition": "hour 23 -> day boundary, because official weed/shop RNG is seed-dependent",
            "collision": "same five-record pre-view + same normalized actions, different five-record post-view",
        },
        "battle_count": len(battles),
        "transition_count": sum(len(b["turns"]) for b in battles),
        "eligible_transition_count": sum(
            1 for b in battles for t in b["turns"] if t["eligible_for_collision"]
        ),
        "repeated_exact_groups": len(repeated),
        "collision_groups": len(collisions),
        "first_repeated_group": repeated[0] if repeated else None,
        "first_collision": collisions[0] if collisions else None,
        "judge": (
            "COUNTEREXAMPLE_FOUND" if collisions else
            ("NO_COUNTEREXAMPLE_IN_EXACT_MATCHES" if repeated else "NO_EXACT_MATCH_PAIR")
        ),
    }


def main():
    battles = []
    summaries = []
    for seed in SEEDS:
        for self_seat in (0, 1):
            b = run_battle(seed, self_seat)
            battles.append(b)
            summaries.append({
                "seed": seed,
                "self_seat": self_seat,
                "terminal_self": b["terminal_self"],
                "terminal_opponent": b["terminal_opponent"],
                "terminal_margin": b["terminal_margin"],
            })

    view = derive_collision_view(battles)
    result = {
        "schema": "dynamic-model-collision-probe-v0-summary",
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "independent_commit": INDEPENDENT_COMMIT,
            "sellesta_commit": SELLESTA_COMMIT,
            "model_commit": MODEL_COMMIT,
        },
        "seeds": SEEDS,
        "battle_summaries": summaries,
        "collision_view": view,
    }
    Path("dynamic_model_probe_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("DYNAMICMODELPROBE " + json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
