#!/usr/bin/env python3
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SEED = int(os.environ["SEED"])

OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"
INDEPENDENT_COMMIT = "eff2cf1ae6eb03809d215248f9b05b79a2b4dfe9"
SELLESTA_COMMIT = "0a5ce7ef83211d6df5824a4e7e17f3d4a40c22e7"

MODELS = {
    "baseline": ROOT / "astra_flow_independent_distilled_v0.py",
    "recovery_v0": ROOT / "mature_melon_recovery_v0.py",
}
BASELINE = ROOT / "astra_flow_independent_distilled_v0.py"
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


def snapshot(env):
    o0 = plain(env.state[0].observation)
    o1 = plain(env.state[1].observation)
    return {
        "public": public_state(o0),
        "private_seat0": plain(o0.get("private", {}) or {}),
        "private_seat1": plain(o1.get("private", {}) or {}),
    }


def crop_count(farm, crop):
    return sum(
        1
        for row in (farm.get("tiles", []) or [])
        for tile in row
        if isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == crop
    )


def worker_positions(farm):
    return [farm.get("farmer"), *(farm.get("hands", []) or [])]


def worker_actions(action):
    return [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]


def intervention_rows(pre_obs, subject_action, baseline_action, seat):
    if subject_action == baseline_action:
        return []

    farm = pre_obs["farms"][seat]
    positions = worker_positions(farm)
    actual = worker_actions(subject_action)
    reference = worker_actions(baseline_action)
    rows = []
    for idx, (pos, a, b) in enumerate(zip(positions, actual, reference)):
        if a == b:
            continue
        tile = None
        if pos and len(pos) == 2:
            x, y = int(pos[0]), int(pos[1])
            tile = plain(farm["tiles"][y][x])
        rows.append({
            "worker_index": idx,
            "position": plain(pos),
            "baseline_action": plain(b),
            "candidate_action": plain(a),
            "tile_pre": tile,
        })

    if subject_action.get("market", []) != baseline_action.get("market", []):
        rows.append({
            "worker_index": None,
            "position": None,
            "baseline_action": plain(baseline_action.get("market", [])),
            "candidate_action": plain(subject_action.get("market", [])),
            "tile_pre": None,
            "surface": "market",
        })
    return rows


def infer_melon_harvest_units(turn, seat):
    farm = turn["pre"]["public"]["farms"][seat]
    total = 0
    rows = []
    for idx, (action, pos) in enumerate(zip(
        worker_actions(turn[f"action_seat{seat}"]),
        worker_positions(farm),
    )):
        if action != ["HARVEST"] or not pos or len(pos) != 2:
            continue
        x, y = int(pos[0]), int(pos[1])
        tile = farm["tiles"][y][x]
        if not (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and tile.get("crop") == "MELON"
        ):
            continue
        day = int(turn["day"])
        raw_planted_day = tile.get("planted_day")
        planted_day = day if raw_planted_day is None else int(raw_planted_day)
        units = int(tile.get("yield_units", 0) or 0)
        if units > 0 and day - planted_day >= 10:
            total += units
            rows.append({
                "step": int(turn["step"]),
                "worker_index": idx,
                "x": x,
                "y": y,
                "units": units,
            })
    return total, rows


def private_total(private, item):
    total = int((private.get("shed", {}) or {}).get(item, 0) or 0)
    for inv in private.get("inventories", []) or []:
        total += int((inv or {}).get(item, 0) or 0)
    return total


def infer_melon_sell_units(turn, seat, harvested_units):
    market = turn[f"action_seat{seat}"].get("market", []) or []
    requested = sum(
        int(order[2])
        for order in market
        if isinstance(order, list)
        and len(order) >= 3
        and order[0] == "SELL"
        and order[1] == "MELON"
    )
    if requested <= 0:
        return 0

    pre = private_total(turn["pre"][f"private_seat{seat}"], "MELON")
    post = private_total(turn["post"][f"private_seat{seat}"], "MELON")
    delta = pre + harvested_units - post
    if 0 <= delta <= requested:
        return int(delta)
    return None


def run_game(label, subject_seat):
    subject = load_module(MODELS[label], f"{label}_{SEED}_{subject_seat}")
    sellesta = load_module(SELLESTA, f"sellesta_{label}_{SEED}_{subject_seat}")
    baseline_ref = load_module(BASELINE, f"baseline_ref_{label}_{SEED}_{subject_seat}")

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    mods = [None, None]
    mods[subject_seat] = subject
    mods[1 - subject_seat] = sellesta

    turns = []
    while not env.done:
        pre0 = plain(shared_obs(env, 0))
        pre1 = plain(shared_obs(env, 1))
        pre = snapshot(env)
        step = int(pre["public"].get("step", len(turns)))
        day = int(pre["public"].get("day", 0))
        hour = int(pre["public"].get("hour", 0))

        actions = [
            plain(mods[0].agent(pre0)),
            plain(mods[1].agent(pre1)),
        ]

        subject_obs = pre0 if subject_seat == 0 else pre1
        baseline_reference_action = plain(baseline_ref.agent(subject_obs))
        intervention = intervention_rows(
            pre["public"],
            actions[subject_seat],
            baseline_reference_action,
            subject_seat,
        ) if label == "recovery_v0" else []

        env.step(actions)
        post = snapshot(env)

        turns.append({
            "step": step,
            "day": day,
            "hour": hour,
            "subject_label": label,
            "subject_seat": subject_seat,
            "pre": pre,
            "action_seat0": actions[0],
            "action_seat1": actions[1],
            "post": post,
            "baseline_reference_action": baseline_reference_action if label == "recovery_v0" else None,
            "intervention": intervention,
        })

    battle_id = f"{label}-vs-sellesta-seat{subject_seat}-seed{SEED}"
    battle_path = Path(f"recovery_v0_{SEED}_{label}_seat{subject_seat}_battle.json.gz")
    battle = {
        "schema": "kaggriculture-battle-evidence-v1",
        "battle_id": battle_id,
        "provenance": {
            "official_commit": OFFICIAL_COMMIT,
            "independent_commit": INDEPENDENT_COMMIT,
            "sellesta_commit": SELLESTA_COMMIT,
            "experiment_git_sha": os.environ.get("GITHUB_SHA"),
            "seed": SEED,
            "subject_label": label,
            "subject_seat": subject_seat,
        },
        "turn_count": len(turns),
        "turns": turns,
    }
    with gzip.open(battle_path, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(battle, fh, ensure_ascii=False, separators=(",", ":"))

    # Views are derived only after the complete Battle is written.
    intervention_events = [
        {
            "step": t["step"],
            "day": t["day"],
            "hour": t["hour"],
            "rows": t["intervention"],
        }
        for t in turns
        if t["intervention"]
    ]

    harvest_units = 0
    harvest_events = []
    sell_units = 0
    sell_inference_failures = []
    for turn in turns:
        units, rows = infer_melon_harvest_units(turn, subject_seat)
        harvest_units += units
        harvest_events.extend(rows)
        sold = infer_melon_sell_units(turn, subject_seat, units)
        if sold is None:
            sell_inference_failures.append(int(turn["step"]))
        else:
            sell_units += sold

    final = turns[-1]["post"]["public"]
    self_cash = float(final["farms"][subject_seat]["money"])
    opp_cash = float(final["farms"][1 - subject_seat]["money"])

    first = turns[0]["pre"]["public"]["farms"][subject_seat]
    return {
        "battle_id": battle_id,
        "battle_file": battle_path.name,
        "turn_count": len(turns),
        "initial_melon_count": crop_count(first, "MELON"),
        "intervention_turn_count": len(intervention_events),
        "intervention_worker_count": sum(len(x["rows"]) for x in intervention_events),
        "intervention_events": intervention_events,
        "melon_harvest_units": harvest_units,
        "melon_harvest_events": harvest_events,
        "melon_sell_units": sell_units,
        "melon_sell_inference_failure_steps": sell_inference_failures,
        "terminal_self": self_cash,
        "terminal_sellesta": opp_cash,
        "margin": self_cash - opp_cash,
    }


def main():
    results = []
    for seat in (0, 1):
        results.append(run_game("baseline", seat))
        results.append(run_game("recovery_v0", seat))

    paired = []
    for seat in (0, 1):
        base = next(r for r in results if r["battle_id"].startswith("baseline-") and f"seat{seat}-" in r["battle_id"])
        cand = next(r for r in results if r["battle_id"].startswith("recovery_v0-") and f"seat{seat}-" in r["battle_id"])
        paired.append({
            "seat": seat,
            "intervention_turn_count": cand["intervention_turn_count"],
            "intervention_worker_count": cand["intervention_worker_count"],
            "melon_harvest_delta": cand["melon_harvest_units"] - base["melon_harvest_units"],
            "melon_sell_units_delta": cand["melon_sell_units"] - base["melon_sell_units"],
            "terminal_self_delta": cand["terminal_self"] - base["terminal_self"],
            "margin_delta": cand["margin"] - base["margin"],
        })

    out = {
        "schema": "mature-melon-recovery-v0-summary",
        "seed": SEED,
        "note": "Battle files are Evidence. Summary is a derived View after full 719-turn completion.",
        "results": results,
        "paired": paired,
    }
    Path(f"recovery_v0_seed{SEED}_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("RECOVERY_V0 " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
