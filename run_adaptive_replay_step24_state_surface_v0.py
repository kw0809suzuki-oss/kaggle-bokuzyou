#!/usr/bin/env python3
"""Step24 State Surface Probe v0.

Question:
    Boundary Observation v0 found the same Contract input can lead to
    positive or negative terminal response. Is that boundary already visible
    in the observable step24 World State?

Scope:
    Seyamalam v21, seat0, fresh seeds 93803001..93803005.
    Seed 93803005 is the observed negative case in Boundary Observation v0.
    Seeds 93803001..04 are observed positive cases.

This probe does not change actions and does not add a Guard.
It only captures the full observable pre-step24 state and compares surfaces.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SUBJECT = ROOT / "decem_replay_distilled_157026_v0.py"
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"
BOUNDARY_RESULT = ROOT / "experiments" / "adaptive_replay_boundary_observation_v0_result.json"

SEEDS = [93803001, 93803002, 93803003, 93803004, 93803005]
NEGATIVE_SEED = 93803005


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


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "reset_agent"):
        mod.reset_agent()
    return mod


def shared(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_to_step24(seed):
    subject = load(SUBJECT, f"surface_subject_{seed}_{os.getpid()}")
    opponent = load(OPPONENT, f"surface_opponent_{seed}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        obs0 = plain(shared(env, 0))
        if int(obs0.get("step", 0) or 0) == 24:
            return obs0

        a0 = plain(subject.agent(obs0))
        a1 = plain(opponent.agent(shared(env, 1)))
        env.step([a0, a1])

    raise RuntimeError(f"step24 not reached for seed {seed}")


def recursive_diff(a, b, path=""):
    diffs = []

    if type(a) != type(b):
        diffs.append({
            "path": path or "$",
            "a": a,
            "b": b,
            "kind": "type",
        })
        return diffs

    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        for key in keys:
            p = f"{path}.{key}" if path else str(key)
            if key not in a:
                diffs.append({"path": p, "a": "<missing>", "b": b[key], "kind": "missing_a"})
            elif key not in b:
                diffs.append({"path": p, "a": a[key], "b": "<missing>", "kind": "missing_b"})
            else:
                diffs.extend(recursive_diff(a[key], b[key], p))
        return diffs

    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append({
                "path": (path or "$") + ".length",
                "a": len(a),
                "b": len(b),
                "kind": "length",
            })
        for i, (x, y) in enumerate(zip(a, b)):
            p = f"{path}[{i}]"
            diffs.extend(recursive_diff(x, y, p))
        return diffs

    if a != b:
        diffs.append({
            "path": path or "$",
            "a": a,
            "b": b,
            "kind": "value",
        })
    return diffs


def root_bucket(path):
    # Keep enough structure to separate self/opponent farms and major surfaces.
    if path.startswith("farms[0]") or path.startswith("farms.0"):
        return "farms.self"
    if path.startswith("farms[1]") or path.startswith("farms.1"):
        return "farms.opponent"
    return path.split(".", 1)[0].split("[", 1)[0]


def main():
    boundary = json.loads(BOUNDARY_RESULT.read_text(encoding="utf-8"))
    observed_delta = {}
    for row in boundary["cases"]:
        if row["opponent"] == "seyamalam_v21" and row["subject_seat"] == 0:
            observed_delta[int(row["seed"])] = float(row["delta_terminal_self"])

    states = {seed: run_to_step24(seed) for seed in SEEDS}

    per_seed = {}
    for seed, state in states.items():
        section_hashes = {}
        for key, value in sorted(state.items()):
            section_hashes[key] = digest(value)
        per_seed[str(seed)] = {
            "terminal_delta_from_boundary_probe": observed_delta.get(seed),
            "full_state_sha256": digest(state),
            "section_hashes": section_hashes,
        }

    negative_state = states[NEGATIVE_SEED]
    comparisons = {}
    common_diff_paths = None

    for seed in SEEDS:
        if seed == NEGATIVE_SEED:
            continue
        diffs = recursive_diff(states[seed], negative_state)
        buckets = Counter(root_bucket(d["path"]) for d in diffs)
        paths = [d["path"] for d in diffs]

        if common_diff_paths is None:
            common_diff_paths = set(paths)
        else:
            common_diff_paths &= set(paths)

        comparisons[f"{seed}_vs_{NEGATIVE_SEED}"] = {
            "positive_seed": seed,
            "positive_terminal_delta": observed_delta.get(seed),
            "negative_seed": NEGATIVE_SEED,
            "negative_terminal_delta": observed_delta.get(NEGATIVE_SEED),
            "full_state_equal": len(diffs) == 0,
            "diff_leaf_count": len(diffs),
            "diff_buckets": dict(buckets),
            "sample_diffs": diffs[:80],
        }

    all_hashes = [per_seed[str(seed)]["full_state_sha256"] for seed in SEEDS]

    # Which top-level state sections vary at all across the five seeds?
    section_values = defaultdict(set)
    for seed in SEEDS:
        for key, h in per_seed[str(seed)]["section_hashes"].items():
            section_values[key].add(h)
    variable_sections = {
        key: len(values)
        for key, values in sorted(section_values.items())
        if len(values) > 1
    }

    summary = {
        "n_seeds": len(SEEDS),
        "unique_full_step24_states": len(set(all_hashes)),
        "variable_top_level_sections": variable_sections,
        "negative_seed": NEGATIVE_SEED,
        "negative_terminal_delta": observed_delta.get(NEGATIVE_SEED),
        "positive_seeds": [s for s in SEEDS if s != NEGATIVE_SEED],
        "positive_terminal_deltas": {
            str(s): observed_delta.get(s) for s in SEEDS if s != NEGATIVE_SEED
        },
        "common_diff_path_count_positive_vs_negative": len(common_diff_paths or set()),
        "common_diff_path_sample": sorted(common_diff_paths or set())[:80],
    }

    out = {
        "schema": "adaptive-replay-step24-state-surface-v0",
        "question": (
            "Is the positive/negative terminal boundary already visible in "
            "the observable pre-step24 World State?"
        ),
        "world": "Seyamalam v21 / subject seat0",
        "summary": summary,
        "per_seed": per_seed,
        "comparisons": comparisons,
        "boundary": {
            "no_action_change": True,
            "no_new_guard": True,
            "no_causal_interpretation": True,
            "negative_case_count": 1,
            "interpretation_limit": (
                "A state difference is only an observed candidate boundary, "
                "not evidence that the differing field causes terminal sign."
            ),
        },
    }

    Path("adaptive_replay_step24_state_surface_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("STEP24_STATE_SURFACE_SUMMARY " + json.dumps(summary, separators=(",", ":")))
    for key, value in comparisons.items():
        compact = {
            "pair": key,
            "full_state_equal": value["full_state_equal"],
            "diff_leaf_count": value["diff_leaf_count"],
            "diff_buckets": value["diff_buckets"],
            "sample_paths": [d["path"] for d in value["sample_diffs"][:30]],
        }
        print("STEP24_STATE_SURFACE_COMPARE " + json.dumps(compact, separators=(",", ":")))


if __name__ == "__main__":
    main()
