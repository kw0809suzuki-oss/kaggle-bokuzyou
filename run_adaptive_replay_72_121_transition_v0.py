#!/usr/bin/env python3
"""Adaptive Replay 72->121 Transition Probe v0.

Question:
    Starting from the first observed seed-to-seed World divergence at step72,
    what observable differences appear before the candidate's first self
    Action divergence at step121?

Compare within each body:
    positive seed 93803004 vs negative seed 93803005

Bodies:
    baseline  = frozen DECEM Replay
    candidate = Adaptive Replay Contract Runtime v0

Observe only. No new Guard, Recovery, feature, or causal explanation.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
MODELS = {
    "baseline": ROOT / "decem_replay_distilled_157026_v0.py",
    "candidate": ROOT / "adaptive_replay_contract_runtime_v0.py",
}
OPPONENT = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"

POSITIVE_SEED = 93803004
NEGATIVE_SEED = 93803005
START_STEP = 72
END_STEP = 122


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


def recursive_diff(a, b, path=""):
    diffs = []

    if type(a) != type(b):
        return [{"path": path or "$", "a": a, "b": b, "kind": "type"}]

    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
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
            diffs.extend(recursive_diff(x, y, f"{path}[{i}]"))
        return diffs

    if a != b:
        diffs.append({"path": path or "$", "a": a, "b": b, "kind": "value"})

    return diffs


def run(seed, model_path, tag):
    model = load(model_path, f"{tag}_model_{seed}_{os.getpid()}")
    opponent = load(OPPONENT, f"{tag}_opponent_{seed}_{os.getpid()}")

    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    trace = {}

    while not env.done:
        obs0 = plain(shared(env, 0))
        obs1 = plain(shared(env, 1))
        step = int(obs0.get("step", 0) or 0)

        a0 = plain(model.agent(obs0))
        a1 = plain(opponent.agent(obs1))

        if START_STEP <= step <= END_STEP:
            trace[step] = {
                "world": obs0,
                "subject_action": a0,
                "opponent_action": a1,
            }

        env.step([a0, a1])

    final = plain(env.state[0].observation)
    return {
        "trace": trace,
        "terminal_self": float(final["farms"][0]["money"]),
    }


def build_timeline(pos_trace, neg_trace):
    timeline = {}
    first_seen = {}

    for step in range(START_STEP, END_STEP + 1):
        p = pos_trace.get(step)
        n = neg_trace.get(step)
        if p is None or n is None:
            continue

        world_diffs = recursive_diff(p["world"], n["world"])
        subject_action_diffs = recursive_diff(p["subject_action"], n["subject_action"])
        opponent_action_diffs = recursive_diff(p["opponent_action"], n["opponent_action"])

        paths = [d["path"] for d in world_diffs]
        new_paths = []
        for path in paths:
            if path not in first_seen:
                first_seen[path] = step
                new_paths.append(path)

        timeline[str(step)] = {
            "world_diff_count": len(world_diffs),
            "world_diff_paths": paths,
            "new_world_diff_paths": new_paths,
            "world_diffs": world_diffs,
            "subject_action_diff_count": len(subject_action_diffs),
            "subject_action_diffs": subject_action_diffs,
            "opponent_action_diff_count": len(opponent_action_diffs),
            "opponent_action_diffs": opponent_action_diffs,
        }

    return timeline, first_seen


def first_action_divergence(timeline, key):
    for step in range(START_STEP, END_STEP + 1):
        row = timeline.get(str(step))
        if row and row[key] > 0:
            return step
    return None


def main():
    data = {}

    for model_name, model_path in MODELS.items():
        pos = run(POSITIVE_SEED, model_path, f"{model_name}_pos")
        neg = run(NEGATIVE_SEED, model_path, f"{model_name}_neg")
        timeline, first_seen = build_timeline(pos["trace"], neg["trace"])

        data[model_name] = {
            "positive_terminal_self": pos["terminal_self"],
            "negative_terminal_self": neg["terminal_self"],
            "timeline": timeline,
            "first_seen_world_diff_paths": first_seen,
            "first_subject_action_divergence_step": first_action_divergence(
                timeline, "subject_action_diff_count"
            ),
            "first_opponent_action_divergence_step": first_action_divergence(
                timeline, "opponent_action_diff_count"
            ),
        }

    baseline_seen = data["baseline"]["first_seen_world_diff_paths"]
    candidate_seen = data["candidate"]["first_seen_world_diff_paths"]

    candidate_only = {
        path: step
        for path, step in candidate_seen.items()
        if path not in baseline_seen
    }
    baseline_only = {
        path: step
        for path, step in baseline_seen.items()
        if path not in candidate_seen
    }
    shared = {
        path: {
            "baseline_first_seen": baseline_seen[path],
            "candidate_first_seen": candidate_seen[path],
        }
        for path in sorted(set(baseline_seen) & set(candidate_seen))
    }

    first_candidate_only_step = (
        min(candidate_only.values()) if candidate_only else None
    )
    first_candidate_only_paths = sorted(
        path for path, step in candidate_only.items()
        if step == first_candidate_only_step
    ) if first_candidate_only_step is not None else []

    candidate_action_step = data["candidate"]["first_subject_action_divergence_step"]
    candidate_pre_action_step = (
        candidate_action_step - 1 if candidate_action_step is not None else None
    )

    compact_timeline = {}
    interesting_steps = set([START_STEP, candidate_action_step, candidate_pre_action_step])
    interesting_steps.update(candidate_only.values())
    interesting_steps.update(
        step for step in candidate_seen.values()
        if step <= (candidate_action_step or END_STEP)
    )
    for step in sorted(s for s in interesting_steps if s is not None):
        row = data["candidate"]["timeline"].get(str(step))
        if row:
            compact_timeline[str(step)] = {
                "world_diff_count": row["world_diff_count"],
                "new_world_diff_paths": row["new_world_diff_paths"],
                "subject_action_diff_count": row["subject_action_diff_count"],
                "subject_action_diffs": row["subject_action_diffs"][:20],
                "opponent_action_diff_count": row["opponent_action_diff_count"],
                "opponent_action_diffs": row["opponent_action_diffs"][:20],
            }

    summary = {
        "candidate_first_subject_action_divergence_step":
            data["candidate"]["first_subject_action_divergence_step"],
        "candidate_first_opponent_action_divergence_step":
            data["candidate"]["first_opponent_action_divergence_step"],
        "baseline_first_subject_action_divergence_step":
            data["baseline"]["first_subject_action_divergence_step"],
        "baseline_first_opponent_action_divergence_step":
            data["baseline"]["first_opponent_action_divergence_step"],
        "first_candidate_only_world_diff_step": first_candidate_only_step,
        "first_candidate_only_world_diff_paths": first_candidate_only_paths,
        "candidate_only_world_diff_first_seen": dict(
            sorted(candidate_only.items(), key=lambda kv: (kv[1], kv[0]))
        ),
        "shared_world_diff_first_seen": shared,
        "candidate_compact_timeline": compact_timeline,
    }

    out = {
        "schema": "adaptive-replay-72-121-transition-v0",
        "question": (
            "Between the shared first World divergence at step72 and the "
            "candidate self Action divergence at step121, what observable "
            "differences first appear?"
        ),
        "world": "Seyamalam v21 / subject seat0",
        "positive_seed": POSITIVE_SEED,
        "negative_seed": NEGATIVE_SEED,
        "summary": summary,
        "models": data,
        "boundary": {
            "no_new_guard": True,
            "no_recovery": True,
            "no_causal_claim": True,
            "comparison_rule": (
                "Candidate-only first-seen path means it appears in the "
                "seed04-vs-seed05 candidate comparison but not anywhere in "
                "the corresponding baseline comparison during this window."
            ),
        },
    }

    Path("adaptive_replay_72_121_transition_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("TRANSITION_72_121_SUMMARY " + json.dumps(summary, separators=(",", ":")))


if __name__ == "__main__":
    main()
