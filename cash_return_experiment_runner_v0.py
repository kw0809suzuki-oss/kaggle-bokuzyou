#!/usr/bin/env python3
"""Reusable paired runner for full-battle cash-return experiments.

Design goals:
- baseline and candidate commits/conditions are externalized in one spec;
- one completed pair is persisted immediately;
- aggregate summary is rewritten after every completed pair;
- seed groups can be sharded without changing experiment code;
- smoke mode only proves one observed cash-return cycle, not score strength.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import kaggle_environments
from kaggle_environments import make

from plan_generator_entrance_v0 import bind_official_state
from relationship_surface_body_v0 import RelationshipSurfaceBody

ROOT = Path(__file__).resolve().parent


def load_json(path: Path):
    return json.loads(path.read_text())


def atomic_write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_opponent(path: Path, tag: str):
    spec = importlib.util.spec_from_file_location(f"pinned_opponent_{tag}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise RuntimeError("pinned opponent has no agent")
    return mod


def candidate_factory(spec):
    module = importlib.import_module(spec["candidate"]["module"])
    factory = getattr(module, spec["candidate"]["factory"])
    return factory(**spec["candidate"].get("kwargs", {}))


def self_obs(env, seat=0):
    return env._Environment__get_shared_state(seat)["observation"]


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    raise TypeError(type(x).__name__)


def terminal_money(env, seat=0):
    raw = bind_official_state(env.state[seat].observation).raw()
    return float(raw["farms"][seat]["money"])


def validate_seed_groups(spec):
    groups = spec["seed_groups"]
    seen = {}
    for group, seeds in groups.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed inside group {group}")
        for seed in seeds:
            if seed in seen:
                raise ValueError(f"seed {seed} overlaps {seen[seed]} and {group}")
            seen[seed] = group


def selected_seeds(spec, group, shard_index, shard_count, max_pairs):
    seeds = list(spec["seed_groups"][group])
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard_index")
    seeds = [seed for i, seed in enumerate(seeds) if i % shard_count == shard_index]
    if max_pairs is not None:
        seeds = seeds[:max_pairs]
    return seeds


def make_world(seed):
    env = make("kaggriculture", configuration={"seed": int(seed)}, debug=False)
    env.reset(num_agents=2)
    return env


def run_smoke(spec, seed):
    candidate = candidate_factory(spec)
    opponent_path = ROOT / spec["opponent"]["local_path"]
    opponent = load_opponent(opponent_path, f"smoke_{seed}")
    env = make_world(seed)

    start_raw = bind_official_state(self_obs(env, 0)).raw()
    starting_money = float(start_raw["farms"][0]["money"])
    turns = 0

    while not env.done:
        obs0 = self_obs(env, 0)
        summary_before_step = candidate.summary() if hasattr(candidate, "summary") else {}
        if (
            summary_before_step.get("counters", {}).get("cash_return_events", 0) >= 1
        ):
            raw = bind_official_state(obs0).raw()
            return {
                "mode": "smoke",
                "seed": seed,
                "cash_return_observed": True,
                "turns_applied": turns,
                "observed_day": int(raw["day"]),
                "observed_hour": int(raw["hour"]),
                "starting_money": starting_money,
                "observed_money": float(raw["farms"][0]["money"]),
                "money_change_at_observation": float(raw["farms"][0]["money"]) - starting_money,
                "candidate_summary": summary_before_step,
                "ran_to_terminal": False,
            }

        action0 = candidate.act(obs0)
        action1 = plain(opponent.agent(self_obs(env, 1)))
        env.step([plain(action0), action1])
        turns += 1

    return {
        "mode": "smoke",
        "seed": seed,
        "cash_return_observed": False,
        "turns_applied": turns,
        "starting_money": starting_money,
        "terminal_self": terminal_money(env, 0),
        "candidate_summary": candidate.summary() if hasattr(candidate, "summary") else {},
        "ran_to_terminal": True,
    }


def run_pair(spec, seed):
    baseline = RelationshipSurfaceBody(
        policy_seed=int(spec["baseline"]["policy_seed"])
    )
    candidate = candidate_factory(spec)

    baseline_env = make_world(seed)
    candidate_env = make_world(seed)
    if baseline_env.configuration != candidate_env.configuration:
        raise RuntimeError("paired environment configuration mismatch")

    baseline_start = bind_official_state(self_obs(baseline_env, 0)).raw()
    candidate_start = bind_official_state(self_obs(candidate_env, 0)).raw()
    if bind_official_state(self_obs(baseline_env, 0)).canonical_hash != bind_official_state(self_obs(candidate_env, 0)).canonical_hash:
        raise RuntimeError("paired initial self worlds differ")

    baseline_starting_money = float(baseline_start["farms"][0]["money"])
    candidate_starting_money = float(candidate_start["farms"][0]["money"])

    opponent_path = ROOT / spec["opponent"]["local_path"]
    baseline_opp = load_opponent(opponent_path, f"baseline_{seed}")
    candidate_opp = load_opponent(opponent_path, f"candidate_{seed}")

    turns = 0
    while not baseline_env.done or not candidate_env.done:
        if baseline_env.done != candidate_env.done:
            raise RuntimeError("paired episode lengths differ")

        b_action = baseline.act(self_obs(baseline_env, 0))
        c_action = candidate.act(self_obs(candidate_env, 0))
        b_opp_action = plain(baseline_opp.agent(self_obs(baseline_env, 1)))
        c_opp_action = plain(candidate_opp.agent(self_obs(candidate_env, 1)))

        baseline_env.step([plain(b_action), b_opp_action])
        candidate_env.step([plain(c_action), c_opp_action])
        turns += 1

    if not all(s.status == "DONE" for s in baseline_env.state):
        raise RuntimeError("baseline did not finish DONE")
    if not all(s.status == "DONE" for s in candidate_env.state):
        raise RuntimeError("candidate did not finish DONE")

    baseline_terminal = terminal_money(baseline_env, 0)
    candidate_terminal = terminal_money(candidate_env, 0)

    return {
        "mode": "paired_terminal",
        "seed": seed,
        "turns": turns,
        "baseline": {
            "starting_money": baseline_starting_money,
            "terminal_self": baseline_terminal,
            "self_growth_from_start": baseline_terminal - baseline_starting_money,
        },
        "candidate": {
            "starting_money": candidate_starting_money,
            "terminal_self": candidate_terminal,
            "self_growth_from_start": candidate_terminal - candidate_starting_money,
            "summary": candidate.summary() if hasattr(candidate, "summary") else {},
        },
        "delta_terminal_self": candidate_terminal - baseline_terminal,
        "ran_to_terminal": True,
    }


def aggregate_rows(rows):
    deltas = [float(r["delta_terminal_self"]) for r in rows]
    candidate_growth = [float(r["candidate"]["self_growth_from_start"]) for r in rows]
    return {
        "pairs_completed": len(rows),
        "improved_vs_baseline": sum(x > 0 for x in deltas),
        "worse_vs_baseline": sum(x < 0 for x in deltas),
        "same_vs_baseline": sum(x == 0 for x in deltas),
        "mean_delta_terminal_self": statistics.fmean(deltas) if deltas else None,
        "median_delta_terminal_self": statistics.median(deltas) if deltas else None,
        "mean_candidate_growth_from_start": statistics.fmean(candidate_growth) if candidate_growth else None,
        "all_candidate_growth_positive": all(x > 0 for x in candidate_growth) if candidate_growth else None,
        "promotion_decision": None,
    }


def run_paired_group(spec, spec_path, group, shard_index, shard_count, max_pairs):
    code_commit = git_head()
    seeds = selected_seeds(spec, group, shard_index, shard_count, max_pairs)
    out_dir = (
        ROOT
        / "cash_return_results_v0"
        / spec["experiment_id"]
        / code_commit
        / group
        / f"shard_{shard_index}_of_{shard_count}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        row = run_pair(spec, seed)
        row["provenance"] = {
            "experiment_commit": code_commit,
            "baseline_commit": spec["baseline"]["commit"],
            "official_commit": spec["official_world"]["commit"],
            "opponent_commit": spec["opponent"]["commit"],
            "kaggle_environments_version": kaggle_environments.__version__,
            "spec_file": str(spec_path.name),
            "seed_group": group,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(out_dir / f"seed_{seed}.json", row)
        rows.append(row)

        summary = {
            "experiment_id": spec["experiment_id"],
            "group": group,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "requested_seeds": seeds,
            "completed_seeds": [r["seed"] for r in rows],
            "aggregate": aggregate_rows(rows),
            "provenance": row["provenance"],
        }
        atomic_write_json(out_dir / "summary.json", summary)
        print(
            "PAIR "
            + json.dumps(
                {
                    "seed": seed,
                    "candidate_terminal": row["candidate"]["terminal_self"],
                    "baseline_terminal": row["baseline"]["terminal_self"],
                    "delta": row["delta_terminal_self"],
                    "candidate_growth": row["candidate"]["self_growth_from_start"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return rows, out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="cash_return_wheat_v0_spec.json")
    parser.add_argument("--mode", choices=["smoke", "paired"], required=True)
    parser.add_argument("--seed-group", default="tuning")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    spec_path = ROOT / args.spec
    spec = load_json(spec_path)
    validate_seed_groups(spec)

    expected_version = spec["official_world"].get("expected_kaggle_environments_version")
    if expected_version and kaggle_environments.__version__ != expected_version:
        raise RuntimeError(
            f"kaggle-environments version mismatch: {kaggle_environments.__version__} != {expected_version}"
        )

    if args.mode == "smoke":
        seed = spec["seed_groups"]["smoke"][0]
        result = run_smoke(spec, seed)
        result["provenance"] = {
            "experiment_commit": git_head(),
            "baseline_commit": spec["baseline"]["commit"],
            "official_commit": spec["official_world"]["commit"],
            "opponent_commit": spec["opponent"]["commit"],
            "kaggle_environments_version": kaggle_environments.__version__,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        out = ROOT / "cash_return_results_v0" / spec["experiment_id"] / git_head() / "smoke"
        atomic_write_json(out / f"seed_{seed}.json", result)
        print("SMOKE " + json.dumps(result, ensure_ascii=False), flush=True)
        if not result["cash_return_observed"]:
            raise SystemExit("smoke failed: no observed cash return")
        return

    rows, out_dir = run_paired_group(
        spec,
        spec_path,
        args.seed_group,
        args.shard_index,
        args.shard_count,
        args.max_pairs,
    )
    print(
        "SUMMARY "
        + json.dumps(
            {
                "output_dir": str(out_dir.relative_to(ROOT)),
                "aggregate": aggregate_rows(rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
