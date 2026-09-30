#!/usr/bin/env python3
"""Replay Fidelity First Divergence v0.

Purpose:
  Compare the recorded source episode's realized state stream against a local
  re-run driven by the recorded requested action stream.

Boundary:
  No intervention is evaluated here.
  This probe only identifies the first realized-state divergence.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from kaggle_environments import get_episode_replay, make

SOURCE_EPISODE_ID = 115521888
PARTS = [
    "replay_realization_fixture_v0_part0.txt",
    "replay_realization_fixture_v0_part1.txt",
    "replay_realization_fixture_v0_part2.txt",
    "replay_realization_fixture_v0_part3.txt",
]
FIXTURE_SHA256 = "be97a1ae891e36a9a387a52eae314fba2ffb8f0db12b4f0174cf29c608c3b33b"


def plain(x: Any) -> Any:
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


def load_fixture() -> dict[str, Any]:
    payload = "".join(Path(p).read_text(encoding="utf-8").strip() for p in PARTS)
    raw = zlib.decompress(base64.b64decode(payload))
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise RuntimeError(f"fixture sha mismatch: {got}")
    return json.loads(raw.decode("utf-8"))


def extract_replay_object(response: Any) -> dict[str, Any]:
    queue = [response]
    while queue:
        x = queue.pop(0)
        if isinstance(x, str):
            try:
                y = json.loads(x)
            except Exception:
                continue
            queue.append(y)
            continue
        if isinstance(x, dict):
            if isinstance(x.get("steps"), list):
                return x
            # Kaggle API commonly returns a replay field containing serialized JSON.
            if "replay" in x:
                queue.insert(0, x["replay"])
            queue.extend(v for k, v in x.items() if k != "replay")
        elif isinstance(x, list):
            queue.extend(x)
    raise RuntimeError("could not locate replay object with steps in GetEpisodeReplay response")


def shared(env, seat: int) -> dict[str, Any]:
    return plain(env._Environment__get_shared_state(seat)["observation"])


def subject_snapshot(obs: dict[str, Any]) -> dict[str, Any]:
    farms = obs.get("farms") or []
    farm = farms[0] if farms else {}
    priv = obs.get("private") or {}
    return {
        "step": int(obs.get("step", 0) or 0),
        "cash": float(farm.get("money", 0) or 0),
        "wheat_seed": int((priv.get("seeds") or {}).get("WHEAT", 0) or 0),
        "shed_wheat": int((priv.get("shed") or {}).get("WHEAT", 0) or 0),
    }


def top_level_diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def source_observation(source_steps: list[Any], index: int, seat: int = 0) -> dict[str, Any] | None:
    if index < 0 or index >= len(source_steps):
        return None
    row = source_steps[index]
    if not isinstance(row, list) or seat >= len(row):
        return None
    cell = row[seat]
    if not isinstance(cell, dict):
        return None
    obs = cell.get("observation")
    return plain(obs) if isinstance(obs, dict) else None


def compare_with_offset(fixture: dict[str, Any], source_steps: list[Any], offset: int) -> dict[str, Any]:
    env = make("kaggriculture", configuration={"seed": fixture["seed"]}, debug=False)
    env.reset(num_agents=2)

    first = None
    checked = 0

    while not env.done:
        step = int(shared(env, 0).get("step", 0) or 0)
        local_obs = shared(env, 0)
        src_obs = source_observation(source_steps, step + offset, 0)

        if src_obs is not None:
            checked += 1
            if first is None and local_obs != src_obs:
                first = {
                    "local_action_step": step,
                    "source_step_index": step + offset,
                    "top_level_diff_keys": top_level_diff(src_obs, local_obs),
                    "source_subject": subject_snapshot(src_obs),
                    "local_subject": subject_snapshot(local_obs),
                }
                break

        if step >= len(fixture["actions"]):
            break
        a0 = deepcopy(fixture["actions"][step][0])
        a1 = deepcopy(fixture["actions"][step][1])
        env.step([a0, a1])

    return {
        "offset": offset,
        "checked_comparable_states": checked,
        "first_divergence": first,
    }


def main() -> None:
    fixture = load_fixture()
    response = get_episode_replay(SOURCE_EPISODE_ID)
    replay = extract_replay_object(response)
    source_steps = replay["steps"]

    # Check the two plausible index conventions explicitly rather than assuming one.
    comparisons = [
        compare_with_offset(fixture, source_steps, 0),
        compare_with_offset(fixture, source_steps, 1),
    ]

    out = {
        "schema": "replay-fidelity-first-divergence-v0",
        "source_episode_id": SOURCE_EPISODE_ID,
        "fixture_seed": fixture["seed"],
        "source_step_count": len(source_steps),
        "fixture_action_count": len(fixture["actions"]),
        "comparisons": comparisons,
        "boundary": {
            "intervention_tested": False,
            "requested_action_stream_changed": False,
            "purpose": "identify first realized-state divergence only",
        },
    }

    Path("replay_fidelity_first_divergence_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("REPLAY_FIDELITY_FIRST_DIVERGENCE " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
