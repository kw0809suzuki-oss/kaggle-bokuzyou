#!/usr/bin/env python3
"""Static/invariant checks for One-Shot View Lens Harness v0."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import one_shot_view_lenses_v0 as lenses
from one_shot_view_lens_harness_v0 import VALID_LENSES
from plan_generator_entrance_v0 import bind_official_state

PROMOTED_BODY_BLOB_SHA = "09c8685b582e3e13fa8e58f0c6089ffb2a4f4c38"

def git_blob_sha(path):
    data = Path(path).read_bytes()
    payload = f"blob {len(data)}\0".encode("utf-8") + data
    return hashlib.sha1(payload).hexdigest()

def candidate(cid, kind, target):
    return SimpleNamespace(candidate_id=cid, kind=kind, target=target)

def minimal_snapshot():
    raw = {
        "player": 0,
        "day": 5,
        "hour": 3,
        "farms": [
            {
                "money": 100,
                "farmer": [0, 0],
                "hands": [],
                "tiles": [[None, None], [None, None]],
            },
            {
                "money": 100,
                "farmer": [1, 1],
                "hands": [],
                "tiles": [[None, None], [None, None]],
            },
        ],
        "private": {
            "seeds": {"CARROT": 0},
            "inventories": [{}],
            "shed": {},
        },
        "market": {"prices": {}},
        "town": {},
    }
    return bind_official_state(raw)

def main():
    snapshot = minimal_snapshot()

    # Duplicate Candidate on the same tile must not inflate nearby target count.
    n1 = candidate("n1", "establish_plant", {"crop": "CARROT", "tile": [0, 0]})
    n2 = candidate("n2", "prepare_surface_for_plant", {"crop": "CARROT", "tile": [0, 0]})
    n3 = candidate("n3", "establish_plant", {"crop": "CARROT", "tile": [1, 0]})
    nearby, _ = lenses._nearby(snapshot, [n1, n2, n3])
    nearby_ok = nearby == {"n1": 1, "n2": 1, "n3": 1}

    # Three Candidates but only two distinct target tiles share seed destination.
    e1 = candidate("e1", "prepare_for_plant", {"crop": "CARROT", "tile": [0, 0]})
    e2 = candidate("e2", "prepare_for_plant", {"crop": "CARROT", "tile": [0, 0]})
    e3 = candidate("e3", "prepare_for_plant", {"crop": "CARROT", "tile": [1, 0]})
    exit_obs, _ = lenses._exit(snapshot, [e1, e2, e3])
    exit_ok = exit_obs == {"e1": 2, "e2": 2, "e3": 2}

    body_sha = git_blob_sha("relationship_surface_body_v0.py")
    body_frozen = body_sha == PROMOTED_BODY_BLOB_SHA

    registry_ok = set(VALID_LENSES) == {
        "nearby_density",
        "relationship_age",
        "exit_proximity",
        "disabled",
    }

    result = {
        "schema": "one-shot-view-lens-harness-acceptance-v0",
        "frozen_body_blob_sha_expected": PROMOTED_BODY_BLOB_SHA,
        "frozen_body_blob_sha_actual": body_sha,
        "frozen_body_unchanged": body_frozen,
        "distinct_object_nearby_no_inflation": nearby_ok,
        "distinct_object_exit_no_inflation": exit_ok,
        "single_harness_registry_has_three_lenses_plus_disabled": registry_ok,
        "passed": all([body_frozen, nearby_ok, exit_ok, registry_ok]),
    }

    Path("one_shot_view_lens_harness_acceptance_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY " + json.dumps(result, separators=(",", ":")))
    if not result["passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
