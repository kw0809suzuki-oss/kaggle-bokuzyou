#!/usr/bin/env python3
"""Replay Fidelity Input Surface v0.

Current question:
  Does the stored fixture contain source replay configuration that the current
  recovery runner is not passing into make()?

No intervention and no terminal comparison are performed here.
"""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

PARTS = [
    "replay_realization_fixture_v0_part0.txt",
    "replay_realization_fixture_v0_part1.txt",
    "replay_realization_fixture_v0_part2.txt",
    "replay_realization_fixture_v0_part3.txt",
]
FIXTURE_SHA256 = "be97a1ae891e36a9a387a52eae314fba2ffb8f0db12b4f0174cf29c608c3b33b"


def load_fixture() -> dict[str, Any]:
    payload = "".join(Path(p).read_text(encoding="utf-8").strip() for p in PARTS)
    raw = zlib.decompress(base64.b64decode(payload))
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise RuntimeError(f"fixture sha mismatch: {got}")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    fixture = load_fixture()
    keys = sorted(fixture.keys())
    config_keys = [k for k in keys if "config" in k.lower() or "seed" in k.lower() or "version" in k.lower()]
    selected = {k: fixture.get(k) for k in config_keys}
    out = {
        "schema": "replay-fidelity-input-surface-v0",
        "fixture_keys": keys,
        "configuration_related": selected,
        "source_episode_id": fixture.get("source_episode_id"),
        "agents": fixture.get("agents"),
        "action_count": len(fixture.get("actions") or []),
        "current_runner_input": {"configuration": {"seed": fixture.get("seed")}},
        "boundary": {
            "intervention_tested": False,
            "terminal_tested": False,
            "question": "whether source replay configuration exists in fixture but is omitted by runner",
        },
    }
    Path("replay_fidelity_first_divergence_v0_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("REPLAY_FIDELITY_INPUT_SURFACE " + json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
