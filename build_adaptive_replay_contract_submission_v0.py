#!/usr/bin/env python3
"""Build a single-file Kaggle submission from the frozen Contract Runtime v0."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "adaptive_replay_contract_runtime_v0.py"
REPLAY = ROOT / "decem_replay_distilled_157026_v0.py"
WORLD = ROOT / "adaptive_replay_world_adapter_v0.py"
CONTRACTS = ROOT / "adaptive_replay_effect_contracts_v0.json"
OUT = ROOT / "main.py"

runtime = RUNTIME.read_text(encoding="utf-8")
replay_source = REPLAY.read_text(encoding="utf-8")
world_source = WORLD.read_text(encoding="utf-8")
contracts_source = CONTRACTS.read_text(encoding="utf-8")

# Validate the contract JSON before embedding it.
json.loads(contracts_source)

start_marker = "ROOT = Path(__file__).resolve().parent"
end_marker = "_contracts = ["
start = runtime.index(start_marker)
end = runtime.index(end_marker)

embedded = f'''# Single-file embedded dependencies generated from the frozen source files.
import types

_REPLAY_SOURCE = {replay_source!r}
_WORLD_SOURCE = {world_source!r}
_CONTRACT_JSON = {contracts_source!r}

replay = types.ModuleType("_adaptive_replay_contract_runtime_body")
exec(compile(_REPLAY_SOURCE, "decem_replay_distilled_157026_v0.py", "exec"), replay.__dict__)

world = types.ModuleType("_adaptive_replay_contract_world_adapter")
exec(compile(_WORLD_SOURCE, "adaptive_replay_world_adapter_v0.py", "exec"), world.__dict__)

_contract_doc = json.loads(_CONTRACT_JSON)

'''

single = runtime[:start] + embedded + runtime[end:]
OUT.write_text(single, encoding="utf-8")

print("BUILT", OUT)
print("BYTES", OUT.stat().st_size)
