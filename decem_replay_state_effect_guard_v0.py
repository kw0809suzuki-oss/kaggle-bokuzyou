"""DECEM Replay State-Effect Guard v0.

One-condition, one-decision probe.
At the confirmed first structural divergence (step 24), if current cash is
higher than the source replay pre-state cash (1), preserve the source replay's
realized HIRE effect by issuing one HIRE instead of three.
Everything else is delegated unchanged to the frozen DECEM replay body.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "decem_replay_distilled_157026_v0.py"

spec = importlib.util.spec_from_file_location("_decem_base_effect_guard_v0", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

trigger_count = 0

def reset_agent():
    global trigger_count
    trigger_count = 0
    if hasattr(base, "reset_agent"):
        base.reset_agent()

def agent(obs, configuration=None):
    global trigger_count
    action = base.agent(obs, configuration)
    step = int(obs.get("step", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    money = float(obs["farms"][player].get("money", 0) or 0)
    if step == 24 and money > 1:
        action["market"] = [["HIRE"]]
        trigger_count += 1
    return action
