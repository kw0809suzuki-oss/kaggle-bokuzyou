"""DECEM Replay State-Effect Guard v1.

v0: preserve the source realized HIRE effect at step 24.
v1: additionally preserve the source realized no-purchase effect at step 85
when the current world would make BUY_SEED STRAWBERRY 1 succeed.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"decem_replay_state_effect_guard_v0.py"

spec=importlib.util.spec_from_file_location("_decem_guard_v0_for_v1",BASE)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

trigger_step24=0
trigger_step85=0

def reset_agent():
    global trigger_step24, trigger_step85
    trigger_step24=0
    trigger_step85=0
    if hasattr(base,"reset_agent"):
        base.reset_agent()

def agent(obs,configuration=None):
    global trigger_step24, trigger_step85
    before=int(getattr(base,"trigger_count",0))
    action=base.agent(obs,configuration)
    after=int(getattr(base,"trigger_count",0))
    if after>before:
        trigger_step24 += after-before

    step=int(obs.get("step",0) or 0)
    player=int(obs.get("player",0) or 0)
    money=float(obs["farms"][player].get("money",0) or 0)
    if step==85 and money>=100:
        market=list(action.get("market",[]) or [])
        filtered=[o for o in market if not (isinstance(o,list) and len(o)>=3 and o[0]=="BUY_SEED" and o[1]=="STRAWBERRY")]
        if len(filtered)!=len(market):
            action["market"]=filtered
            trigger_step85 += 1
    return action
