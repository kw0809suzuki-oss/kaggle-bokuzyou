#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from strong_model_v0_reimplementation.agent import agent, reset_agent, debug_state

SEED=92804001
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

spec=importlib.util.spec_from_file_location("hire_pulse_opponent",OPPONENT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reset_agent()
if hasattr(module,"reset_agent"): module.reset_agent()

env=make("kaggriculture",configuration={"seed":SEED},debug=False)
env.run([agent,module.agent])
obs=env.state[0].observation
farms=obs["farms"] if isinstance(obs,dict) else obs.farms
farm=farms[0]
money=float(farm["money"] if isinstance(farm,dict) else farm.money)
result={"seed":SEED,"seat":0,"terminal_self":money,"debug":debug_state(0)}
print(json.dumps(result,ensure_ascii=False,sort_keys=True))
