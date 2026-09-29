#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

from kaggle_environments import make

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from strong_model_v0_reimplementation.agent import agent as strong_agent, reset_agent as reset_strong

SEED=92804001
SEAT=0
OPPONENT_PATH=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
OUT=Path(__file__).with_name("initial_hire2_v0_result.json")


def _load_opponent():
    spec=importlib.util.spec_from_file_location("initial_hire2_opponent",OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {OPPONENT_PATH}")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset(obj:Any)->None:
    fn=getattr(obj,"reset_agent",None)
    if callable(fn):
        fn()


def _money(final_state:list[Any],seat:int)->float:
    obs=final_state[seat].observation
    farms=obs["farms"] if isinstance(obs,dict) else obs.farms
    farm=farms[seat]
    return float(farm["money"] if isinstance(farm,dict) else farm.money)


class InitialHire2:
    def __init__(self):
        self.first_action=None

    def reset(self):
        self.first_action=None
        reset_strong()

    def agent(self,obs,configuration):
        action=copy.deepcopy(strong_agent(obs,configuration))
        if int(obs["step"])==0:
            action.setdefault("market",[])
            action["market"].extend([["HIRE"],["HIRE"]])
            self.first_action=copy.deepcopy(action)
        return action


def _run(label:str,agent_fn:Callable,reset_fn:Callable,first_action_getter=None)->dict[str,Any]:
    opponent=_load_opponent()
    reset_fn()
    _reset(opponent)

    env=make("kaggriculture",configuration={"seed":SEED},debug=False)
    agents=[None,None]
    agents[SEAT]=agent_fn
    agents[1-SEAT]=opponent.agent
    env.run(agents)

    self_cash=_money(env.state,SEAT)
    opp_cash=_money(env.state,1-SEAT)
    row={
        "label":label,
        "seed":SEED,
        "seat":SEAT,
        "terminal_self":self_cash,
        "terminal_opponent":opp_cash,
        "margin":self_cash-opp_cash,
        "status":str(env.state[SEAT].status),
    }
    if first_action_getter is not None:
        row["step0_action"]=first_action_getter()
    return row


def main():
    baseline=_run("strong_baseline",strong_agent,reset_strong)
    candidate=InitialHire2()
    hire2=_run("strong_plus_step0_hire2",candidate.agent,candidate.reset,lambda:candidate.first_action)

    result={
        "schema":"strong-initial-hire2-v0",
        "purpose":"Direct one-point Battle test: append HIRE x2 only to Strong step0 market action; all later decisions are native Strong on the resulting Official World state.",
        "conditions":{
            "seed":SEED,
            "seat":SEAT,
            "opponent":"Seyamalam pinned v21",
            "official_world_commit":"d7729da06cc1382eb742d6980dc3180aa85caa28",
        },
        "baseline":baseline,
        "candidate":hire2,
        "terminal_self_delta":hire2["terminal_self"]-baseline["terminal_self"],
        "terminal_margin_delta":hire2["margin"]-baseline["margin"],
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))


if __name__=="__main__":
    main()
