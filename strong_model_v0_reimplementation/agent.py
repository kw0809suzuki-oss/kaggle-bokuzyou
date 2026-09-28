"""Kaggle callable for the Strong Model v0 reimplementation."""
from __future__ import annotations

import copy
from typing import Any

from plan_generator_entrance_v0 import bind_official_state

from .jobs import fresh_jobs, still_needed
from .planner import choose, settings_from, step_of


class Runtime:
    def __init__(self,cfg):
        self.cfg=cfg
        self.active={}
        self.last_step=-1
        self.last_choice=None

    def reset(self):
        self.active={}
        self.last_step=-1
        self.last_choice=None

    def act(self,obs:Any):
        snapshot=bind_official_state(obs)
        raw=snapshot.raw()
        raw["step"]=step_of(raw,self.cfg)
        step=int(raw["step"])
        if step<=self.last_step:
            self.reset()

        self.active={
            key:spec for key,spec in self.active.items()
            if still_needed(spec,raw)
        }

        chosen,chosen_rep,active_jobs,continuation=choose(raw,self.cfg,self.active)

        self.active={spec["key"]:spec for spec in chosen.commitments}

        self.last_choice={
            "step":step,
            "active_before":[j.key for j in active_jobs],
            "continuation":{
                "strict_terminal_cash":continuation.envelope.strict_cash,
                "central_terminal_cash":continuation.envelope.central_cash,
                "scheduled":list(continuation.scheduled),
            },
            "chosen":{
                "strict_terminal_cash":chosen.envelope.strict_cash,
                "central_terminal_cash":chosen.envelope.central_cash,
                "scheduled":list(chosen.scheduled),
                "representative":None if chosen_rep is None else chosen_rep.spec(),
            },
            "active_after":copy.deepcopy(self.active),
        }
        self.last_step=step
        return copy.deepcopy(chosen.action)


_RUNTIMES={}


def reset_agent():
    _RUNTIMES.clear()


def agent(obs,configuration=None):
    cfg=settings_from(configuration)
    seat=int(obs["player"] if isinstance(obs,dict) else obs.player)
    runtime=_RUNTIMES.get(seat)
    if runtime is None or runtime.cfg!=cfg:
        runtime=Runtime(cfg)
        _RUNTIMES[seat]=runtime
    return runtime.act(obs)


def debug_state(seat=0):
    rt=_RUNTIMES.get(int(seat))
    if rt is None:return None
    return {
        "active_jobs":copy.deepcopy(rt.active),
        "last_choice":copy.deepcopy(rt.last_choice),
    }
