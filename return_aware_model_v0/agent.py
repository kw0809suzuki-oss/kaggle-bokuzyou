"""Stateful Kaggle entry point, with commitments retained between observations."""
from __future__ import annotations

import copy

from plan_generator_entrance_v0 import bind_official_state
from strong_model_v0_reimplementation.jobs import still_needed
from strong_model_v0_reimplementation.planner import settings_from, step_of
from .model import choose


class Runtime:
    def __init__(self, cfg, max_candidates=6):
        if max_candidates < 0:
            raise ValueError("max_candidates must be nonnegative")
        self.cfg = cfg
        self.max_candidates = max_candidates
        self.reset()

    def reset(self):
        self.active = {}
        self.last_step = -1
        self.last_choice = None

    def act(self, obs):
        raw = bind_official_state(obs).raw()
        step = step_of(raw, self.cfg)
        raw["step"] = step
        if step <= self.last_step:
            self.reset()
        self.active = {k: s for k, s in self.active.items() if still_needed(s, raw)}
        decision = choose(raw, self.cfg, self.active, self.max_candidates)
        self.active = {s["key"]: copy.deepcopy(s) for s in decision.bundle.commitments}
        self.last_choice = decision.report
        self.last_step = step
        return copy.deepcopy(decision.bundle.action)


_RUNTIMES = {}


def reset_agent():
    _RUNTIMES.clear()


def agent(obs, configuration=None):
    cfg = settings_from(configuration)
    seat = int(obs["player"] if isinstance(obs, dict) else obs.player)
    runtime = _RUNTIMES.get(seat)
    if runtime is None or runtime.cfg != cfg:
        runtime = Runtime(cfg)
        _RUNTIMES[seat] = runtime
    return runtime.act(obs)


def debug_state(seat=0):
    runtime = _RUNTIMES.get(int(seat))
    if runtime is None:
        return None
    return {"active_jobs": copy.deepcopy(runtime.active),
            "last_choice": copy.deepcopy(runtime.last_choice)}
