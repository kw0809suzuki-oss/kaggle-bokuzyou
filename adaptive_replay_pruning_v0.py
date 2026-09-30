"""One-shot spending-order omission wrapper over frozen Contract Runtime v0."""
from __future__ import annotations
from copy import deepcopy
import adaptive_replay_contract_runtime_v0 as base

target_step = None
target_market_index = None
target_order = None
activation_count = 0
last_event = None

ELIGIBLE = {"HIRE","BUY_ANIMAL","BUY_SEED","BUY_LAND","BUY_PRODUCT"}

def configure(step=None, market_index=None, order=None):
    global target_step, target_market_index, target_order
    target_step = step
    target_market_index = market_index
    target_order = deepcopy(order)

def reset_agent():
    global activation_count, last_event
    activation_count = 0
    last_event = None
    base.reset_agent()

def agent(obs, configuration=None):
    global activation_count, last_event
    action = deepcopy(base.agent(obs, configuration))
    step = int(obs.get("step", 0) or 0)
    if target_step is None or step != int(target_step):
        return action
    market = list(action.get("market", []) or [])
    idx = int(target_market_index)
    if idx < 0 or idx >= len(market):
        return action
    order = market[idx]
    if not isinstance(order, (list, tuple)) or not order or order[0] not in ELIGIBLE:
        return action
    if target_order is not None and list(order) != list(target_order):
        return action
    action["market"] = market[:idx] + market[idx+1:]
    activation_count += 1
    last_event = {"step":step,"market_index":idx,"omitted_order":deepcopy(order)}
    return action
