import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as base
from run_phase_a_teacher_probe_v0 import collect, predict, features, SAMPLE_HOURS

TRAIN_SEEDS = [7001, 7002, 7003]
TRAIN = [r for seed in TRAIN_SEEDS for r in collect(seed)]

PREFERRED_OP = {
    "SURFACE": "BUY_LAND",
    "THROUGHPUT": "HIRE",
    "CROP_ENGINE": "BUY_SEED",
    "ANIMAL_ENGINE": "BUY_ANIMAL",
}

_STATS = {"seen": 0, "matching_action_present": 0, "reordered": 0}


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0


def get_stats():
    return dict(_STATS)


def agent(obs):
    action = copy.deepcopy(base.agent(obs))
    day, hour = obs["day"], obs["hour"]

    if not (7 <= day <= 10 and hour in SAMPLE_HOURS):
        return action

    direction = predict(TRAIN, features(obs))
    preferred = PREFERRED_OP.get(direction)
    if not preferred:
        return action

    _STATS["seen"] += 1
    market = list(action.get("market", []))
    if not market:
        return action

    preferred_orders = [o for o in market if o and o[0] == preferred]
    if not preferred_orders:
        return action

    _STATS["matching_action_present"] += 1

    sells = [o for o in market if o and o[0] == "SELL"]
    others = [o for o in market if o and o[0] != "SELL" and o[0] != preferred]
    reordered = sells + preferred_orders + others

    if reordered != market:
        action["market"] = reordered
        _STATS["reordered"] += 1

    return action
