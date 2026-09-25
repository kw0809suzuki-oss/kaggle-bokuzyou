import copy
import sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "selfsrc"))
sys.path.insert(0, str(ROOT / "opponents"))

import whole_flow_control_agent as base
import seyamalam_v21 as opponent
from run_phase_a_teacher_probe_v0 import (
    features, investment_spend, predict, scales, dist, SAMPLE_HOURS, fib
)

TRAIN_SEEDS = [7001, 7002, 7003]
SEED_COST = {"WHEAT":10,"CARROT":20,"TOMATO":50,"STRAWBERRY":100,"MELON":80}
ANIMAL_COST = {"GOOSE":300,"COW":400,"SHEEP":500}
LAND_COST = {1:1000,2:2000,3:4000}
CLASS_OP = {
    "SURFACE": "BUY_LAND",
    "THROUGHPUT": "HIRE",
    "CROP_ENGINE": "BUY_SEED",
    "ANIMAL_ENGINE": "BUY_ANIMAL",
}


def collect_teacher_examples(seed):
    records = []

    def traced_opp(obs):
        action = opponent.agent(obs)
        records.append({
            "features": features(obs),
            "farm": copy.deepcopy(obs["farms"][obs["player"]]),
            "action": copy.deepcopy(action),
        })
        return action

    env = make("kaggriculture", configuration={"seed": seed, "episodeSteps": 265}, debug=False)
    env.run([base.agent, traced_opp])

    rows = []
    for i, r in enumerate(records):
        f = r["features"]
        if not (7 <= f["day"] <= 10 and f["hour"] in SAMPLE_HOURS):
            continue
        spend = Counter()
        for future in records[i:i+24]:
            spend.update(investment_spend(future))
        label = spend.most_common(1)[0][0] if spend else "HOLD"
        wanted = CLASS_OP.get(label)
        exemplar = None
        if wanted:
            for order in r["action"].get("market", []):
                if order and order[0] == wanted:
                    exemplar = copy.deepcopy(order)
                    break
        rows.append({"features": f, "label": label, "exemplar": exemplar})
    return rows


TRAIN = [r for seed in TRAIN_SEEDS for r in collect_teacher_examples(seed)]
MM = scales(TRAIN)

_STATS = {
    "decision_points": 0,
    "direction_missing_from_self": 0,
    "teacher_exemplar_available": 0,
    "feasible_minimal_action": 0,
    "injected": 0,
    "injected_by_direction": Counter(),
}


def reset_stats():
    _STATS["decision_points"] = 0
    _STATS["direction_missing_from_self"] = 0
    _STATS["teacher_exemplar_available"] = 0
    _STATS["feasible_minimal_action"] = 0
    _STATS["injected"] = 0
    _STATS["injected_by_direction"] = Counter()


def get_stats():
    out = dict(_STATS)
    out["injected_by_direction"] = dict(_STATS["injected_by_direction"])
    return out


def nearest_exemplar(f, direction):
    candidates = [r for r in TRAIN if r["label"] == direction and r["exemplar"] is not None]
    if not candidates:
        return None
    return copy.deepcopy(min(candidates, key=lambda r: dist(r["features"], f, MM))["exemplar"])


def minimal_feasible_order(obs, direction, exemplar):
    me = obs["farms"][obs["player"]]
    money = float(me.get("money", 0))

    if direction == "SURFACE":
        lands = len(me.get("unlocked_quadrants", []))
        cost = LAND_COST.get(lands)
        return ["BUY_LAND"] if cost is not None and money >= cost else None

    if direction == "THROUGHPUT":
        hires = int(me.get("hires_today", 0))
        cost = fib(hires)
        return ["HIRE"] if money >= cost else None

    if direction == "CROP_ENGINE":
        if not exemplar or len(exemplar) < 3:
            return None
        crop = exemplar[1]
        cost = SEED_COST.get(crop)
        return ["BUY_SEED", crop, 1] if cost is not None and money >= cost else None

    if direction == "ANIMAL_ENGINE":
        if not exemplar or len(exemplar) < 3:
            return None
        animal = exemplar[1]
        cost = ANIMAL_COST.get(animal)
        return ["BUY_ANIMAL", animal, 1] if cost is not None and money >= cost else None

    return None


def agent(obs):
    action = copy.deepcopy(base.agent(obs))
    day, hour = obs["day"], obs["hour"]
    if not (7 <= day <= 10 and hour in SAMPLE_HOURS):
        return action

    f = features(obs)
    direction = predict(TRAIN, f)
    wanted = CLASS_OP.get(direction)
    if not wanted:
        return action

    _STATS["decision_points"] += 1
    market = list(action.get("market", []))

    if any(o and o[0] == wanted for o in market):
        return action

    _STATS["direction_missing_from_self"] += 1

    if len(market) >= 10:
        return action

    exemplar = nearest_exemplar(f, direction)
    if exemplar is None:
        return action
    _STATS["teacher_exemplar_available"] += 1

    order = minimal_feasible_order(obs, direction, exemplar)
    if order is None:
        return action
    _STATS["feasible_minimal_action"] += 1

    sells = [o for o in market if o and o[0] == "SELL"]
    rest = [o for o in market if not (o and o[0] == "SELL")]
    action["market"] = sells + [order] + rest
    _STATS["injected"] += 1
    _STATS["injected_by_direction"][direction] += 1
    return action
