"""Phase A Lens Model v0.

Experimental only. This is not a Battle agent and does not execute actions.
It applies one Evaluation Lens to compact observed states:

    Convert cash into productive assets that can return within remaining time.

The output is a direction, not an action rule:
SURFACE / THROUGHPUT / CROP_ENGINE / ANIMAL_ENGINE / HOLD

All candidate directions are evaluated from the same state. The proxy equations
are model internals, not Evidence and not game rules.
"""

from __future__ import annotations

LENS = (
    "今は牧場を育てる時間。"
    "Cashを、残り時間内に回収できる生産資産へ変えて、牧場の稼ぐ力を高める。"
)

CROPS = {
    "WHEAT":      {"seed_cost": 10,  "first": 2,  "max_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed_cost": 20,  "first": 2,  "max_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed_cost": 50,  "first": 8,  "max_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed_cost": 100, "first": 10, "max_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed_cost": 80,  "first": 10, "max_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "first": 4, "interval": 1, "product": "EGG"},
    "COW":   {"cost": 400, "first": 8, "interval": 2, "product": "MILK"},
    "SHEEP": {"cost": 500, "first": 6, "interval": 3, "product": "WOOL"},
}

LAND_COST = {1: 1000, 2: 2000, 3: 4000}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _fib_hire_cost(n: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, n)):
        a, b = b, a + b
    return a


def _best_crop_roi_proxy(remaining_days: int, prices: dict) -> tuple[str | None, float]:
    best_name, best_roi = None, 0.0
    for name, spec in CROPS.items():
        if remaining_days < spec["first"]:
            continue
        if spec["ongoing"]:
            cycles = 1 + (remaining_days - spec["first"]) // spec["interval"]
            units = cycles * spec["max_yield"]
        else:
            if remaining_days >= spec["max_day"]:
                units = spec["max_yield"]
            else:
                span = max(1, spec["max_day"] - spec["first"] + 1)
                units = max(1, int(spec["max_yield"] * (remaining_days - spec["first"] + 1) / span))
        gross = prices.get(name, 0) * units
        roi = max(0.0, (gross - spec["seed_cost"]) / max(1, spec["seed_cost"]))
        if roi > best_roi:
            best_name, best_roi = name, roi
    return best_name, best_roi


def _best_animal_roi_proxy(remaining_days: int, prices: dict) -> tuple[str | None, float]:
    best_name, best_roi = None, 0.0
    for name, spec in ANIMALS.items():
        if remaining_days < spec["first"]:
            continue
        cycles = 1 + (remaining_days - spec["first"]) // spec["interval"]
        gross = prices.get(spec["product"], 0) * cycles
        roi = max(0.0, (gross - spec["cost"]) / spec["cost"])
        if roi > best_roi:
            best_name, best_roi = name, roi
    return best_name, best_roi


def evaluate(state: dict) -> dict:
    day = int(state["day"])
    hour = int(state["hour"])
    money = float(state["money"])
    lands = int(state["land"])
    hands = int(state["hands"])
    hires_today = int(state.get("hires_today", hands))
    plants = int(state["plants"])
    pastures = int(state["pastures"])
    empty = int(state["empty"])
    weeds = int(state["weeds"])
    seeds = int(state["seeds"])
    prices = dict(state["prices"])

    capacity = max(1, plants + pastures + empty + weeds)
    occupied = capacity - empty
    remaining_days = max(0, 30 - day)
    remaining_day_fraction = remaining_days / 30.0
    hours_left_fraction = _clip01((24 - hour) / 24.0)

    # One common Lens, four different conversion surfaces.
    # No day-specific "if X then action Y" rule exists here.
    surface_pressure = occupied / capacity

    near_term_work = plants + pastures + weeds + min(empty, seeds)
    daily_work_capacity = max(1, (1 + hands) * 8)
    throughput_deficit = max(0.0, near_term_work - daily_work_capacity) / max(1, near_term_work)
    labor_readiness = _clip01(daily_work_capacity / max(1, near_term_work))

    next_land_cost = LAND_COST.get(lands)
    surface_affordability = (
        _clip01(money / next_land_cost) if next_land_cost else 0.0
    )

    next_hire_cost = _fib_hire_cost(hires_today)
    throughput_affordability = _clip01(money / max(1, next_hire_cost))

    crop_name, crop_roi = _best_crop_roi_proxy(remaining_days, prices)
    animal_name, animal_roi = _best_animal_roi_proxy(remaining_days, prices)

    def roi_norm(x: float) -> float:
        return x / (1.0 + x) if x > 0 else 0.0

    free_surface = empty / capacity

    scores = {
        "SURFACE": (
            surface_pressure
            * remaining_day_fraction
            * surface_affordability
        ),
        "THROUGHPUT": (
            throughput_deficit
            * hours_left_fraction
            * throughput_affordability
        ),
        "CROP_ENGINE": (
            free_surface
            * labor_readiness
            * remaining_day_fraction
            * roi_norm(crop_roi)
            * _clip01((money + seeds * 20) / 300.0)
        ),
        "ANIMAL_ENGINE": (
            free_surface
            * labor_readiness
            * remaining_day_fraction
            * roi_norm(animal_roi)
            * _clip01(money / 400.0)
        ),
        "HOLD": 0.01,
    }

    direction = max(scores, key=scores.get)
    return {
        "lens": LENS,
        "direction": direction,
        "scores": {k: round(v, 6) for k, v in scores.items()},
        "best_crop_proxy": crop_name,
        "best_animal_proxy": animal_name,
    }
