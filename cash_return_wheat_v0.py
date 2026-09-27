"""Continuous cash-return driving prototype v0.

This is not a claim that one active WHEAT plant is optimal.
It is one full-battle operating policy used to test a different experimental unit:
judge work with its downstream recovery actions in view, rather than replacing one
Selection once and then returning immediately to the frozen Body.

The prototype:
- uses one fixed WHEAT target at the default farmer spawn / shed-access tile;
- starts new work only when current-price gross at the configured harvest age
  exceeds seed cost and there is enough season time to reach recovery;
- services an existing plant only while recovery remains reachable;
- harvests, drops, sells, and may buy the next seed using legal simultaneous
  unit + market actions;
- never claims future price is known.
"""

from __future__ import annotations

from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from plan_generator_entrance_v0 import bind_official_state
from short_plan_action_projector_v0 import baseline_pass_bundle


def _plain(x: Any):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): _plain(v) for k, v in x.items()}
    raise TypeError(type(x).__name__)


def _shed_access_tiles(board_size: int):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _move_toward(pos, target):
    x, y = int(pos[0]), int(pos[1])
    tx, ty = int(target[0]), int(target[1])
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


class CashReturnWheatBody:
    def __init__(
        self,
        *,
        crop: str = "WHEAT",
        max_active_plants: int = 1,
        harvest_age_days: int = 4,
        season_days: int = 30,
        turns_per_day: int = 24,
        parallel_market: bool = True,
    ):
        if crop not in CROPS:
            raise ValueError(crop)
        if crop != "WHEAT":
            raise ValueError("v0 is intentionally WHEAT-only")
        if max_active_plants != 1:
            raise ValueError("v0 only implements the one-active-plant pilot")
        self.crop = crop
        self.max_active_plants = int(max_active_plants)
        self.harvest_age_days = int(harvest_age_days)
        self.season_days = int(season_days)
        self.turns_per_day = int(turns_per_day)
        self.parallel_market = bool(parallel_market)
        self.reset()

    def reset(self):
        self.target_tile = None
        self._last_clock = None
        self._pending_sale = None
        self.counters = {
            "buy_seed_orders": 0,
            "plant_actions": 0,
            "water_actions": 0,
            "harvest_actions": 0,
            "drop_actions": 0,
            "sell_orders": 0,
            "cash_return_events": 0,
            "realized_sold_units_observed": 0,
            "abandon_unreachable_turns": 0,
            "start_rejected_price": 0,
            "start_rejected_time": 0,
        }

    def _reset_if_new_episode(self, raw):
        clock = int(raw["day"]) * self.turns_per_day + int(raw["hour"])
        if self._last_clock is not None and clock <= self._last_clock:
            self.reset()
        self._last_clock = clock

    def _observe_pending_sale(self, raw):
        if self._pending_sale is None:
            return
        current_shed = int((raw["private"].get("shed", {}) or {}).get(self.crop, 0) or 0)
        predicted_stock = int(self._pending_sale["predicted_stock_before_market"])
        sold = max(0, predicted_stock - current_shed)
        if sold > 0:
            self.counters["cash_return_events"] += 1
            self.counters["realized_sold_units_observed"] += sold
        self._pending_sale = None

    def _expected_yield_at_target_age(self):
        cd = CROPS[self.crop]
        if cd["ongoing"]:
            raise ValueError("v0 expects one-time crop")
        window_start = (int(cd["max_yield_day"]) + 1) // 2
        bonus_days = max(
            0,
            min(self.harvest_age_days, int(cd["max_yield_day"])) - window_start + 1,
        )
        return min(int(cd["max_yield"]), 1 + bonus_days)

    def _remaining_turns(self, raw):
        day = int(raw["day"])
        hour = int(raw["hour"])
        return max(0, self.season_days * self.turns_per_day - (day * self.turns_per_day + hour))

    def _can_start(self, raw):
        day = int(raw["day"])
        price = int((raw["market"].get("prices", {}) or {}).get(self.crop, 0) or 0)
        seed_cost = int(CROPS[self.crop]["seed"])
        expected_yield = self._expected_yield_at_target_age()
        expected_gross_at_current_price = price * expected_yield

        # Leave one full day after the target harvest day for harvest/drop/sale
        # delays. This is deliberately conservative and is a v0 test condition,
        # not a general optimal cutoff.
        time_ok = day + self.harvest_age_days + 1 < self.season_days
        price_ok = expected_gross_at_current_price > seed_cost
        return {
            "ok": time_ok and price_ok,
            "time_ok": time_ok,
            "price_ok": price_ok,
            "current_price": price,
            "seed_cost": seed_cost,
            "expected_yield": expected_yield,
            "expected_gross_at_current_price": expected_gross_at_current_price,
        }

    def _market_orders(self, raw, *, predicted_drop_qty=0, need_seed=False):
        orders = []
        shed = int((raw["private"].get("shed", {}) or {}).get(self.crop, 0) or 0)
        predicted_sale_stock = shed + int(predicted_drop_qty)

        if predicted_sale_stock > 0:
            orders.append(["SELL", self.crop, predicted_sale_stock])
            self.counters["sell_orders"] += 1
            self._pending_sale = {
                "predicted_stock_before_market": predicted_sale_stock,
                "pre_cash": float(raw["farms"][raw["player"]].get("money", 0) or 0),
            }

        if need_seed:
            seeds = int((raw["private"].get("seeds", {}) or {}).get(self.crop, 0) or 0)
            cash = float(raw["farms"][raw["player"]].get("money", 0) or 0)
            seed_cost = int(CROPS[self.crop]["seed"])
            if seeds <= 0 and cash >= seed_cost:
                orders.append(["BUY_SEED", self.crop, 1])
                self.counters["buy_seed_orders"] += 1

        return orders

    def act(self, obs):
        pre = bind_official_state(obs)
        raw = pre.raw()
        self._reset_if_new_episode(raw)
        self._observe_pending_sale(raw)

        bundle = _plain(baseline_pass_bundle(pre))
        p = int(raw["player"])
        farm = raw["farms"][p]
        board_size = len(farm.get("tiles", []) or [])
        farmer_pos = list(farm["farmer"])

        if self.target_tile is None:
            self.target_tile = list(farmer_pos)

        tx, ty = int(self.target_tile[0]), int(self.target_tile[1])
        tile = farm["tiles"][ty][tx]
        farmer_inv = (raw["private"].get("inventories", []) or [{}])[0]
        carried = int((farmer_inv or {}).get(self.crop, 0) or 0)
        shed_access = set(_shed_access_tiles(board_size))
        at_target = tuple(farmer_pos) == (tx, ty)
        at_shed = tuple(farmer_pos) in shed_access

        start = self._can_start(raw)
        seeds = int((raw["private"].get("seeds", {}) or {}).get(self.crop, 0) or 0)

        # Downstream recovery takes priority over starting new work.
        if carried > 0:
            if at_shed:
                bundle["farmer"] = ["DROP"]
                self.counters["drop_actions"] += 1
                if self.parallel_market:
                    bundle["market"] = self._market_orders(
                        raw,
                        predicted_drop_qty=carried,
                        need_seed=start["ok"],
                    )
            else:
                access = min(
                    shed_access,
                    key=lambda q: (
                        abs(int(farmer_pos[0]) - q[0]) + abs(int(farmer_pos[1]) - q[1]),
                        list(_shed_access_tiles(board_size)).index(q),
                    ),
                )
                bundle["farmer"] = _move_toward(farmer_pos, access)
                if self.parallel_market:
                    bundle["market"] = self._market_orders(raw, need_seed=False)
            return bundle

        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == self.crop:
            # Shed realization may proceed while the farmer services the live plant.
            if self.parallel_market:
                bundle["market"] = self._market_orders(raw, need_seed=False)
            planted_day = int(tile.get("planted_day", raw["day"]))
            age = int(raw["day"]) - planted_day
            yield_units = int(tile.get("yield_units", 0) or 0)
            first_yield_day = int(CROPS[self.crop]["first_yield_day"])
            max_yield_day = int(CROPS[self.crop]["max_yield_day"])
            watered = bool(tile.get("watered_today", False))

            earliest_recovery_day = planted_day + first_yield_day
            recovery_reachable = earliest_recovery_day < self.season_days
            if not recovery_reachable:
                self.counters["abandon_unreachable_turns"] += 1
                return bundle

            # Near the end, take already-recoverable output instead of insisting
            # on the configured target age.
            near_terminal = self._remaining_turns(raw) <= self.turns_per_day
            if yield_units > 0 and age >= first_yield_day and (
                age >= self.harvest_age_days or near_terminal
            ):
                # At the normal target age, collect the current day's legal
                # watering bonus first when there is another turn left.
                in_bonus_window = ((max_yield_day + 1) // 2) <= age <= max_yield_day
                if (
                    not near_terminal
                    and not watered
                    and in_bonus_window
                    and int(raw["hour"]) < self.turns_per_day - 1
                ):
                    if at_target:
                        bundle["farmer"] = ["WATER"]
                        self.counters["water_actions"] += 1
                    else:
                        bundle["farmer"] = _move_toward(farmer_pos, (tx, ty))
                    return bundle

                if at_target:
                    bundle["farmer"] = ["HARVEST"]
                    self.counters["harvest_actions"] += 1
                else:
                    bundle["farmer"] = _move_toward(farmer_pos, (tx, ty))
                return bundle

            # While recovery is still reachable, keep the live plant serviced.
            if not watered:
                if at_target:
                    bundle["farmer"] = ["WATER"]
                    self.counters["water_actions"] += 1
                else:
                    bundle["farmer"] = _move_toward(farmer_pos, (tx, ty))
            return bundle

        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            if at_target:
                bundle["farmer"] = ["DIG"]
            else:
                bundle["farmer"] = _move_toward(farmer_pos, (tx, ty))
            if self.parallel_market:
                bundle["market"] = self._market_orders(
                    raw,
                    need_seed=start["ok"] and seeds <= 0,
                )
            return bundle

        if tile is None:
            if not start["time_ok"]:
                self.counters["start_rejected_time"] += 1
                if self.parallel_market:
                    bundle["market"] = self._market_orders(raw, need_seed=False)
                return bundle
            if not start["price_ok"]:
                self.counters["start_rejected_price"] += 1
                if self.parallel_market:
                    bundle["market"] = self._market_orders(raw, need_seed=False)
                return bundle

            if seeds > 0:
                if at_target:
                    bundle["farmer"] = ["PLANT", self.crop]
                    self.counters["plant_actions"] += 1
                else:
                    bundle["farmer"] = _move_toward(farmer_pos, (tx, ty))
                if self.parallel_market:
                    bundle["market"] = self._market_orders(raw, need_seed=False)
                return bundle

            bundle["market"] = self._market_orders(raw, need_seed=True)
            return bundle

        # Do not destroy unknown/foreign structures to preserve a narrow v0.
        if self.parallel_market:
            bundle["market"] = self._market_orders(raw, need_seed=False)
        return bundle

    def summary(self):
        return {
            "crop": self.crop,
            "target_tile": list(self.target_tile) if self.target_tile is not None else None,
            "max_active_plants": self.max_active_plants,
            "harvest_age_days": self.harvest_age_days,
            "parallel_market": self.parallel_market,
            "counters": dict(self.counters),
        }


def make_candidate(**kwargs):
    return CashReturnWheatBody(**kwargs)
