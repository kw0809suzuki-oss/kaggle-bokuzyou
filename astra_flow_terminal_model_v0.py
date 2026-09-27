"""Astra Flow v0: expert production with receding-horizon inventory execution.

Design implementation, not a validated improvement. See ASTRA_FLOW_MODEL_V0.md.
Official rules are a runtime dependency; Teacher source is pinned and attributed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
import math

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

PREMIUM = ("MILK", "STRAWBERRY", "WOOL", "MELON")
TEACHER_SOURCE = Path(__file__).with_name("astra_flow_vendor").joinpath("seyamalam_v21.py").read_text()
TEACHER_CODE = compile(TEACHER_SOURCE, "seyamalam_v21.pinned.py", "exec")


@dataclass(frozen=True)
class Settings:
    horizon: int = 24
    risk_weight: float = 0.25
    flow_alpha: float = 0.15
    episodeSteps: int = 720
    turnsPerDay: int = 24
    boardSize: int = 10
    shedCapacity: int = 100
    maxMarketOrdersPerTurn: int = 10
    farmHandCostMult: int = 1
    townShopSellInterval: int = 4
    townCenterSellInterval: int = 24


def demand(obs, start, stop, item, cfg):
    """Known town consumption in [start, stop), before the future decision.

    Future random shop unlocks are deliberately not predicted.
    """
    per_tick = sum((2 if len(rules.SHOPS[s]) == 1 else 1)
                   for s in obs.get("town", {}).get("unlocked_shops", [])
                   if item in rules.SHOPS[s])
    def ticks(interval):
        return (stop - 1) // interval - (start - 1) // interval
    return (per_tick * ticks(cfg.townShopSellInterval)
            + (ticks(cfg.townCenterSellInterval) if item in rules.TOWN_CENTER_PRODUCTS else 0))


def project_actors(obs, action, cfg):
    """Use the official actor implementation, including atomic seed validation."""
    farm = deepcopy(obs["farms"][obs["player"]])
    private = deepcopy(obs["private"])
    actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    requests = {}
    for a in actions:
        if a and a[0] == "PLANT" and len(a) > 1:
            requests[a[1]] = requests.get(a[1], 0) + 1
    blocked = {c for c, n in requests.items() if n > private["seeds"].get(c, 0)}
    for i, a in enumerate(actions):
        if a and a[0] == "PLANT" and len(a) > 1 and a[1] in blocked:
            a = ["PASS"]
        rules._apply_unit_action(farm, private, i, a, cfg.boardSize,
                                obs["day"], cfg.turnsPerDay, cfg.shedCapacity)
    return farm, private


class AstraFlowAgent:
    def __init__(self, settings=None):
        self.cfg = settings or Settings()
        self.reset()

    def reset(self):
        # Each instance owns Teacher globals: two seats and resets never share them.
        self.teacher = ModuleType("astra_flow_isolated_teacher")
        exec(TEACHER_CODE, self.teacher.__dict__)
        self.last_step = -1
        self.expected_inventory = None
        self.flow = {item: 0.0 for item in PREMIUM}
        self.noise = {item: 0.0 for item in PREMIUM}

    def _observe(self, obs, step):
        if self.expected_inventory is not None and step == self.last_step + 1:
            for item in PREMIUM:
                residual = obs["market"]["inventory"][item] - self.expected_inventory[item]
                # A residual, not a claim to have identified the opponent's action:
                # simultaneous quoting can also affect our own executed sale.
                error = residual - self.flow[item]
                a = self.cfg.flow_alpha
                self.flow[item] += a * error
                self.noise[item] = (1 - a) * self.noise[item] + a * abs(error)

    def _terminal_transport(self, obs, action, remaining):
        farm = obs["farms"][obs["player"]]
        half = self.cfg.boardSize // 2
        accesses = [(x, y) for x in (half - 1, half) for y in (half - 1, half)
                    if farm["tiles"][y][x] != "LOCKED"]
        if not accesses:
            return
        positions = [farm["farmer"], *farm["hands"]]
        actions = [action["farmer"], *action["hands"]]
        for i, (pos, inv) in enumerate(zip(positions, obs["private"]["inventories"])):
            if not any(inv.get(p, 0) > 0 for p in rules.PRODUCTS):
                continue
            x, y = pos
            tx, ty = min(accesses, key=lambda p: (abs(p[0]-x)+abs(p[1]-y), p))
            distance = abs(tx-x)+abs(ty-y)
            if remaining <= distance + 1:
                actions[i] = (["DROP"] if distance == 0 else
                              ["EAST" if tx > x else "WEST"] if tx != x else
                              ["SOUTH" if ty > y else "NORTH"])
        action["farmer"], action["hands"] = actions[0], actions[1:]

    def _quantity(self, obs, item, stock, lower, remaining):
        """Two-stage liquidation: q now, remainder at one shared future checkpoint.

        Replan next turn; prices include our own marginal market impact. Score is
        mean cash across supply scenarios minus a spread penalty. This forecasts
        proceeds on current stock, not the value of the entire future farm.
        """
        if remaining == 1:
            return stock
        cfg = self.cfg
        inv = obs["market"]["inventory"][item]
        params = obs["market"].get("params")
        @lru_cache(maxsize=None)
        def quote(level):
            return rules.market_price(item, level, params)
        cash, levels = [0], [inv]
        for _ in range(stock):
            p = quote(levels[-1])
            cash.append(cash[-1] + p)
            levels.append(levels[-1] + (p > rules.PRICE_FLOOR))
        horizon = min(cfg.horizon, remaining - 1)
        waits = sorted({min(t, horizon) for t in (1, 4, 8, 12, 24, horizon)})
        best_q, best_value = stock, float(cash[stock])
        step = obs["step"]
        for wait in waits:
            consumed = demand(obs, step, step + wait, item, cfg)
            mu = self.flow[item] * wait
            spread = self.noise[item] * math.sqrt(wait)
            # No external net flow is retained as a scenario even after observing it.
            shifts = (0, round(mu - spread), round(mu + spread))
            future_tables = {}
            for shift in set(shifts):
                # Prefix sums make all quantity alternatives cheap. At the price
                # floor, not incrementing inventory produces the same block cash.
                start = inv + shift - consumed
                prefix = [0]
                for j in range(stock):
                    prefix.append(prefix[-1] + quote(start+j))
                future_tables[shift] = prefix
            for q in range(lower, stock):
                impact = levels[q] - inv
                values = [cash[q] + future_tables[s][impact+stock-q]
                          - future_tables[s][impact] for s in shifts]
                score = sum(values)/len(values) - cfg.risk_weight*(max(values)-min(values))
                # Ties release cash now; holding requires a strict modeled benefit.
                if score > best_value + 1e-9 or (abs(score-best_value) <= 1e-9 and q > best_q):
                    best_q, best_value = q, score
        return best_q

    def _expected_market(self, obs, action, farm, private):
        """Official no-opponent market transition, only for next residual estimate."""
        market = deepcopy(obs["market"])
        farms = [farm, deepcopy(farm)]
        states = [SimpleNamespace(observation=SimpleNamespace(
            market=market, farms=farms, private=private), action=action),
            SimpleNamespace(observation=SimpleNamespace(private=rules._new_private()),
                            action={"market": []})]
        rules._process_market(states, SimpleNamespace(configuration=self.cfg))
        return {item: market["inventory"][item] - demand(
            obs, obs["step"], obs["step"]+1, item, self.cfg) for item in PREMIUM}

    def act(self, obs):
        step = int(obs["step"])
        if step <= self.last_step:
            self.reset()
        self._observe(obs, step)
        # Reuse the attributed pre-recovery Teacher, not the V21 cash-gap latch.
        action = deepcopy(self.teacher._V18S_BASE_AGENT(obs))
        hands = len(obs["farms"][obs["player"]]["hands"])
        action["hands"] = (action.get("hands", []) + [["PASS"]]*hands)[:hands]
        remaining = self.cfg.episodeSteps - 1 - step
        if remaining <= 0:
            return {"farmer": ["PASS"], "hands": [["PASS"]]*hands, "market": []}
        self._terminal_transport(obs, action, remaining)
        farm, private = project_actors(obs, action, self.cfg)
        stock = private["shed"]
        orders = []
        for order in action.get("market", []):
            # Necessary timing condition only, not a profitability guarantee.
            if order[0] in ("BUY_SEED", "BUY_ANIMAL"):
                table = rules.CROPS if order[0] == "BUY_SEED" else rules.ANIMALS
                placement_step = step + (1 if order[0] == "BUY_SEED" else 2)
                maturity_step = (placement_step // self.cfg.turnsPerDay
                                 + table[order[1]]["first_yield_day"]) * self.cfg.turnsPerDay
                if maturity_step + 1 > self.cfg.episodeSteps - 2:
                    continue
            if remaining == 1 and order[0] != "SELL":
                continue
            orders.append(order)
        cap = self.cfg.maxMarketOrdersPerTurn
        if remaining == 1:
            action["market"] = [["SELL", p, int(stock.get(p, 0))]
                                for p in rules.PRODUCTS if stock.get(p, 0) > 0][:cap]
        else:
            funding = any(o[0] != "SELL" for o in orders)
            covered = {p: sum(int(o[2]) for o in orders if o[0] == "SELL" and o[1] == p)
                       for p in PREMIUM}
            lower = {p: min(stock.get(p, 0), covered[p]) if funding else 0 for p in PREMIUM}
            # Reserve room for currently carried goods. This is a joint warehouse
            # constraint, not four independent assumptions of unlimited storage.
            carried = sum(sum(inv.values()) for inv in private["inventories"])
            nonpremium_sales = sum(min(stock.get(o[1], 0), int(o[2])) for o in orders
                                   if o[0] == "SELL" and o[1] not in PREMIUM)
            need = max(0, sum(stock.values()) + carried - self.cfg.shedCapacity
                       - nonpremium_sales - sum(lower.values()))
            for p in sorted(PREMIUM, key=lambda p: (-obs["market"]["prices"][p], p)):
                extra = min(need, max(0, stock.get(p, 0)-lower[p]))
                lower[p] += extra
                need -= extra
            quantities = {p: self._quantity(obs, p, int(stock.get(p, 0)), int(lower[p]), remaining)
                          for p in PREMIUM if stock.get(p, 0) > 0}
            result, seen = [], set()
            for order in orders:
                if order[0] == "SELL" and order[1] in PREMIUM:
                    p = order[1]
                    if p not in seen and quantities.get(p, 0):
                        result.append(["SELL", p, quantities[p]])
                    seen.add(p)
                else:
                    result.append(order)
            # Preserve original order positions and purchasing priority. Extra
            # sales occupy only free slots; they never displace production orders.
            for p in sorted(quantities, key=lambda p: (-quantities[p]*obs["market"]["prices"][p], p)):
                if p not in seen and quantities[p] and len(result) < cap:
                    result.append(["SELL", p, quantities[p]])
            action["market"] = result[:cap]
        self.expected_inventory = self._expected_market(obs, action, farm, private)
        self.last_step = step
        return action


_agents = {}


def reset_agent():
    _agents.clear()


def agent(obs, configuration=None):
    """Kaggle callable; configuration is accepted to bind official settings."""
    seat = int(obs["player"])
    if seat not in _agents:
        values = {k: configuration[k] for k in Settings.__dataclass_fields__
                  if configuration is not None and k in configuration}
        _agents[seat] = AstraFlowAgent(Settings(**values))
    return _agents[seat].act(obs)
