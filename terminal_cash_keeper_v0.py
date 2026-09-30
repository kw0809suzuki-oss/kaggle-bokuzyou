"""Terminal Cash Keeper v0.

Objective:
    Finish the episode with cash.

This model is intentionally independent of Replay and Teacher policies.
It reuses only the current Official-World-grounded ShortPlan generator and
projector already present in the repository.

Decision principle:
    Prefer work that is closer to becoming cash.
    Do not start a new crop investment when even the coarse recovery-distance
    estimate cannot fit before terminal.

Important boundary:
    Recovery distance and value_hint are decision heuristics, not evidence that
    a candidate will be profitable or will actually recover by terminal.
"""

from __future__ import annotations

from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_plan_match,
)

DEFAULT_EPISODE_STEPS = 720
DEFAULT_TURNS_PER_DAY = 24
MAX_PLAN_STEPS = 12
RETURN_BUFFER_STEPS = 4

# Lower is closer to terminal cash.
CASH_DISTANCE_STAGE = {
    "realize_shed_stock_sale": 0,
    "deliver_carried_to_shed": 1,
    "collect_plant_output": 2,
    "maintain_plant_today": 3,
    "establish_plant": 4,
    "prepare_surface_for_plant": 5,
    "prepare_for_plant": 6,
}


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


def _config_value(configuration, key, default):
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(key, default)
    return getattr(configuration, key, default)


def _semantic_matches(plans, spec):
    return [
        p
        for p in plans
        if semantic_plan_match(p, kind=spec["kind"], target=spec["target"])
    ]


def _is_blocked(snapshot, plan):
    if plan.kind != "prepare_for_plant":
        return False
    raw = snapshot.raw()
    p = raw["player"]
    crop = str(plan.target["crop"])
    qty = int(plan.target["missing_seed_quantity"])
    cash = float(raw["farms"][p].get("money", 0) or 0)
    return cash < int(CROPS[crop]["seed"]) * qty


def _positions(raw):
    farm = raw["farms"][raw["player"]]
    return [list(farm["farmer"])] + [list(p) for p in (farm.get("hands", []) or [])]


def _shed_access(raw):
    board_size = len(raw["farms"][raw["player"]].get("tiles", []) or [])
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _nearest_unit_distance(raw, tile):
    ps = _positions(raw)
    return min((_manhattan(p, tile) for p in ps), default=0)


def _to_shed_distance(raw, tile):
    access = _shed_access(raw)
    return min((_manhattan(tile, p) for p in access), default=0)


def _crop_for(plan):
    crop = plan.target.get("crop")
    return str(crop) if crop is not None else None


def _recovery_steps(raw, plan, turns_per_day):
    """Coarse distance from the candidate's current stage to a possible cash sale.

    This is deliberately a small ordering heuristic. Official World remains the
    authority on whether each transition actually succeeds.
    """
    kind = plan.kind

    if kind == "realize_shed_stock_sale":
        return 1

    if kind == "deliver_carried_to_shed":
        unit = int(plan.target["unit_index"])
        ps = _positions(raw)
        if unit < 0 or unit >= len(ps):
            return 10**9
        dist = min((_manhattan(ps[unit], s) for s in _shed_access(raw)), default=0)
        return dist + 2  # DROP + later SELL

    tile = plan.target.get("tile")
    if not isinstance(tile, (list, tuple)) or len(tile) != 2:
        return 10**9
    tile = [int(tile[0]), int(tile[1])]
    approach = _nearest_unit_distance(raw, tile)
    exit_dist = _to_shed_distance(raw, tile)

    if kind == "collect_plant_output":
        return approach + 1 + exit_dist + 2

    crop = _crop_for(plan)
    if crop not in CROPS:
        return 10**9
    first_yield_days = int(CROPS[crop]["first_yield_day"])

    if kind == "maintain_plant_today":
        planted_day = int(plan.target.get("planted_day", raw["day"]))
        age_days = max(0, int(raw["day"]) - planted_day)
        days_left = max(0, first_yield_days - age_days)
        maturity = days_left * int(turns_per_day)
        return approach + 1 + maturity + exit_dist + 2 + RETURN_BUFFER_STEPS

    if kind == "establish_plant":
        maturity = first_yield_days * int(turns_per_day)
        return approach + 1 + maturity + exit_dist + 2 + RETURN_BUFFER_STEPS

    if kind == "prepare_surface_for_plant":
        maturity = first_yield_days * int(turns_per_day)
        return approach + 2 + maturity + exit_dist + 2 + RETURN_BUFFER_STEPS

    if kind == "prepare_for_plant":
        maturity = first_yield_days * int(turns_per_day)
        return 1 + approach + 1 + maturity + exit_dist + 2 + RETURN_BUFFER_STEPS

    return 10**9


def _value_hint(raw, plan):
    """Observed-current-price hint only; no future price or yield forecast."""
    prices = raw.get("market", {}).get("prices", {}) or {}
    kind = plan.kind

    if kind == "realize_shed_stock_sale":
        item = str(plan.target["item"])
        qty = float(plan.target.get("available_quantity", 0) or 0)
        return float(prices.get(item, 0) or 0) * qty

    if kind == "deliver_carried_to_shed":
        items = plan.target.get("carried_items", {}) or {}
        return sum(float(prices.get(str(item), 0) or 0) * float(qty or 0)
                   for item, qty in items.items())

    crop = _crop_for(plan)
    if crop is None:
        return 0.0
    price = float(prices.get(crop, 0) or 0)

    if kind == "collect_plant_output":
        qty = float(plan.target.get("available_yield_units", 0) or 0)
        return price * qty

    if kind == "prepare_for_plant":
        seed_cost = float(CROPS[crop]["seed"])
        return price / max(seed_cost, 1.0)

    return price


def _rank_key(raw, plan, turns_per_day):
    stage = CASH_DISTANCE_STAGE.get(plan.kind, 99)
    recovery = _recovery_steps(raw, plan, turns_per_day)
    value = _value_hint(raw, plan)
    density = value / max(float(recovery), 1.0)
    return (stage, -density, recovery, plan.candidate_id)


class TerminalCashKeeper:
    def __init__(self, *, max_plan_steps=MAX_PLAN_STEPS):
        self.max_plan_steps = int(max_plan_steps)
        self.reset()

    def reset(self):
        self.active = None
        self.active_steps = 0
        self.plan_sequence = 0
        self._pending_plan = None
        self._pending_pre = None
        self._last_clock_index = None
        self.last_decision = None

    def _reset_if_new_episode(self, pre, turns_per_day):
        raw = pre.raw()
        clock_index = int(raw["day"]) * int(turns_per_day) + int(raw["hour"])
        if self._last_clock_index is not None and clock_index <= self._last_clock_index:
            self.reset()
        self._last_clock_index = clock_index

    def _close_previous_transition(self, current_pre):
        if self._pending_plan is None or self._pending_pre is None:
            return
        completion = completion_from_states(self._pending_plan, self._pending_pre, current_pre)
        self.active_steps += 1
        if completion.get("complete"):
            self.active = None
            self.active_steps = 0
        self._pending_plan = None
        self._pending_pre = None

    def act(self, obs, configuration=None):
        turns_per_day = int(_config_value(configuration, "turnsPerDay", DEFAULT_TURNS_PER_DAY))
        episode_steps = int(_config_value(configuration, "episodeSteps", DEFAULT_EPISODE_STEPS))

        pre = bind_official_state(obs)
        self._reset_if_new_episode(pre, turns_per_day)
        self._close_previous_transition(pre)

        raw = pre.raw()
        step = int(raw.get("step", obs.get("step", 0)) or 0)
        remaining = max(0, episode_steps - 1 - step)

        plans = generate_plans(pre)
        selectable = [
            p for p in plans
            if not _is_blocked(pre, p)
            and _recovery_steps(raw, p, turns_per_day) <= remaining
        ]

        # Validate the active plan against the current World.
        active_plan = None
        if self.active is not None:
            matches = _semantic_matches(plans, self.active)
            if (
                self.active_steps >= self.max_plan_steps
                or not matches
                or _is_blocked(pre, matches[0])
                or _recovery_steps(raw, matches[0], turns_per_day) > remaining
            ):
                self.active = None
                self.active_steps = 0
            else:
                active_plan = matches[0]

        # Cash-nearer work can preempt a farther active commitment.
        if active_plan is not None and selectable:
            best_now = min(selectable, key=lambda p: _rank_key(raw, p, turns_per_day))
            active_stage = CASH_DISTANCE_STAGE.get(active_plan.kind, 99)
            best_stage = CASH_DISTANCE_STAGE.get(best_now.kind, 99)
            if best_stage < active_stage:
                self.active = None
                self.active_steps = 0
                active_plan = None

        if self.active is None and selectable:
            chosen = min(selectable, key=lambda p: _rank_key(raw, p, turns_per_day))
            self.plan_sequence += 1
            self.active = {
                "sequence": self.plan_sequence,
                "kind": chosen.kind,
                "target": _plain(chosen.target),
            }
            self.active_steps = 0
            active_plan = chosen

        if active_plan is None and self.active is not None:
            matches = _semantic_matches(plans, self.active)
            if matches:
                active_plan = matches[0]

        if active_plan is None:
            self.last_decision = {
                "step": step,
                "remaining": remaining,
                "kind": "PASS",
                "reason": "no_recoverable_current_world_candidate",
                "selectable_count": len(selectable),
            }
            return _plain(baseline_pass_bundle(pre))

        try:
            bundle = project_short_plan(pre, active_plan)
        except Exception:
            self.active = None
            self.active_steps = 0
            self._pending_plan = None
            self._pending_pre = None
            self.last_decision = {
                "step": step,
                "remaining": remaining,
                "kind": "PASS",
                "reason": "projector_failure",
            }
            return _plain(baseline_pass_bundle(pre))

        self.last_decision = {
            "step": step,
            "remaining": remaining,
            "kind": active_plan.kind,
            "candidate_id": active_plan.candidate_id,
            "cash_distance_stage": CASH_DISTANCE_STAGE.get(active_plan.kind, 99),
            "estimated_recovery_steps": _recovery_steps(raw, active_plan, turns_per_day),
            "value_hint": _value_hint(raw, active_plan),
            "selectable_count": len(selectable),
        }
        self._pending_plan = active_plan
        self._pending_pre = pre
        return _plain(bundle)


_DEFAULT = TerminalCashKeeper()
last_decision = None


def reset_agent():
    global last_decision
    _DEFAULT.reset()
    last_decision = None


def agent(obs, configuration=None):
    global last_decision
    action = _DEFAULT.act(obs, configuration)
    last_decision = _DEFAULT.last_decision
    return action
