"""Build the final single-file Kaggriculture submission.

Output:
  dist/free_slack_submission/main.py

The generated main.py is self-contained with respect to our own repository:
- DECEM replay program is embedded as source text.
- World Lens free generator is embedded as source text.
- the promoted step24 HIRE Effect Contract is inlined.
- the free candidate surface is live but non-interfering without a warrant.

Official kaggle-environments remains the runtime dependency.
"""
from __future__ import annotations

from pathlib import Path
import sys


TEMPLATE = r'''"""Kaggriculture submission — Adaptive Replay + Free Slack Surface v0.

Standing behavior:
    Adaptive Replay Contract Runtime v0

Slack:
    An unconstrained World Lens ActionBundle is generated every turn.
    Generated alternatives are candidates, not Evidence. No replacement warrant
    is currently promoted, so standing behavior keeps decision authority.

Principle:
    GENERATE FREE + DETERMINE WITH WARRANT
"""
from __future__ import annotations

import json
import sys
import types
from copy import deepcopy


# ---------------------------------------------------------------------------
# Embedded standing replay body
# ---------------------------------------------------------------------------

_REPLAY_SOURCE = __REPLAY_SOURCE__
_replay = types.ModuleType("_submission_embedded_replay")
_replay.__file__ = "<embedded:decem_replay_distilled_157026_v0.py>"
sys.modules[_replay.__name__] = _replay
exec(compile(_REPLAY_SOURCE, _replay.__file__, "exec"), _replay.__dict__)


# ---------------------------------------------------------------------------
# Embedded free World Lens generator
# ---------------------------------------------------------------------------

_WORLD_LENS_SOURCE = __WORLD_LENS_SOURCE__
_world_lens = types.ModuleType("_submission_embedded_world_lens")
_world_lens.__file__ = "<embedded:world_lens_agent_v0.py>"
sys.modules[_world_lens.__name__] = _world_lens
exec(compile(_WORLD_LENS_SOURCE, _world_lens.__file__, "exec"), _world_lens.__dict__)


# ---------------------------------------------------------------------------
# Standing Adaptive Replay Effect Contract
# ---------------------------------------------------------------------------

surface_count = 0
distinct_alternative_count = 0
last_surface = None
first_distinct_surface = None
trigger_count = 0
last_guard_event = None


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if hasattr(v, "items"):
        return {str(k): _plain(x) for k, x in v.items()}
    if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
        return [_plain(x) for x in v]
    return v


def _cost_mult(configuration):
    if configuration is None:
        return 1.0
    if isinstance(configuration, dict):
        return float(configuration.get("farmHandCostMult", 1) or 1)
    return float(getattr(configuration, "farmHandCostMult", 1) or 1)


def _fib_hire_cost_unit(nth_hire_today):
    a, b = 1, 1
    for _ in range(max(0, int(nth_hire_today) - 1)):
        a, b = b, a + b
    return a


def _count_feasible_hires(money, hires_today, requested, cost_mult=1.0):
    cash = float(money)
    done = 0
    for offset in range(int(requested)):
        nth = int(hires_today) + offset + 1
        cost = float(cost_mult) * _fib_hire_cost_unit(nth)
        if cash < cost:
            break
        cash -= cost
        done += 1
    return done


def _count_hires(market_orders):
    return sum(
        1 for order in market_orders
        if isinstance(order, (list, tuple))
        and len(order) >= 1
        and order[0] == "HIRE"
    )


def _cap_hires(market_orders, count):
    kept = 0
    out = []
    for order in market_orders:
        is_hire = (
            isinstance(order, (list, tuple))
            and len(order) >= 1
            and order[0] == "HIRE"
        )
        if not is_hire:
            out.append(order)
            continue
        if kept < int(count):
            out.append(order)
            kept += 1
    return out


def _apply_promoted_contract(action, obs, configuration):
    global trigger_count, last_guard_event

    if int(obs.get("step", 0) or 0) != 24:
        return action

    market_orders = list(action.get("market", []) or [])
    requested = _count_hires(market_orders)
    protected_count = 1

    if requested <= protected_count:
        return action

    player = int(obs.get("player", 0) or 0)
    farm = obs["farms"][player]
    money = float(farm.get("money", 0) or 0)
    hires_today = int(farm.get("hires_today", 0) or 0)

    feasible = _count_feasible_hires(
        money=money,
        hires_today=hires_today,
        requested=requested,
        cost_mult=_cost_mult(configuration),
    )

    if feasible <= protected_count:
        return action

    transformed = deepcopy(action)
    transformed["market"] = _cap_hires(market_orders, protected_count)

    trigger_count += 1
    last_guard_event = {
        "step": 24,
        "money": money,
        "hires_today": hires_today,
        "requested_hires": requested,
        "current_feasible_hires": feasible,
        "source_realized_hires": protected_count,
    }
    return transformed


def _normalize_bundle(obs, action):
    p = int(obs["player"])
    hands = len(obs["farms"][p].get("hands", []) or [])
    raw = _plain(action) if isinstance(action, dict) else {}

    farmer = raw.get("farmer", ["PASS"])
    if not isinstance(farmer, list) or not farmer:
        farmer = ["PASS"]

    hand_actions = list(raw.get("hands", []) or [])
    hand_actions = (hand_actions + [["PASS"]] * hands)[:hands]

    market = list(raw.get("market", []) or [])
    return {"farmer": farmer, "hands": hand_actions, "market": market}


def _canonical(action):
    return json.dumps(
        _plain(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def reset_agent():
    global surface_count, distinct_alternative_count
    global last_surface, first_distinct_surface
    global trigger_count, last_guard_event

    surface_count = 0
    distinct_alternative_count = 0
    last_surface = None
    first_distinct_surface = None
    trigger_count = 0
    last_guard_event = None

    if hasattr(_replay, "reset_agent"):
        _replay.reset_agent()
    if hasattr(_world_lens, "reset_agent"):
        _world_lens.reset_agent()


def _standing_candidate(obs, configuration=None):
    action = deepcopy(_replay.agent(obs, configuration))
    return _apply_promoted_contract(action, obs, configuration)


def _free_candidate(obs, configuration=None):
    try:
        return _normalize_bundle(obs, _world_lens.agent(obs, configuration))
    except Exception:
        return None


def _generate_candidate_surface(obs, configuration=None):
    standing = _normalize_bundle(obs, _standing_candidate(obs, configuration))
    alternative = _free_candidate(obs, configuration)
    return standing, alternative


def _determine_with_warrant(standing, alternative):
    # No generated alternative currently has a promoted terminal-supported
    # replacement warrant. Generation stays free; decision authority does not.
    return standing


def _record_surface(obs, standing, alternative):
    global surface_count, distinct_alternative_count
    global last_surface, first_distinct_surface

    distinct = (
        alternative is not None
        and _canonical(alternative) != _canonical(standing)
    )
    surface_count += 1
    if distinct:
        distinct_alternative_count += 1

    row = {
        "step": int(obs.get("step", 0) or 0),
        "day": int(obs.get("day", 0) or 0),
        "hour": int(obs.get("hour", 0) or 0),
        "alternative_present": alternative is not None,
        "alternative_distinct": bool(distinct),
        "standing_action": deepcopy(standing),
        "alternative_action": deepcopy(alternative),
        "warrant": "NO_PROMOTED_WARRANT",
    }
    last_surface = row
    if distinct and first_distinct_surface is None:
        first_distinct_surface = deepcopy(row)


# Keep this as the final top-level callable in the file.
def agent(obs, configuration=None):
    standing, alternative = _generate_candidate_surface(obs, configuration)
    _record_surface(obs, standing, alternative)
    chosen = _determine_with_warrant(standing, alternative)
    return deepcopy(chosen)
'''


def build(destination: Path) -> None:
    root = Path(__file__).resolve().parent
    replay = (root / "decem_replay_distilled_157026_v0.py").read_text(encoding="utf-8")
    world_lens = (root / "world_lens_agent_v0.py").read_text(encoding="utf-8")

    source = TEMPLATE.replace("__REPLAY_SOURCE__", repr(replay))
    source = source.replace("__WORLD_LENS_SOURCE__", repr(world_lens))

    compile(source, str(destination), "exec")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist/free_slack_submission/main.py")
    build(target)
    print(target)
