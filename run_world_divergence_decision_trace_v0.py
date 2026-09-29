#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def first_diff_paths(left: Any, right: Any, path: str = "") -> list[str]:
    """Return leaf paths whose values differ between two JSON-like values."""
    if type(left) is not type(right):
        return [path or "$"]

    if isinstance(left, dict):
        out: list[str] = []
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                out.append(child)
            else:
                out.extend(first_diff_paths(left[key], right[key], child))
        return out

    if isinstance(left, list):
        out: list[str] = []
        limit = max(len(left), len(right))
        for idx in range(limit):
            child = f"{path}[{idx}]"
            if idx >= len(left) or idx >= len(right):
                out.append(child)
            else:
                out.extend(first_diff_paths(left[idx], right[idx], child))
        return out

    return [] if left == right else [path or "$"]


def first_world_divergence_index(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> int | None:
    """Return the first aligned trace index whose World view differs."""
    for idx in range(min(len(baseline), len(candidate))):
        if baseline[idx].get("world") != candidate[idx].get("world"):
            return idx
    if len(baseline) != len(candidate):
        return min(len(baseline), len(candidate))
    return None


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def first_nonactive_record(steps: list[Any], seat: int) -> dict[str, Any] | None:
    """Return the first recorded frame where the selected seat is no longer ACTIVE."""
    for frame in steps:
        if seat >= len(frame):
            continue
        state = frame[seat]
        status = str(_field(state, "status", ""))
        if status == "ACTIVE":
            continue
        obs = _field(state, "observation", {})
        return {
            "step": int(_field(obs, "step", 0) or 0),
            "status": status,
            "remaining_overage": float(_field(obs, "remainingOverageTime", 0.0) or 0.0),
        }
    return None


def candidate_identity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip score/rank fields so candidate structure can be compared directly."""
    ignored = {"rank", "central_terminal_cash", "strict_terminal_cash"}
    return [
        {key: copy.deepcopy(value) for key, value in row.items() if key not in ignored}
        for row in rows
    ]


def summarize_first_divergence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe how the first World difference propagates through Strong's decision trace."""
    idx = first_world_divergence_index(baseline, candidate)
    if idx is None:
        return {"index": None, "step": None, "world_diff_paths": []}
    if idx >= len(baseline) or idx >= len(candidate):
        return {
            "index": idx,
            "step": None,
            "world_diff_paths": ["$trace_length"],
            "decision_input_diff_paths": [],
            "candidate_identity_diff_paths": [],
            "candidate_score_diff_paths": [],
            "chosen_diff_paths": [],
            "action_diff_paths": [],
        }

    b = baseline[idx]
    c = candidate[idx]
    bd = b.get("debug") or {}
    cd = c.get("debug") or {}
    b_scores = bd.get("candidate_scores") or []
    c_scores = cd.get("candidate_scores") or []

    return {
        "index": idx,
        "step": (b.get("world") or {}).get("step"),
        "world_diff_paths": first_diff_paths(b.get("world"), c.get("world")),
        "decision_input_diff_paths": first_diff_paths(
            bd.get("decision_inputs") or {},
            cd.get("decision_inputs") or {},
        ),
        "candidate_identity_diff_paths": first_diff_paths(
            candidate_identity_rows(b_scores),
            candidate_identity_rows(c_scores),
        ),
        "candidate_score_diff_paths": first_diff_paths(b_scores, c_scores),
        "chosen_diff_paths": first_diff_paths(
            bd.get("chosen") or {},
            cd.get("chosen") or {},
        ),
        "action_diff_paths": first_diff_paths(
            b.get("action") or {},
            c.get("action") or {},
        ),
    }


ROOT = Path(__file__).resolve().parent
SEED = 92804001
SEAT = 0
OPPONENT_PATH = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"
OUT = ROOT / "world_divergence_decision_trace_v0_result.json"
WORLD_FIELDS = ("step", "day", "hour", "market", "town", "farms", "private")


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "items"):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _world_view(obs: Any) -> dict[str, Any]:
    raw = _plain(obs)
    return {key: copy.deepcopy(raw[key]) for key in WORLD_FIELDS if key in raw}


def _load_opponent():
    spec = importlib.util.spec_from_file_location("world_divergence_trace_opponent", OPPONENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load opponent: {OPPONENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reset(obj: Any) -> None:
    fn = getattr(obj, "reset_agent", None)
    if callable(fn):
        fn()


def _money(final_state: list[Any], seat: int) -> float:
    obs = final_state[seat].observation
    farms = obs["farms"] if isinstance(obs, dict) else obs.farms
    farm = farms[seat]
    return float(farm["money"] if isinstance(farm, dict) else farm.money)


class TracedStrong:
    def __init__(self, intervene: bool):
        self.intervene = bool(intervene)
        self.trace: list[dict[str, Any]] = []
        self.last_call_step: int | None = None

    def reset(self) -> None:
        from strong_model_v0_reimplementation.agent import reset_agent

        self.trace = []
        self.last_call_step = None
        reset_agent()

    def agent(self, obs: Any, configuration: Any) -> dict[str, Any]:
        from strong_model_v0_reimplementation.agent import agent as strong_agent, debug_state

        step = int(obs["step"])
        self.last_call_step = step
        native = strong_agent(obs, configuration)

        if self.intervene and step == 0:
            emitted = copy.deepcopy(native)
            emitted.setdefault("market", [])
            emitted["market"].extend([["HIRE"], ["HIRE"]])
        else:
            emitted = native

        if step <= 1:
            native_plain = _plain(native)
            emitted_plain = _plain(emitted)
            debug = _plain(debug_state(SEAT)) or {}
            self.trace.append({
                "step": step,
                "world": _world_view(obs),
                "native_action": copy.deepcopy(native_plain),
                "action": copy.deepcopy(emitted_plain),
                "debug": copy.deepcopy(debug.get("last_choice")),
            })
        return emitted


def _run(label: str, intervene: bool) -> dict[str, Any]:
    from kaggle_environments import make

    opponent = _load_opponent()
    traced = TracedStrong(intervene)
    traced.reset()
    _reset(opponent)

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    agents = [opponent.agent, opponent.agent]
    agents[SEAT] = traced.agent
    env.run(agents)

    self_cash = _money(env.state, SEAT)
    opp_cash = _money(env.state, 1 - SEAT)
    return {
        "label": label,
        "terminal_self": self_cash,
        "terminal_opponent": opp_cash,
        "margin": self_cash - opp_cash,
        "status": str(env.state[SEAT].status),
        "env_steps": len(env.steps) - 1,
        "first_nonactive": first_nonactive_record(env.steps, SEAT),
        "last_agent_step": traced.last_call_step,
        "last_traced_step": traced.trace[-1]["step"] if traced.trace else None,
        "trace": traced.trace,
    }


def _activation_record(candidate_trace: list[dict[str, Any]]) -> dict[str, Any]:
    row = candidate_trace[0]
    native = row["native_action"]
    emitted = row["action"]
    expected_market = list(native.get("market", [])) + [["HIRE"], ["HIRE"]]
    return {
        "step": row["step"],
        "added_orders": [["HIRE"], ["HIRE"]],
        "native_action": native,
        "emitted_action": emitted,
        "activated": emitted.get("market", []) == expected_market,
    }


def main() -> None:
    baseline = _run("strong_baseline", False)
    candidate = _run("strong_plus_step0_hire2", True)

    btrace = baseline.pop("trace")
    ctrace = candidate.pop("trace")
    summary = summarize_first_divergence(btrace, ctrace)
    idx = summary["index"]

    context = None
    previous_world_equal = None
    if isinstance(idx, int) and idx < len(btrace) and idx < len(ctrace):
        context = {
            "baseline": btrace[idx],
            "candidate": ctrace[idx],
        }
        if idx > 0:
            previous_world_equal = btrace[idx - 1]["world"] == ctrace[idx - 1]["world"]

    result = {
        "schema": "world-divergence-decision-trace-v0",
        "purpose": (
            "Observe how the first Official World difference after one fixed HIRE x2 intervention "
            "propagates through Strong decision inputs, candidate structure, scores, choice, and action."
        ),
        "conditions": {
            "seed": SEED,
            "seat": SEAT,
            "opponent": "Seyamalam pinned v21",
            "official_world_commit": "d7729da06cc1382eb742d6980dc3180aa85caa28",
            "world_view_fields": list(WORLD_FIELDS),
            "excluded_from_world_comparison": ["remainingOverageTime"],
        },
        "intervention": _activation_record(ctrace),
        "baseline_terminal": baseline,
        "candidate_terminal": candidate,
        "terminal_self_delta": candidate["terminal_self"] - baseline["terminal_self"],
        "terminal_margin_delta": candidate["margin"] - baseline["margin"],
        "trace_lengths": {
            "baseline": len(btrace),
            "candidate": len(ctrace),
        },
        "first_world_divergence": summary,
        "previous_aligned_world_equal": previous_world_equal,
        "first_divergence_context": context,
        "boundary": {
            "diagnostic_only": True,
            "no_adaptation_judgment": True,
            "no_causal_label_beyond_observed_propagation": True,
            "candidate_step0_only": True,
        },
    }

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "intervention_activated": result["intervention"]["activated"],
        "terminal_self_delta": result["terminal_self_delta"],
        "first_world_divergence": result["first_world_divergence"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
