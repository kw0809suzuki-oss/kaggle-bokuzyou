"""Generate feasible recovery operations; rank whole-farm cash forecasts.

The imported rollout is a self projection, not a complete World simulation.
No future shop path, opponent policy or random stream is inferred here.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import zip_longest

from strong_model_v0_reimplementation.jobs import (
    Job, distance, materialize_active, nearest_shed, positions,
)
from strong_model_v0_reimplementation.planner import (
    Bundle, Envelope, Settings, _project_one_turn, _service_action,
    choose as strong_choose, operating_jobs, rollout, step_of,
)


@dataclass
class Candidate:
    family: str
    commitments: list[Job]
    forced: set[str]
    action: dict
    scheduled: list[str]


@dataclass
class Decision:
    bundle: Bundle
    report: dict


def _identity(action, commitments):
    return json.dumps(
        [action, sorted((j.spec() for j in commitments), key=lambda s: s["key"])],
        sort_keys=True,
    )


def generate_candidates(raw, cfg: Settings, commitments, max_candidates=6):
    """Change planned work before calling Strong's unchanged action generator.

    The budget bounds *extra* forecasts, not farm assets or workers. Existing
    Strong alternatives are always evaluated in full by strong_choose().
    """
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    if max_candidates == 0 or step_of(raw, cfg) >= cfg.episodeSteps - 1:
        return []
    planned = copy.deepcopy(list(commitments))
    planned_by_key = {j.key: j for j in planned}
    recovery = [j for j in operating_jobs(raw, cfg, harvest_now=True)
                if j.kind in ("harvest", "harvest_animal", "deliver")]
    pos = positions(raw)

    def travel(job):
        if job.kind == "deliver":
            location = pos[job.target["unit_index"]]
            return distance(location, nearest_shed(location, cfg.boardSize))
        return min(distance(p, job.target["tile"]) for p in pos)

    carry = sorted((j for j in recovery if j.kind == "deliver"),
                   key=lambda j: (travel(j), j.key))
    harvest = sorted((j for j in recovery if j.kind != "deliver"),
                     key=lambda j: (travel(j), j.key))
    rows, seen = [], set()

    def add(family, jobs, forced):
        if len(rows) >= max_candidates:
            return
        jobs = copy.deepcopy(jobs)
        action, scheduled, live, _ = _service_action(raw, cfg, jobs, forced)
        if forced and not forced.intersection(scheduled):
            return
        key = _identity(action, live)
        if key not in seen:
            seen.add(key)
            rows.append(Candidate(family, live, set(forced), action, scheduled))

    # Keep maintenance and existing output while making room for recovery.
    retained = [j for j in planned if j.category not in ("production_start", "expansion")]
    if len(retained) != len(planned):
        add("hold_investment", retained, set())

    # Alternate stages so a candidate budget cannot be consumed by one stage.
    for pair in zip_longest(carry, harvest):
        for job in pair:
            if job is None:
                continue
            retained_job = planned_by_key.get(job.key, job)
            jobs = [j for j in planned if j.key != job.key] + [retained_job]
            family = "carry_to_shed" if job.kind == "deliver" else "harvest_ready"
            add(family, jobs, {job.key})
        if len(rows) >= max_candidates:
            break
    return rows


def choose(raw, cfg: Settings, active_specs, max_candidates=6):
    if max_candidates < 0:
        raise ValueError("max_candidates must be nonnegative")
    baseline, representative, active, _ = strong_choose(raw, cfg, active_specs)
    chosen = baseline
    family = "strong"
    lookup = {j.key: j for j in operating_jobs(raw, cfg, True) + active}
    if representative is not None:
        lookup[representative.key] = representative
    planned = []
    for spec in baseline.commitments:
        job = lookup.get(spec["key"]) or materialize_active(spec, raw)
        if job is not None:
            planned.append(copy.deepcopy(job))

    candidates = generate_candidates(raw, cfg, planned, max_candidates)
    rows = []
    # Strong may use harvest_now throughout its forecast. Equal first actions
    # alone do not establish equal continuation policies.
    seen = set()
    winner = None
    for candidate in candidates:
        key = _identity(candidate.action, candidate.commitments)
        if key in seen:
            continue
        seen.add(key)
        value, _ = rollout(raw, cfg, candidate.commitments,
                           first_action=candidate.action)
        rows.append({
            "family": candidate.family,
            "terminal_cash": value,
            "forced": sorted(candidate.forced),
            "scheduled": list(candidate.scheduled),
            "commitment_keys": [j.key for j in candidate.commitments],
            "action": copy.deepcopy(candidate.action),
        })
        # Preserve Strong on equal forecasts; no liquidity bonus is introduced.
        if value > chosen.envelope.central_cash:
            projected = _project_one_turn(raw, candidate.action, cfg)
            chosen = Bundle(candidate.action, candidate.scheduled, projected,
                            Envelope(value, value),
                            float(projected["farms"][raw["player"]]["money"]))
            chosen.commitments = [j.spec() for j in candidate.commitments]
            keys = {j.key for j in candidate.commitments}
            chosen.commitments.extend(
                j.spec() for j in operating_jobs(raw, cfg, True)
                if j.key in candidate.scheduled and j.key not in keys)
            winner = candidate
            family = candidate.family
    if winner is not None:
        stress, _ = rollout(raw, cfg, winner.commitments,
                            first_action=winner.action, stress=True)
        chosen.envelope = Envelope(stress, chosen.envelope.central_cash)
    return Decision(chosen, {
        "step": step_of(raw, cfg),
        "remaining_actions": max(0, cfg.episodeSteps - 1 - step_of(raw, cfg)),
        "generated_count": len(candidates),
        "evaluated_count": len(rows),
        "baseline_terminal_cash": baseline.envelope.central_cash,
        "chosen_terminal_cash": chosen.envelope.central_cash,
        "stress_terminal_cash": chosen.envelope.strict_cash,
        "chosen_family": family,
        "extension_selected": winner is not None,
        "action_changed": chosen.action != baseline.action,
        "commitments_changed": chosen.commitments != baseline.commitments,
        "baseline_action": copy.deepcopy(baseline.action),
        "chosen_action": copy.deepcopy(chosen.action),
        "candidates": rows,
        "forecast_kind": "existing_strong_self_projection",
    })
