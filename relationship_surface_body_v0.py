"""Promoted Relationship Surface Body v0.

This is the first standalone Body implementation of the behavior promoted by
Relationship Surface Promotion Boundary v0.

Promotion evidence:
- Shadow: non-interference PASS
- Fixed World: terminal 23 -> 1080
- Fresh10: 9 improved / 1 worse
- Fresh30 Promotion Boundary: 30 improved / 0 worse / 0 same

Meaning of promotion is intentionally narrow:
use this Body as the next observation baseline under current evidence.
It is not a claim that Relationship Surface is a true/correct policy and it
does not encode a Return theory.

Frozen behavior:
- Generator: unchanged
- confirmed-BLOCKED rule: unchanged
- Candidate semantics: unchanged
- active Plan continuation: unchanged
- Projector: unchanged
- Completion: unchanged
- Selection change only:
    selectable Candidates
      -> group by Current-World relationship signature
      -> uniformly choose a signature group
      -> uniformly choose a Candidate inside that group

Public interface:
    agent(obs) -> Kaggriculture ActionBundle

For controlled tests or multiple independent episodes in one process, use
RelationshipSurfaceBody directly or call reset_agent().
"""

from __future__ import annotations

import random
from typing import Any

from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from current_world_relationship_lens_v0 import annotate_candidates
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import (
    baseline_pass_bundle,
    completion_from_states,
    project_short_plan,
    semantic_plan_match,
)

POLICY_SEED = 20260926
MAX_PLAN_STEPS = 12


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


def _semantic_matches(plans, spec):
    return [
        p
        for p in plans
        if semantic_plan_match(p, kind=spec["kind"], target=spec["target"])
    ]


def _first_action_status(snapshot, plan):
    """Same narrow evidence-supported BLOCKED rule as the promoted Body."""
    if plan.kind == "prepare_for_plant":
        raw = snapshot.raw()
        p = raw["player"]
        crop = str(plan.target["crop"])
        qty = int(plan.target["missing_seed_quantity"])
        cash = float(raw["farms"][p].get("money", 0) or 0)
        unit_cost = int(CROPS[crop]["seed"])
        required = unit_cost * qty
        if cash < required:
            return {
                "status": "BLOCKED",
                "reason": "cash_below_official_seed_cost",
                "cash": cash,
                "unit_cost": unit_cost,
                "quantity": qty,
                "required_cost": required,
            }
    return {
        "status": "UNKNOWN",
        "reason": "not_closed_by_current_evidence",
    }


def _is_blocked(snapshot, plan):
    return _first_action_status(snapshot, plan)["status"] == "BLOCKED"


def _signature_tuple(annotation):
    return tuple(annotation["derived_relationship_signature"])


class RelationshipSurfaceBody:
    """Stateful Kaggriculture Body matching the promoted P1 behavior."""

    def __init__(
        self,
        *,
        policy_seed: int = POLICY_SEED,
        max_plan_steps: int = MAX_PLAN_STEPS,
    ):
        self.policy_seed = int(policy_seed)
        self.max_plan_steps = int(max_plan_steps)
        self.reset()

    def reset(self):
        self.rng = random.Random(self.policy_seed)
        self.active = None
        self.active_steps = 0
        self.plan_sequence = 0

        # Completion in the original runner is observed after env.step().
        # A normal agent sees that PostState as the next call's PreState, so
        # carry exactly one previous projected Plan across calls.
        self._pending_plan = None
        self._pending_pre = None
        self._last_clock_index = None

    def _reset_if_new_episode(self, pre):
        raw = pre.raw()
        clock_index = int(raw["day"]) * 24 + int(raw["hour"])
        if (
            self._last_clock_index is not None
            and clock_index <= self._last_clock_index
        ):
            self.reset()
        self._last_clock_index = clock_index

    def _close_previous_transition(self, current_pre):
        if self._pending_plan is None or self._pending_pre is None:
            return

        completion = completion_from_states(
            self._pending_plan,
            self._pending_pre,
            current_pre,
        )
        self.active_steps += 1
        if completion.get("complete"):
            self.active = None
            self.active_steps = 0

        self._pending_plan = None
        self._pending_pre = None

    def act(self, obs):
        pre = bind_official_state(obs)
        self._reset_if_new_episode(pre)
        self._close_previous_transition(pre)

        plans = generate_plans(pre)
        blocked_ids = {
            p.candidate_id for p in plans if _is_blocked(pre, p)
        }
        selectable = [
            p for p in plans if p.candidate_id not in blocked_ids
        ]

        if self.active is not None:
            matches = _semantic_matches(plans, self.active)
            if self.active_steps >= self.max_plan_steps:
                self.active = None
                self.active_steps = 0
            elif not matches:
                self.active = None
                self.active_steps = 0
            elif _is_blocked(pre, matches[0]):
                self.active = None
                self.active_steps = 0

        if self.active is None and selectable:
            annotations = annotate_candidates(pre, selectable)
            by_id = {a["candidate_id"]: a for a in annotations}

            groups = {}
            group_order = []
            for candidate in selectable:
                sig = _signature_tuple(by_id[candidate.candidate_id])
                if sig not in groups:
                    groups[sig] = []
                    group_order.append(sig)
                groups[sig].append(candidate)

            selected_signature = group_order[
                self.rng.randrange(len(group_order))
            ]
            within = groups[selected_signature]
            chosen = within[self.rng.randrange(len(within))]

            self.plan_sequence += 1
            self.active = {
                "sequence": self.plan_sequence,
                "kind": chosen.kind,
                "target": _plain(chosen.target),
            }
            self.active_steps = 0

        current_plan = None
        if self.active is not None:
            matches = _semantic_matches(plans, self.active)
            if matches:
                current_plan = matches[0]

        if current_plan is None:
            return _plain(baseline_pass_bundle(pre))

        try:
            bundle = project_short_plan(pre, current_plan)
        except Exception:
            # Exact promoted runner behavior on projector failure:
            # emit baseline PASS and drop the active Plan.
            self.active = None
            self.active_steps = 0
            self._pending_plan = None
            self._pending_pre = None
            return _plain(baseline_pass_bundle(pre))

        self._pending_plan = current_plan
        self._pending_pre = pre
        return _plain(bundle)


_DEFAULT_BODY = RelationshipSurfaceBody()


def reset_agent():
    """Reset the module-level agent before a new controlled episode."""
    _DEFAULT_BODY.reset()


def agent(obs):
    """Kaggriculture agent entry point."""
    return _DEFAULT_BODY.act(obs)
