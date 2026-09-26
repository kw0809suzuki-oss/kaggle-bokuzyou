"""One-Shot View Lens Harness v0.

Thin supporting layer around the frozen Relationship Surface Body v0.
The Body selects normally first. On the first eligible fresh Selection only,
the harness may replace that selected Candidate once. Body RNG is never
consumed by the lens. After replacement, the lens is disabled for the battle.
"""

from __future__ import annotations
import hashlib

import relationship_surface_body_v0 as promoted
from one_shot_view_lenses_v0 import CANONICAL_OBSERVABLES, evaluate_lens
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from short_plan_action_projector_v0 import project_short_plan

VALID_LENSES = tuple(CANONICAL_OBSERVABLES) + ("disabled",)

def _rng_digest(state):
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()

class OneShotViewLensHarness:
    def __init__(self, lens_name, *, policy_seed=promoted.POLICY_SEED):
        if lens_name not in VALID_LENSES:
            raise ValueError(f"unknown lens: {lens_name}")
        self.lens_name = lens_name
        self.body = promoted.RelationshipSurfaceBody(policy_seed=policy_seed)
        self._episode_clock = None
        self._turn = 0
        self.intervened = False
        self.intervention_log = None
        self.fresh_selection_checks = 0
        self.eligible_count = 0

    def reset(self):
        self.body.reset()
        self._episode_clock = None
        self._turn = 0
        self.intervened = False
        self.intervention_log = None
        self.fresh_selection_checks = 0
        self.eligible_count = 0

    def _prepare_episode(self, snapshot):
        raw = snapshot.raw()
        clock = int(raw["day"]) * 24 + int(raw["hour"])
        if self._episode_clock is not None and clock <= self._episode_clock:
            # Preserve the promoted Body's own reset behavior. Only the
            # supporting-layer state is reset here; Body.act() will reset itself.
            self._turn = 0
            self.intervened = False
            self.intervention_log = None
            self.fresh_selection_checks = 0
            self.eligible_count = 0
        self._episode_clock = clock

    def act(self, obs):
        pre = bind_official_state(obs)
        self._prepare_episode(pre)
        turn = self._turn
        sequence_before = self.body.plan_sequence

        baseline_action = self.body.act(obs)
        sequence_after = self.body.plan_sequence

        # Disabled means exact pass-through. After one intervention, the lens
        # is permanently off for the rest of this battle.
        if (
            self.lens_name == "disabled"
            or self.intervened
            or sequence_after == sequence_before
        ):
            self._turn += 1
            return baseline_action

        self.fresh_selection_checks += 1

        # If the promoted Body dropped the selection because projection failed,
        # there is no stable fresh Selection to replace.
        if self.body.active is None:
            self._turn += 1
            return baseline_action

        plans = generate_plans(pre)
        selectable = [p for p in plans if not promoted._is_blocked(pre, p)]
        baseline_matches = promoted._semantic_matches(selectable, self.body.active)
        if len(baseline_matches) != 1:
            raise RuntimeError(
                f"expected one baseline selected Candidate, got {len(baseline_matches)}"
            )
        baseline_candidate = baseline_matches[0]

        # Lens boundary starts after the frozen Body has consumed exactly the
        # RNG it normally consumes for its fresh Selection.
        rng_before_state = self.body.rng.getstate()
        rng_before = _rng_digest(rng_before_state)

        proposal = evaluate_lens(
            self.lens_name,
            pre,
            selectable,
            baseline_candidate.candidate_id,
        )

        rng_after_state = self.body.rng.getstate()
        rng_after = _rng_digest(rng_after_state)
        if rng_before_state != rng_after_state:
            raise RuntimeError("Lens consumed or mutated Body policy RNG")

        if not proposal["eligible"]:
            self._turn += 1
            return baseline_action

        self.eligible_count += 1
        by_id = {p.candidate_id: p for p in selectable}
        lens_candidate = by_id[proposal["selected_candidate_id"]]
        lens_action = promoted._plain(project_short_plan(pre, lens_candidate))

        # One-shot Selection commitment replacement. Continuation from the next
        # turn onward is entirely the frozen Body's existing logic.
        self.body.active = {
            "sequence": self.body.plan_sequence,
            "kind": lens_candidate.kind,
            "target": promoted._plain(lens_candidate.target),
        }
        self.body.active_steps = 0
        self.body._pending_plan = lens_candidate
        self.body._pending_pre = pre

        self.intervened = True
        self.intervention_log = {
            "turn": turn,
            "pre_state_hash": pre.canonical_hash,
            "selectable_candidate_ids": [p.candidate_id for p in selectable],
            "lens_name": self.lens_name,
            "canonical_observable": proposal["canonical_observable"],
            "observable_by_candidate": proposal["observable_by_candidate"],
            "observable_metadata": proposal["metadata"],
            "baseline_selected_candidate": baseline_candidate.candidate_id,
            "lens_selected_candidate": lens_candidate.candidate_id,
            "baseline_projected_action": promoted._plain(baseline_action),
            "lens_projected_action": promoted._plain(lens_action),
            "post_state_hash": None,
            "body_rng_before": rng_before,
            "body_rng_after": rng_after,
            "body_rng_unchanged": rng_before == rng_after,
            "proposal": {
                k: v for k, v in proposal.items()
                if k not in ("observable_by_candidate", "metadata")
            },
        }

        self._turn += 1
        return lens_action

    def observe_post(self, obs):
        """Attach the Official post-state hash for the intervention turn."""
        if self.intervention_log is None:
            return
        if self.intervention_log["post_state_hash"] is not None:
            return
        post = bind_official_state(obs)
        self.intervention_log["post_state_hash"] = post.canonical_hash

    @property
    def intervention_count(self):
        return 1 if self.intervened else 0

    def summary(self):
        return {
            "lens_name": self.lens_name,
            "intervention_count": self.intervention_count,
            "fresh_selection_checks": self.fresh_selection_checks,
            "eligible_count": self.eligible_count,
            "intervention_log": self.intervention_log,
        }
