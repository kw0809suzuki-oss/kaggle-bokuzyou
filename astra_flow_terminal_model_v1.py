"""Astra Flow v1: Frozen Body production + Astra Flow v0 recovery planning.

Comparison model only. Recovery / market / terminal logic is inherited unchanged
from Astra Flow Terminal Model v0. The production proposal seam is replaced with
an isolated RelationshipSurfaceBody instance from the frozen baseline.
"""
from __future__ import annotations

from astra_flow_terminal_model_v0 import AstraFlowAgent, Settings
from relationship_surface_body_v0 import RelationshipSurfaceBody


class AstraFlowFrozenProductionAgent(AstraFlowAgent):
    def reset(self):
        # Keep Astra Flow v0 recovery state reset behavior unchanged.
        super().reset()
        # Replace only the production proposal source.
        self.production_body = RelationshipSurfaceBody()
        self.teacher._V18S_BASE_AGENT = self.production_body.act


_agents = {}


def reset_agent():
    _agents.clear()


def agent(obs, configuration=None):
    seat = int(obs["player"])
    if seat not in _agents:
        values = {
            k: configuration[k]
            for k in Settings.__dataclass_fields__
            if configuration is not None and k in configuration
        }
        _agents[seat] = AstraFlowFrozenProductionAgent(Settings(**values))
    return _agents[seat].act(obs)
