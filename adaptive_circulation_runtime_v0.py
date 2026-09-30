"""Adaptive Circulation Runtime v0 — observer-first form.

Parent objective:
    Increase terminal self in Official Kaggriculture.

v0 deliberately changes NO action behavior.
It preserves Adaptive Replay Contract Runtime v0 exactly and establishes a
separate model identity for circulation-effect observation.

Reason:
    The next repair must be justified by a Source-vs-Current effect break.
    No new BUY / HARVEST / SELL / routing rule is allowed before that break
    is observed.

Future architecture:
    Frozen Replay Skeleton
      -> Source Circulation Contract
      -> Current World Effect Check
      -> Minimal Effect Repair

Current v0:
    Frozen Adaptive Replay Contract Runtime v0 passthrough only.
"""

from __future__ import annotations

from copy import deepcopy

import adaptive_replay_contract_runtime_v0 as body


def reset_agent():
    if hasattr(body, "reset_agent"):
        body.reset_agent()


def agent(obs, configuration=None):
    return deepcopy(body.agent(obs, configuration))
