"""Replay Distilled Asakusa 138890 v0.

Teacher-free behavioral distillation from the observed 719-turn replay trajectory:
Episode 114449406, seed 2112196784, seat 1,
Asakusa Agricultural Co., Ltd, terminal self 138890.

Battle replay is the source evidence. This model preserves the observed action
program first. It does not claim that the trajectory generalizes to other Worlds.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import gzip
import json

_SOURCE_EPISODE = 114449406
_SOURCE_SEED = 2112196784
_SOURCE_SEAT = 1
_SOURCE_AGENT = "Asakusa Agricultural Co., Ltd"
_SOURCE_TERMINAL_SELF = 138890.0

_DATA_B85 = """"""
