"""Seyamalam Teacher Body v0.

Purpose:
- Use the pinned public Seyamalam agent as an external Teacher Candidate.
- Do not reinterpret or rewrite its policy.
- Provide one thin seam where our own later experimental overlays can be inserted.

This file itself does not claim the teacher is optimal.
The pinned teacher source is supplied by the experiment runner.
"""

from __future__ import annotations

from typing import Any


class SeyamalamTeacherBody:
    def __init__(self, teacher_module: Any):
        if not hasattr(teacher_module, "agent"):
            raise ValueError("teacher_module must expose agent(obs)")
        self.teacher_module = teacher_module

    def act(self, obs):
        return self.teacher_module.agent(obs)


def make_teacher_body(teacher_module: Any) -> SeyamalamTeacherBody:
    return SeyamalamTeacherBody(teacher_module)
