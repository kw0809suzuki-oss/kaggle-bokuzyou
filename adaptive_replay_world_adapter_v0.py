"""World Adapter v0 for Adaptive Replay.

This module does not choose actions.
It only answers the one Current-World feasibility question required by the
single promoted Effect Contract in v0: how many HIRE orders can succeed now.

No other World understanding is implemented.
"""
from __future__ import annotations


def fib_hire_cost_unit(nth_hire_today: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, int(nth_hire_today) - 1)):
        a, b = b, a + b
    return a


def count_feasible_hires(
    *,
    money: float,
    hires_today: int,
    requested: int,
    cost_mult: float = 1.0,
) -> int:
    cash = float(money)
    done = 0
    for offset in range(int(requested)):
        nth = int(hires_today) + offset + 1
        cost = float(cost_mult) * fib_hire_cost_unit(nth)
        if cash < cost:
            break
        cash -= cost
        done += 1
    return done
