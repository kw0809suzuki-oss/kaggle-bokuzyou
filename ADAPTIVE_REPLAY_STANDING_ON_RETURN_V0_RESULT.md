# Adaptive Replay Standing-on-Return v0 — Result

## Objective

Strengthen the current model without changing the frozen Adaptive Replay Contract Runtime v0 body.

## Candidate

Baseline:
- `engineering/adaptive-replay-contract-runtime-v0-20260930`
- DECEM 157,026 Replay
- step24 HIRE Effect Contract only

Candidate change:
- After the baseline action is produced, if a self worker is already standing on a crop tile with `yield_units > 0`, replace only that worker action with `HARVEST`.
- Market orders are unchanged.
- Other worker actions are unchanged.
- Baseline runtime file is unchanged.

## TDD gate

RED:
- Same synthetic state returned baseline `NORTH`.
- Expected `HARVEST`.
- Test failed as intended.

GREEN:
- Wrapper candidate changed only the standing worker to `HARVEST`.
- Other worker action and market orders stayed unchanged.

## Fixed10 terminal A/B

Opponent: Seyamalam v21  
Seat: 0  
Seeds: 92802001..92802010  
Objective: terminal self

Summary:
- improved: 0
- worsened: 10
- same: 0
- activated cases: 10/10
- standing-on-return triggers: 470 total (47 each seed)
- baseline mean terminal self: 103,945.2
- candidate mean terminal self: 0.0
- mean delta terminal self: -103,945.2
- median delta terminal self: -97,363
- min delta terminal self: -163,879
- max delta terminal self: -78,365
- mean delta margin: -177,573.4

## Decision

REJECT this candidate.

Confirmed only for this intervention:
- Unconditionally overriding Replay unit actions with standing crop HARVEST is catastrophic on this fixed10 gate.
- The candidate activated in every case, so this is not a non-activation result.
- The frozen Adaptive Replay Contract Runtime v0 remains unchanged.

Not concluded:
- Standing-on-work as a general Kaggriculture principle is not rejected.
- Task persistence / claim retention is not tested here.
- Return logistics after HARVEST is not tested here.
- The reason terminal self falls to zero is not established by this gate and is not needed for rejection.

## Return to parent

Keep Adaptive Replay Contract Runtime v0 as the current model.
Do not promote standing-on-return v0.
