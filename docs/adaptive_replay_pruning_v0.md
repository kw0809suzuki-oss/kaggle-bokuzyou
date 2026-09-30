# Adaptive Replay Pruning v0

## Parent objective
Increase terminal self in Official World.

## Frozen baseline
Adaptive Replay Contract Runtime v0:
DECEM 157,026 Replay + step24 HIRE Effect Contract only.

## Forbidden
- add new policy
- add recovery
- reorder actions
- change worker actions
- alter the frozen baseline module
- rescue a rejected omission with extra conditions

## Intervention
One market order only is omitted at one replay step after the baseline runtime emits its action.

Stage-1 eligible high-commitment operations:
- HIRE
- BUY_ANIMAL
- BUY_LAND
- BUY_PRODUCT

BUY_SEED is intentionally deferred. There are 244 distinct seed-order omissions; v0 first spends the time budget on larger commitments.

SELL is never removed.

Candidate identity:
(step, market_index, exact_order)

All other market orders, farmer action, hand actions, and step24 HIRE contract behavior remain unchanged.

## Stage 1 — screen
World: seed 92802001, seat0, Seyamalam v21.
Enumerate distinct omission outcomes only. If removing different identical orders at the same step produces the same final market list, keep one representative candidate.
For each occurrence, run a full terminal A/B with only that one order omitted.

Record:
- activation count
- terminal self
- delta terminal self
- terminal opponent
- delta margin

Only candidates with activation_count == 1 and delta_terminal_self > 0 survive.

## Stage 2 — fixed10
Take at most the top 3 Stage-1 candidates by delta terminal self.
Run each on seeds 92802001..92802010, seat0, Seyamalam v21.

Primary judge:
mean delta terminal self.

No candidate is promoted merely for winning the screen.
No rescue tuning.

## Interpretation boundary
A positive omission means only:
the omitted replay commitment was unnecessary or harmful in those tested Current Worlds.

It does not establish why.
It does not authorize broad removal of that operation type.
