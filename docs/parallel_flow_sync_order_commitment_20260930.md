# Parallel Flow Sync｜Order × Commitment｜2026-09-30

## Purpose
Synchronize two independently developed Kaggriculture research lines without merging implementations or promoting a new policy.

Parent objective remains:
Official Worldで terminal self を高く残す。

Current production model remains frozen:
Adaptive Replay Contract Runtime v0
= DECEM 157,026 Replay + step24 HIRE Effect Contract only.

---

## Line C｜Order Selector

Branch:
experiment/order-terminal-necessity-gate-v0-20260930

Run:
36670219953

Intervention:
At step217, preserve the same operation set and change only execution order.

Original:
HIRE → BUY_PRODUCT WHEAT 19

Swap:
BUY_PRODUCT WHEAT 19 → HIRE

Terminal result across 20 Worlds:
- Swap better: 14
- Original better: 6
- Tie: 0
- mean Δterminal_self (Swap - Original): +702.85
- min: -1,815
- max: +3,333

Confirmed boundary:
The better order is World-dependent.
Same action set does not imply one globally best order.

---

## Line P｜Replay Pruning

Branch:
experiment/adaptive-replay-pruning-v0-20260930

Run:
36704259280

Intervention:
Preserve Replay except omit exactly one spending commitment.

Step0 candidate:
omit BUY_PRODUCT WHEAT 5

Fixed10:
- improved: 6
- worsened: 4
- mean Δterminal_self: +12,214.5
- min: -41,177
- max: +74,453

Step0 Context Probe:
Run 36706608145

Confirmed:
- pre-State identical 10/10
- immediate step0→1 intervention effect same shape 10/10
- terminal sign still splits 6 improve / 4 worsen

Therefore:
The sign is not encoded in the visible entry State or the immediate one-transition effect.

---

## Synchronized Observation

The two lines differ in intervention type:

Order line:
same commitments, different order.

Pruning line:
same Replay skeleton, one commitment removed.

But both independently show:

```
a fixed Replay instruction pattern
does not have one globally dominant realization
across Current Worlds.
```

What varies can be:
- whether a commitment should be admitted at all
- when an admitted commitment should execute relative to another

This is an observation boundary, not yet an architecture.

---

## Do Not Promote Yet

Do not infer:
- WHEAT buying is bad
- early spending should be reduced
- Swap order is globally better
- pruning and ordering should be combined into one controller
- a learned selector is already justified

Do not modify the Current production model from this sync alone.

---

## Shared Open Question

What Current-World information first separates the Worlds where:

- keep vs omit changes sign, or
- original vs swap changes sign?

The next useful abstraction, if pursued, is not a new action.

It is the smallest observable boundary that determines whether an existing Replay commitment should:
1. be admitted,
2. be ordered before/after another commitment,
3. or remain untouched.

Until such a boundary is observed, preserve both lines as independent Evidence.
