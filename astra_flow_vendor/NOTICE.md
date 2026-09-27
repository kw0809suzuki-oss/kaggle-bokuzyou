# Pinned Teacher dependency

`seyamalam_v21.py` is an unchanged copy of:

- https://github.com/Seyamalam/Kaggriculture/blob/8b8c421eb10634c756583ce10c75189f50c83a72/main.py
- SHA-256: `0cd14b653102d276c4f902fa3b8c6bd81d869b8ab64c422cb881b9d2346ec639`
- License: Apache-2.0, reproduced in `LICENSE-2.0.txt`.

Its original attribution header remains intact. It credits Kaito Fukami's public
v18 implementation and the episode authors used for embedded expert schedules.
Astra Flow does not claim authorship of that production policy or those schedules.

The new model calls `_V18S_BASE_AGENT`: the upstream policy before the V18/V21
additional recovery sweep and cash-gap abstention layer. This is a private,
version-pinned seam; updating the Teacher requires reviewing it again.
