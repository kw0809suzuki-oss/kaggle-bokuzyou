"""Pure observables for One-Shot View Lens Harness v0."""

from __future__ import annotations
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

NEARBY_MANHATTAN_RADIUS = 2
CANONICAL_OBSERVABLES = {
    "nearby_density": "distinct_targets_within_radius",
    "relationship_age": "current_object_age",
    "exit_proximity": "distinct_targets_sharing_destination",
}

class LensIntegrityError(RuntimeError):
    pass

def _stable(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def target_key(c):
    t = c.target
    tile = t.get("tile")
    if isinstance(tile, Sequence) and not isinstance(tile, (str, bytes)) and len(tile) == 2:
        return ("tile", int(tile[0]), int(tile[1]))
    if c.kind == "deliver_carried_to_shed":
        return ("unit_inventory", int(t["unit_index"]))
    if c.kind == "realize_shed_stock_sale":
        return ("shed_item", str(t["item"]))
    return None

def _positions(raw):
    p = raw["player"]
    farm = raw["farms"][p]
    rows = [farm["farmer"]] + list(farm.get("hands", []) or [])
    return [(int(x[0]), int(x[1])) for x in rows]

def _anchor(snapshot, c):
    key = target_key(c)
    if key is None:
        return None
    if key[0] == "tile":
        return (key[1], key[2])
    if key[0] == "unit_inventory":
        pos = _positions(snapshot.raw())
        i = key[1]
        if i < 0 or i >= len(pos):
            raise LensIntegrityError("unit target outside current units")
        return pos[i]
    return None

def _destination(c):
    t = c.target
    if c.kind == "deliver_carried_to_shed":
        return ("shed",)
    if c.kind == "realize_shed_stock_sale":
        return ("cash",)
    if c.kind == "prepare_for_plant":
        return ("seed_inventory", str(t["crop"]))
    if c.kind in ("establish_plant", "maintain_plant_today", "prepare_surface_for_plant"):
        tile = t.get("tile")
        if isinstance(tile, Sequence) and not isinstance(tile, (str, bytes)) and len(tile) == 2:
            return ("tile", int(tile[0]), int(tile[1]))
    return None

def _tile(raw, x, y):
    p = raw["player"]
    return raw["farms"][p]["tiles"][y][x]

def _nearby(snapshot, candidates):
    anchors = {}
    for c in candidates:
        key = target_key(c)
        a = _anchor(snapshot, c)
        if key is None or a is None:
            continue
        if key in anchors and anchors[key] != a:
            raise LensIntegrityError("inconsistent target anchor")
        anchors[key] = a
    by_target = {}
    for key, a in anchors.items():
        by_target[key] = len({
            k for k, b in anchors.items()
            if k != key and abs(a[0]-b[0]) + abs(a[1]-b[1]) <= NEARBY_MANHATTAN_RADIUS
        })
    return {
        c.candidate_id: by_target.get(target_key(c))
        for c in candidates
    }, {
        "distance": "manhattan",
        "radius": NEARBY_MANHATTAN_RADIUS,
        "self_target_excluded": True,
        "count_unit": "distinct_world_targets",
    }

def _age(snapshot, candidates):
    raw = snapshot.raw()
    day = int(raw["day"])
    by_target = {}
    for c in candidates:
        key = target_key(c)
        if key is None or key[0] != "tile":
            continue
        tile = _tile(raw, key[1], key[2])
        if not (isinstance(tile, Mapping) and tile.get("kind") == "PLANT"):
            continue
        pd = tile.get("planted_day")
        if isinstance(pd, int) and not isinstance(pd, bool):
            age = day - pd
            if age < 0:
                raise LensIntegrityError("negative plant age")
            by_target[key] = age
    return {
        c.candidate_id: by_target.get(target_key(c))
        for c in candidates
    }, {
        "unit": "official_day",
        "source": "current PLANT.planted_day",
        "history_inference_used": False,
    }

def _exit(snapshot, candidates):
    del snapshot
    destinations = {}
    targets_by_destination = defaultdict(set)
    for c in candidates:
        key = target_key(c)
        dest = _destination(c)
        if key is None or dest is None:
            continue
        destinations[c.candidate_id] = dest
        targets_by_destination[dest].add(key)
    return {
        c.candidate_id: (
            len(targets_by_destination[destinations[c.candidate_id]])
            if c.candidate_id in destinations else None
        )
        for c in candidates
    }, {
        "count_unit": "distinct_world_targets",
        "collect_plant_output_destination": "undefined_in_v0",
    }

def evaluate_lens(lens_name, snapshot, candidates, baseline_selected_candidate_id):
    if lens_name not in CANONICAL_OBSERVABLES:
        raise ValueError(lens_name)
    by_id = {c.candidate_id: c for c in candidates}
    baseline = by_id.get(baseline_selected_candidate_id)
    if baseline is None:
        raise LensIntegrityError("baseline selected candidate missing")

    if lens_name == "nearby_density":
        obs, meta = _nearby(snapshot, candidates)
    elif lens_name == "relationship_age":
        obs, meta = _age(snapshot, candidates)
    else:
        obs, meta = _exit(snapshot, candidates)

    values_by_target = {}
    ids_by_target = defaultdict(list)
    for c in candidates:
        value = obs.get(c.candidate_id)
        key = target_key(c)
        if value is None or key is None:
            continue
        if key in values_by_target and values_by_target[key] != value:
            raise LensIntegrityError("inconsistent observable for distinct target")
        values_by_target[key] = value
        ids_by_target[key].append(c.candidate_id)

    base = {
        "lens_name": lens_name,
        "canonical_observable": CANONICAL_OBSERVABLES[lens_name],
        "observable_by_candidate": obs,
        "metadata": meta,
    }
    if len(values_by_target) < 2:
        return {**base, "eligible": False, "reason": "fewer_than_two_distinct_targets", "selected_candidate_id": None}

    lo = min(values_by_target.values())
    hi = max(values_by_target.values())
    if lo == hi:
        return {**base, "eligible": False, "reason": "no_observable_variation", "selected_candidate_id": None}

    max_targets = {k for k, v in values_by_target.items() if v == hi}
    baseline_target = target_key(baseline)
    if baseline_target in max_targets:
        return {**base, "eligible": False, "reason": "baseline_target_already_at_maximum", "selected_candidate_id": None}

    selected_target = sorted(max_targets, key=_stable)[0]
    selected_id = sorted(ids_by_target[selected_target])[0]
    return {
        **base,
        "eligible": True,
        "reason": "distinct_observable_replacement_available",
        "selected_candidate_id": selected_id,
        "selected_target": list(selected_target),
        "baseline_target": list(baseline_target) if baseline_target is not None else None,
        "min_observable": lo,
        "max_observable": hi,
    }
