import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "opponents"))

SEED = 7004
START_DAY = 7
END_DAY = 13

def load_seyamalam(name):
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def scan_farm(farm):
    c = Counter()
    animals = Counter()
    crop_yield_tiles = 0
    animal_yield_tiles = 0
    crop_yield_units = 0
    animal_yield_units = 0

    for row in farm.get("tiles", []):
        for t in row:
            if t is None:
                c["EMPTY"] += 1
            elif t == "LOCKED":
                c["LOCKED"] += 1
            elif isinstance(t, dict):
                kind = t.get("kind")
                if kind:
                    c[kind] += 1
                y = float(t.get("yield_units", 0) or 0)
                if kind == "PLANT" and y > 0:
                    crop_yield_tiles += 1
                    crop_yield_units += y
                if kind in ("PASTURE", "COOP"):
                    animal = t.get("animal")
                    if animal:
                        animals[animal] += 1
                        if y > 0:
                            animal_yield_tiles += 1
                            animal_yield_units += y

    return {
        "land": len(farm.get("unlocked_quadrants", [])),
        "hands": len(farm.get("hands", [])),
        "plants": c["PLANT"],
        "pastures": c["PASTURE"],
        "coops": c["COOP"],
        "empty": c["EMPTY"],
        "animals": dict(animals),
        "animal_count": sum(animals.values()),
        "crop_yield_tiles": crop_yield_tiles,
        "crop_yield_units": crop_yield_units,
        "animal_yield_tiles": animal_yield_tiles,
        "animal_yield_units": animal_yield_units,
    }

def summarize_action(action):
    market = copy.deepcopy(action.get("market", []))
    units = [copy.deepcopy(action.get("farmer", ["PASS"]))] + copy.deepcopy(action.get("hands", []))
    unit_ops = Counter(a[0] for a in units if a)
    return {
        "market": market,
        "unit_ops": dict(unit_ops),
    }

def main():
    self_s = load_seyamalam("seyamalam_cycle_self")
    opp_s = load_seyamalam("seyamalam_cycle_opp")
    trace = []

    def traced_self(obs):
        action = self_s.agent(obs)
        if START_DAY <= obs["day"] <= END_DAY:
            me = obs["farms"][0]
            private = obs["private"]
            trace.append({
                "step": len(trace),
                "day": obs["day"],
                "hour": obs["hour"],
                "money": float(me.get("money", 0)),
                **scan_farm(me),
                "shed": copy.deepcopy(private.get("shed", {})),
                "seeds": copy.deepcopy(private.get("seeds", {})),
                "action": summarize_action(action),
            })
        return action

    env = make(
        "kaggriculture",
        configuration={"seed": SEED, "episodeSteps": (END_DAY + 1) * 24 + 1},
        debug=False,
    )
    env.run([traced_self, opp_s.agent])

    events = []
    prev = None
    for r in trace:
        if prev is not None:
            for key in ("money","land","hands","plants","pastures","coops","animal_count",
                        "crop_yield_tiles","crop_yield_units","animal_yield_tiles","animal_yield_units"):
                d = r[key] - prev[key]
                if d != 0:
                    events.append({
                        "day": r["day"], "hour": r["hour"],
                        "type": "state_delta", "field": key, "delta": d,
                        "value": r[key],
                    })

        market = r["action"]["market"]
        for order in market:
            if order:
                events.append({
                    "day": r["day"], "hour": r["hour"],
                    "type": "market_action", "order": order,
                    "money_before": r["money"],
                })

        for op, count in r["action"]["unit_ops"].items():
            if op in ("PLANT","BUILD_PASTURE","BUILD_COOP","PLACE","FEED","CARE","HARVEST","DROP"):
                events.append({
                    "day": r["day"], "hour": r["hour"],
                    "type": "unit_action", "op": op, "count": count,
                })

        prev = r

    def first_event(predicate):
        for e in events:
            if predicate(e):
                return e
        return None

    landmarks = {
        "first_buy_land": first_event(lambda e: e["type"]=="market_action" and e["order"][0]=="BUY_LAND"),
        "first_hire": first_event(lambda e: e["type"]=="market_action" and e["order"][0]=="HIRE"),
        "first_buy_seed": first_event(lambda e: e["type"]=="market_action" and e["order"][0]=="BUY_SEED"),
        "first_buy_animal": first_event(lambda e: e["type"]=="market_action" and e["order"][0]=="BUY_ANIMAL"),
        "first_plant": first_event(lambda e: e["type"]=="unit_action" and e["op"]=="PLANT"),
        "first_place": first_event(lambda e: e["type"]=="unit_action" and e["op"]=="PLACE"),
        "first_crop_yield_increase": first_event(lambda e: e["type"]=="state_delta" and e["field"]=="crop_yield_units" and e["delta"]>0),
        "first_animal_yield_increase": first_event(lambda e: e["type"]=="state_delta" and e["field"]=="animal_yield_units" and e["delta"]>0),
        "first_harvest": first_event(lambda e: e["type"]=="unit_action" and e["op"]=="HARVEST"),
        "first_sell": first_event(lambda e: e["type"]=="market_action" and e["order"][0]=="SELL"),
    }

    daily = []
    for day in range(START_DAY, END_DAY + 1):
        rows = [r for r in trace if r["day"] == day]
        if not rows:
            continue
        first = rows[0]
        last = rows[-1]
        daily.append({
            "day": day,
            "start": {k:first[k] for k in ("money","land","hands","plants","pastures","coops","animal_count","crop_yield_units","animal_yield_units")},
            "end": {k:last[k] for k in ("money","land","hands","plants","pastures","coops","animal_count","crop_yield_units","animal_yield_units")},
        })

    result = {
        "probe": "seyamalam_cash_to_cash_cycle_trace_v0",
        "seed": SEED,
        "scope": f"Seat0 Seyamalam vs independent Seat1 Seyamalam, days {START_DAY}-{END_DAY}.",
        "boundary": "Raw observation and mechanical event extraction only; no causal attribution between individual purchases and later yields.",
        "landmarks": landmarks,
        "daily": daily,
        "events": events,
        "trace": trace,
    }

    Path("seyamalam_cash_to_cash_cycle_result_v0.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    print(json.dumps({"landmarks": landmarks, "daily": daily}, ensure_ascii=False))

if __name__ == "__main__":
    main()
