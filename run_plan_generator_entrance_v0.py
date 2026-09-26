#!/usr/bin/env python3
import copy
import json
import sys
from pathlib import Path

from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as current
import seyamalam_v21 as opponent
from plan_generator_entrance_v0 import (
    StateSchemaError,
    bind_official_state,
    generate_plans,
)

SEEDS=[7001,7004,7007]
SAMPLE_DAYS={4,8,12,16,20,24}
SAMPLE_HOUR=0
FORBIDDEN_DERIVED_KEYS={"productive_tiles","occupied_capacity","empty_capacity"}


def plain(x):
    if isinstance(x,dict):
        return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:
        return x
    if hasattr(x,"items"):
        return {str(k):plain(v) for k,v in x.items()}
    raise TypeError(type(x).__name__)


def contains_forbidden_key(x):
    if isinstance(x,dict):
        for k,v in x.items():
            if k in FORBIDDEN_DERIVED_KEYS:
                return True
            if contains_forbidden_key(v):
                return True
    elif isinstance(x,list):
        return any(contains_forbidden_key(v) for v in x)
    return False


def main():
    rows=[]
    current.reset_telemetry()
    for seed in SEEDS:
        current.reset_telemetry()
        env=make("kaggriculture",configuration={"seed":seed},debug=False)
        env.reset(num_agents=2)

        while not env.done:
            s0=env._Environment__get_shared_state(0)
            s1=env._Environment__get_shared_state(1)
            obs0=s0["observation"]
            obs1=s1["observation"]

            day=int(obs0.get("day",0) or 0)
            hour=int(obs0.get("hour",0) or 0)

            if day in SAMPLE_DAYS and hour==SAMPLE_HOUR:
                official_plain=plain(obs0)
                a=bind_official_state(obs0)
                b=bind_official_state(copy.deepcopy(obs0))

                same_snapshot=(a.canonical_hash==b.canonical_hash and a.canonical_json()==b.canonical_json())
                raw_lossless=(a.raw()==official_plain)

                # Raw accessor must be exact and defensive.
                raw_money=official_plain["farms"][official_plain["player"]]["money"]
                accessor_money=a.get("farms",official_plain["player"],"money")
                accessor_exact=(accessor_money==raw_money)

                mutated=a.raw()
                mutated["day"]=-999
                defensive_copy=(a.get("day")==official_plain["day"])

                before_hash=a.canonical_hash
                plans=generate_plans(a)
                after_hash=a.canonical_hash
                generator_connected=(plans==[])
                generator_non_mutating=(before_hash==after_hash and a.raw()==official_plain)

                no_forbidden_derived=(not contains_forbidden_key(a.raw()))

                rows.append({
                    "seed":seed,
                    "day":day,
                    "hour":hour,
                    "canonical_hash":a.canonical_hash,
                    "same_snapshot":same_snapshot,
                    "raw_lossless":raw_lossless,
                    "accessor_exact":accessor_exact,
                    "defensive_copy":defensive_copy,
                    "generator_connected":generator_connected,
                    "generator_non_mutating":generator_non_mutating,
                    "no_forbidden_derived":no_forbidden_derived,
                })

            a0=plain(current.agent(obs0))
            a1=plain(opponent.agent(obs1))
            env.step([a0,a1])

    # Negative schema guard.
    sample=bind_official_state(
        make("kaggriculture",configuration={"seed":7001},debug=False)
        .reset(num_agents=2)[0]["observation"]
    ).raw()
    broken=copy.deepcopy(sample)
    broken.pop("farms",None)
    schema_rejected=False
    try:
        bind_official_state(broken)
    except StateSchemaError:
        schema_rejected=True

    checks={
        "samples":len(rows),
        "same_official_state_same_snapshot":all(r["same_snapshot"] for r in rows),
        "lossless_raw_preservation":all(r["raw_lossless"] for r in rows),
        "raw_accessor_exact":all(r["accessor_exact"] for r in rows),
        "raw_accessor_defensive":all(r["defensive_copy"] for r in rows),
        "generate_plans_connected":all(r["generator_connected"] for r in rows),
        "generate_plans_non_mutating":all(r["generator_non_mutating"] for r in rows),
        "no_unconfirmed_derived_fields":all(r["no_forbidden_derived"] for r in rows),
        "invalid_schema_rejected":schema_rejected,
    }
    checks["all_acceptance_pass"]=all(v for k,v in checks.items() if k!="samples")

    out={
        "schema":"plan-generator-entrance-closure-v0",
        "purpose":"Close Official State -> canonical raw-backed StateSnapshot -> generate_plans entrance without derived economic interpretation.",
        "seeds":SEEDS,
        "sample_days":sorted(SAMPLE_DAYS),
        "sample_hour":SAMPLE_HOUR,
        "acceptance":checks,
        "rows":rows,
        "boundary":{
            "generate_plans_returns":"[]",
            "not_implemented":[
                "productive_tiles",
                "occupied_capacity",
                "empty_capacity",
                "plan candidate semantics",
                "plan evaluation",
                "multi-step comparison",
            ],
        },
    }
    Path("plan_generator_entrance_closure_v0.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SUMMARY "+json.dumps(checks,separators=(",",":")))


if __name__=="__main__":
    main()
