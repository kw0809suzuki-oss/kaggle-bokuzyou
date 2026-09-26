"""Plan Generator entrance + explicit ShortPlan candidate space v2.

OfficialState -> StateSnapshot -> generate_plans(state)

Rules:
- StateSnapshot is lossless/raw-backed.
- generate_plans discovers raw-grounded short jobs.
- It does not score, rank, select, truncate, or project ActionBundles.
- Enumeration order is only deterministic serialization; every matching
  candidate is returned.
- prepare_for_plant is a targeted prerequisite job: it exists only when the
  specified establish_plant(crop,tile) lacks its required seed input.
- No economic derived fields (productive_tiles / occupied_capacity /
  empty_capacity) are introduced here.
"""

from __future__ import annotations
import copy, hashlib, json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA_VERSION="official-state-snapshot-v0"
PLAN_SCHEMA_VERSION="short-plan-candidate-v2"
REQUIRED_TOP_LEVEL=("player","day","hour","farms","private","market","town")

class StateSchemaError(ValueError): pass

def _plain(v:Any)->Any:
    if isinstance(v,Mapping): return {str(k):_plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_plain(x) for x in v]
    if isinstance(v,(str,int,float,bool)) or v is None: return v
    if hasattr(v,"items"): return {str(k):_plain(x) for k,x in v.items()}
    raise StateSchemaError(f"Unsupported Official State value type: {type(v).__name__}")

def validate_official_state(raw:Mapping[str,Any])->None:
    if not isinstance(raw,Mapping): raise StateSchemaError("Official State must be mapping-like")
    missing=[k for k in REQUIRED_TOP_LEVEL if k not in raw]
    if missing: raise StateSchemaError(f"Missing required Official State field(s): {missing}")
    p=raw["player"]
    if not isinstance(p,int) or isinstance(p,bool): raise StateSchemaError("player must be int")
    farms=raw["farms"]
    if not isinstance(farms,Sequence) or isinstance(farms,(str,bytes)) or not farms:
        raise StateSchemaError("farms must be a non-empty sequence")
    if p<0 or p>=len(farms): raise StateSchemaError("player index outside farms")
    for k in ("day","hour"):
        if not isinstance(raw[k],int) or isinstance(raw[k],bool): raise StateSchemaError(f"{k} must be int")
    for k in ("private","market","town"):
        if not isinstance(raw[k],Mapping): raise StateSchemaError(f"{k} must be mapping-like")

@dataclass(frozen=True)
class StateSnapshot:
    schema_version:str
    _raw:dict[str,Any]
    _canonical_json:str
    _canonical_hash:str

    @classmethod
    def bind(cls,official_state:Any)->"StateSnapshot":
        raw=_plain(official_state); validate_official_state(raw)
        owned=copy.deepcopy(raw)
        cj=json.dumps(owned,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        return cls(SCHEMA_VERSION,owned,cj,hashlib.sha256(cj.encode()).hexdigest())

    @property
    def canonical_hash(self): return self._canonical_hash
    def canonical_json(self): return self._canonical_json
    def raw(self): return copy.deepcopy(self._raw)
    def get(self,*path):
        v=self._raw
        for k in path:
            if isinstance(v,Mapping): v=v[k]
            elif isinstance(v,Sequence) and not isinstance(v,(str,bytes)): v=v[k]
            else: raise KeyError(path)
        return copy.deepcopy(v)

@dataclass(frozen=True)
class ShortPlanCandidate:
    plan_schema_version:str
    candidate_id:str
    kind:str
    target:dict[str,Any]
    completion:dict[str,Any]
    requirements:tuple[dict[str,Any],...]
    unknowns:tuple[str,...]
    source_paths:tuple[tuple[Any,...],...]

    def to_dict(self):
        return {
            "plan_schema_version":self.plan_schema_version,
            "candidate_id":self.candidate_id,
            "kind":self.kind,
            "target":copy.deepcopy(self.target),
            "completion":copy.deepcopy(self.completion),
            "requirements":[copy.deepcopy(x) for x in self.requirements],
            "unknowns":list(self.unknowns),
            "source_paths":[list(x) for x in self.source_paths],
        }

def bind_official_state(x): return StateSnapshot.bind(x)

def _candidate_id(state,kind,target):
    payload={"state":state.canonical_hash,"kind":kind,"target":_plain(target)}
    h=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()[:16]
    return f"{kind}:{h}"

def _positions_and_inventories(raw):
    invs=raw["private"].get("inventories",[]) or []
    for i,inv in enumerate(invs):
        if not isinstance(inv,Mapping): continue
        items={str(k):q for k,q in inv.items() if isinstance(q,(int,float)) and q>0}
        if items: yield i,dict(sorted(items.items()))

def _sellable_shed(raw):
    shed=raw["private"].get("shed",{}) or {}
    prices=raw["market"].get("prices",{}) or {}
    for item in sorted(prices):
        q=shed.get(item,0)
        if isinstance(q,(int,float)) and q>0: yield item,q

def _raw_empty_tiles(raw):
    p=raw["player"]; tiles=raw["farms"][p].get("tiles",[]) or []
    for y,row in enumerate(tiles):
        if not isinstance(row,Sequence) or isinstance(row,(str,bytes)): continue
        for x,tile in enumerate(row):
            if tile is None: yield x,y

def _seed_rows(raw):
    seeds=raw["private"].get("seeds",{}) or {}
    for crop in sorted(seeds):
        q=seeds.get(crop,0)
        if isinstance(q,(int,float)):
            yield crop,q

def generate_plans(state:StateSnapshot)->list[ShortPlanCandidate]:
    if not isinstance(state,StateSnapshot): raise TypeError("generate_plans expects StateSnapshot")
    raw=state.raw(); p=raw["player"]; plans=[]

    # Every unit currently carrying something gets a concrete delivery job.
    for unit_index,items in _positions_and_inventories(raw):
        target={"unit_index":unit_index,"carried_items":items}
        plans.append(ShortPlanCandidate(
            PLAN_SCHEMA_VERSION,_candidate_id(state,"deliver_carried_to_shed",target),
            "deliver_carried_to_shed",target,
            {"observable":"targeted carried items leave the unit inventory and any realized shed transfer is visible in Official World"},
            (
                {"raw_fact":"private.inventories[unit_index] contains positive quantity","unit_index":unit_index},
                {"official_condition":"unit must reach a shed-access position before DROP can realize"},
            ),
            (
                "Action sequence and routing are generated only by the projector.",
                "Realized transfer remains an Official World result.",
            ),
            (("private","inventories",unit_index),),
        ))

    # Every market-visible positive shed item gets its own sale-realization job.
    for item,qty in _sellable_shed(raw):
        target={"item":item,"available_quantity":qty}
        plans.append(ShortPlanCandidate(
            PLAN_SCHEMA_VERSION,_candidate_id(state,"realize_shed_stock_sale",target),
            "realize_shed_stock_sale",target,
            {"observable":"Official World shows target shed quantity decrease and self cash increase",
             "pre_cash":raw["farms"][p].get("money",0),"pre_shed_quantity":qty},
            (
                {"raw_fact":"private.shed[item] > 0","item":item,"quantity":qty},
                {"raw_fact":"market.prices exposes the same item","item":item},
            ),
            (
                "Exact realized price is left to the shared Official World.",
                "Opponent market actions are not predicted here.",
            ),
            (("private","shed",item),("market","prices",item),("farms",p,"money")),
        ))

    empties=list(_raw_empty_tiles(raw))
    seed_rows=list(_seed_rows(raw))

    # Existing seed + raw empty tile -> establish_plant.
    for crop,qty in seed_rows:
        if qty<=0: continue
        for x,y in empties:
            target={"crop":crop,"tile":[x,y]}
            plans.append(ShortPlanCandidate(
                PLAN_SCHEMA_VERSION,_candidate_id(state,"establish_plant",target),
                "establish_plant",target,
                {"observable":"Official World shows target raw tile as a PLANT of the target crop"},
                (
                    {"raw_fact":"private.seeds[crop] > 0","crop":crop,"quantity":qty},
                    {"raw_fact":"farms[player].tiles[y][x] is None","tile":[x,y]},
                    {"official_condition":"a farmer/hand must be assigned to reach the target tile and PLANT"},
                ),
                (
                    "Plant establishment completes only this ShortPlan; economic recovery is not claimed.",
                ),
                (("private","seeds",crop),("farms",p,"tiles",y,x)),
            ))

    # Missing seed for a specific future establish_plant(crop,tile) -> targeted preparation.
    # required seed quantity is exactly one for one target planting job.
    for crop,current_qty in seed_rows:
        required=1
        missing=max(0,required-int(current_qty))
        if missing<=0:
            continue
        for x,y in empties:
            target={
                "crop":crop,
                "tile":[x,y],
                "required_seed_quantity":required,
                "current_seed_quantity":int(current_qty),
                "missing_seed_quantity":missing,
            }
            plans.append(ShortPlanCandidate(
                PLAN_SCHEMA_VERSION,_candidate_id(state,"prepare_for_plant",target),
                "prepare_for_plant",target,
                {
                    "observable":"the exact establish_plant(crop,tile) candidate becomes present after Official World processing"
                },
                (
                    {"raw_fact":"farms[player].tiles[y][x] is None","tile":[x,y]},
                    {"raw_fact":"private.seeds[crop] is below one required seed","crop":crop,
                     "current_quantity":int(current_qty),"required_quantity":required,
                     "missing_quantity":missing},
                    {"official_condition":"BUY_SEED must realize in Official World"},
                ),
                (
                    "This preparation does not claim that planting succeeds.",
                    "Purchase success and any cash limitation are observed from Official World, not assumed here.",
                ),
                (("private","seeds",crop),("farms",p,"tiles",y,x),("farms",p,"money")),
            ))

    return plans
