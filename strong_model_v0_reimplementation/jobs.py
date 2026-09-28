"""Strong Model v0 work layer.

Existing ShortPlan candidates remain the source for already-defined crop,
delivery, sale, and surface work.  This module only adds the work that the
existing candidate space does not express yet: animal maintenance/placement
and next-quadrant expansion.

No crop ratio, day cutoff, occupancy threshold, cash reserve, worker cap, or
weighted score is introduced here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from plan_generator_entrance_v0 import (
    ShortPlanCandidate,
    bind_official_state,
    generate_plans,
)
from short_plan_action_projector_v0 import semantic_plan_match


PRODUCTS=tuple(rules.PRODUCTS)
CROPS=tuple(rules.CROPS)
ANIMALS=tuple(rules.ANIMALS)


@dataclass
class Job:
    key:str
    kind:str
    category:str
    target:dict[str,Any]
    central_delta:float=0.0
    strict_delta:float=0.0
    active:bool=False
    one_turn:bool=False
    base_plan:ShortPlanCandidate|None=None

    def spec(self)->dict[str,Any]:
        return {
            "key":self.key,
            "kind":self.kind,
            "category":self.category,
            "target":copy.deepcopy(self.target),
            "base_kind":None if self.base_plan is None else self.base_plan.kind,
            "base_target":None if self.base_plan is None else copy.deepcopy(self.base_plan.target),
        }


def distance(a:Iterable[int],b:Iterable[int])->int:
    ax,ay=[int(v) for v in a]; bx,by=[int(v) for v in b]
    return abs(ax-bx)+abs(ay-by)


def shed_access(board_size:int)->list[tuple[int,int]]:
    h=board_size//2
    return [(h-1,h-1),(h,h-1),(h-1,h),(h,h)]


def nearest_shed(pos:Iterable[int],board_size:int)->tuple[int,int]:
    pts=shed_access(board_size)
    return min(pts,key=lambda p:(distance(pos,p),pts.index(p)))


def move_toward(pos:Iterable[int],target:Iterable[int])->list[str]:
    x,y=[int(v) for v in pos]; tx,ty=[int(v) for v in target]
    if x<tx:return ["EAST"]
    if x>tx:return ["WEST"]
    if y<ty:return ["SOUTH"]
    if y>ty:return ["NORTH"]
    return ["PASS"]


def positions(raw:dict[str,Any])->list[list[int]]:
    p=int(raw["player"]); farm=raw["farms"][p]
    return [list(farm["farmer"])]+[list(v) for v in (farm.get("hands",[]) or [])]


def inventories(raw:dict[str,Any])->list[dict[str,int]]:
    rows=[dict(v or {}) for v in (raw["private"].get("inventories",[]) or [])]
    while len(rows)<len(positions(raw)): rows.append({})
    return rows


def all_tiles(raw:dict[str,Any]):
    p=int(raw["player"])
    for y,row in enumerate(raw["farms"][p].get("tiles",[]) or []):
        for x,tile in enumerate(row or []):
            yield (x,y),tile


def tile_at(raw:dict[str,Any],xy:tuple[int,int]):
    p=int(raw["player"]); x,y=xy
    return raw["farms"][p]["tiles"][y][x]


def quadrant(x:int,y:int,board_size:int)->str:
    h=board_size//2
    return ("N" if y<h else "S")+("W" if x<h else "E")


def next_quadrant(raw:dict[str,Any])->str|None:
    p=int(raw["player"])
    idx=len(list(raw["farms"][p].get("unlocked_quadrants",[]) or []))-1
    return str(rules.LAND_ORDER[idx]) if 0<=idx<len(rules.LAND_ORDER) else None


def land_cost(raw:dict[str,Any])->int|None:
    p=int(raw["player"])
    idx=len(list(raw["farms"][p].get("unlocked_quadrants",[]) or []))-1
    return int(rules.LAND_PRICES[idx]) if 0<=idx<len(rules.LAND_PRICES) else None


def owned(raw:dict[str,Any],item:str)->int:
    q=int((raw["private"].get("shed",{}) or {}).get(item,0) or 0)
    for inv in inventories(raw): q+=int(inv.get(item,0) or 0)
    return q


def crop_future_units(crop:str,planted_day:int,current_day:int,final_day:int,tile:dict[str,Any]|None=None)->int:
    spec=rules.CROPS[crop]
    first=planted_day+int(spec["first_yield_day"])
    if first>final_day:return 0
    if not spec["ongoing"]:
        current=int((tile or {}).get("yield_units",1) or 0)
        max_day=planted_day+int(spec["max_yield_day"])
        if max_day>final_day:return max(1,current)
        units=current
        start=(int(spec["max_yield_day"])+1)//2
        age=max(0,current_day-planted_day)
        for a in range(max(age,start),int(spec["max_yield_day"])+1):
            units+=1
        return min(int(spec["max_yield"]),max(1,units))
    first_day=planted_day+int(spec["first_yield_day"])
    interval=max(1,int(spec["interval"]))
    return min(int(spec["max_yield"]),1+(final_day-first_day)//interval)


def animal_future_units(animal:str,placed_day:int,current_day:int,final_day:int)->int:
    spec=rules.ANIMALS[animal]
    first=placed_day+int(spec["first_yield_day"])
    if first>final_day:return 0
    interval=max(1,int(spec["interval"]))
    start=max(first,current_day)
    rem=(start-first)%interval
    if rem:start+=interval-rem
    if start>final_day:return 0
    return 1+(final_day-start)//interval


def _future_asset_value(raw:dict[str,Any],tile:dict[str,Any],final_day:int)->float:
    day=int(raw["day"]); prices=raw["market"]["prices"]
    if tile.get("kind")=="PLANT":
        crop=str(tile["crop"])
        return crop_future_units(crop,int(tile.get("planted_day",day)),day,final_day,tile)*float(prices[crop])
    if tile.get("animal"):
        animal=str(tile["animal"]); product=str(rules.ANIMALS[animal]["product"])
        return animal_future_units(animal,int(tile.get("placed_day",day)),day,final_day)*float(prices[product])
    return 0.0


def _base_key(plan:ShortPlanCandidate)->str:
    t=plan.target
    if plan.kind=="deliver_carried_to_shed":
        return f"base:deliver:{int(t['unit_index'])}"
    if plan.kind=="realize_shed_stock_sale":
        return f"base:sell:{t['item']}"
    if plan.kind in ("establish_plant","prepare_for_plant","maintain_plant_today","collect_plant_output","prepare_surface_for_plant"):
        xy=list(t.get("tile",[]))
        crop=str(t.get("crop",""))
        return f"base:{plan.kind}:{crop}:{xy[0]}:{xy[1]}"
    return f"base:{plan.kind}:{plan.candidate_id}"


def _base_job(plan:ShortPlanCandidate,raw:dict[str,Any],episode_steps:int,turns_per_day:int)->Job:
    day=int(raw["day"]); final_day=(episode_steps-2)//turns_per_day
    prices=raw["market"]["prices"]; t=plan.target
    central=0.0; strict=0.0; one_turn=False
    category="production_start"

    if plan.kind=="deliver_carried_to_shed":
        category="recovery"
        central=sum(int(q)*float(prices.get(item,0)) for item,q in dict(t["carried_items"]).items() if item in PRODUCTS)
    elif plan.kind=="realize_shed_stock_sale":
        category="trade"; one_turn=True
        central=int(t["available_quantity"])*float(prices[str(t["item"])])
        strict=int(t["available_quantity"])*float(rules.PRICE_FLOOR)
    elif plan.kind=="maintain_plant_today":
        category="maintenance"
        x,y=[int(v) for v in t["tile"]]
        tile=raw["farms"][raw["player"]]["tiles"][y][x]
        if isinstance(tile,dict):
            central=_future_asset_value(raw,tile,final_day)
    elif plan.kind=="collect_plant_output":
        category="recovery"
        central=int(t.get("available_yield_units",0) or 0)*float(prices[str(t["crop"])])
    elif plan.kind=="establish_plant":
        category="production_start"
        crop=str(t["crop"])
        central=crop_future_units(crop,day,day,final_day,None)*float(prices[crop])
    elif plan.kind=="prepare_for_plant":
        category="production_start"; one_turn=True
        crop=str(t["crop"]); seed_cost=int(t.get("missing_seed_quantity",1))*float(rules.CROPS[crop]["seed"])
        central=crop_future_units(crop,day,day,final_day,None)*float(prices[crop])-seed_cost
        strict=-seed_cost
    elif plan.kind=="prepare_surface_for_plant":
        category="production_start"
        crop=str(t["crop"])
        central=crop_future_units(crop,day,day,final_day,None)*float(prices[crop])

    return Job(_base_key(plan),plan.kind,category,copy.deepcopy(plan.target),central,strict,False,one_turn,plan)


def fresh_jobs(raw:dict[str,Any],episode_steps:int,turns_per_day:int,board_size:int)->list[Job]:
    """Existing ShortPlans first; extensions only where that vocabulary is absent."""
    snapshot=bind_official_state(raw)
    jobs=[_base_job(p,raw,episode_steps,turns_per_day) for p in generate_plans(snapshot)]
    day=int(raw["day"]); step=int(raw.get("step",day*turns_per_day+int(raw["hour"])))
    final_day=(episode_steps-2)//turns_per_day
    prices=raw["market"]["prices"]

    # Animal maintenance/output work is not present in the existing ShortPlan generator.
    for xy,tile in all_tiles(raw):
        if not (isinstance(tile,Mapping) and tile.get("animal")): continue
        animal=str(tile["animal"]); product=str(rules.ANIMALS[animal]["product"])
        if not bool(tile.get("fed_today",False)):
            value=_future_asset_value(raw,dict(tile),final_day) if int(tile.get("consecutive_unfed",0) or 0)>=1 else 0.0
            jobs.append(Job(f"ext:feed:{day}:{xy[0]}:{xy[1]}:{animal}","feed_animal","maintenance",
                            {"tile":list(xy),"animal":animal,"day":day},value,0.0))
        if bool(tile.get("fed_today",False)) and not bool(tile.get("cared_today",False)):
            units=animal_future_units(animal,int(tile.get("placed_day",day)),day,final_day)
            value=float(prices[product]) if units>0 else 0.0
            jobs.append(Job(f"ext:care:{day}:{xy[0]}:{xy[1]}:{animal}","care_animal","maintenance",
                            {"tile":list(xy),"animal":animal,"day":day},value,0.0))
        yld=int(tile.get("yield_units",0) or 0)
        if yld>0:
            jobs.append(Job(f"ext:harvest_animal:{xy[0]}:{xy[1]}:{animal}","harvest_animal","recovery",
                            {"tile":list(xy),"animal":animal,"item":product},yld*float(prices[product]),0.0))

    # Animal establishment on observed empty structure/ground.
    pos=positions(raw)
    unlocked=[xy for xy,t in all_tiles(raw) if t is None]
    structures={"COOP":[],"PASTURE":[]}
    for xy,t in all_tiles(raw):
        if isinstance(t,Mapping) and t.get("kind") in structures and not t.get("animal"):
            structures[str(t["kind"])].append(xy)
    def nearest(cells):
        return min(cells,key=lambda xy:(min(distance(p,xy) for p in pos),xy)) if cells else None

    for animal in ANIMALS:
        struct=str(rules.ANIMALS[animal]["structure"])
        xy=nearest(structures[struct] or unlocked)
        if xy is not None:
            purchase=0.0 if owned(raw,animal)>0 else float(rules.ANIMALS[animal]["cost"])
            placed_day=(step+2+distance(nearest_shed(xy,board_size),xy))//turns_per_day
            units=animal_future_units(animal,placed_day,placed_day,final_day)
            if units>0:
                product=str(rules.ANIMALS[animal]["product"])
                feed_days=max(0,final_day-placed_day+1)
                central=units*float(prices[product])-purchase-feed_days*float(prices["WHEAT"])
                jobs.append(Job(f"ext:establish_animal:{animal}:{xy[0]}:{xy[1]}","establish_animal","production_start",
                                {"tile":list(xy),"animal":animal},central,-purchase))

    # Expansion is the only crop surface absent from existing ShortPlans because
    # locked tiles are intentionally excluded there.  Offer the next legal
    # quadrant together with one concrete first asset.
    nq=next_quadrant(raw); lc=float(land_cost(raw) or 0)
    locked=[xy for xy,t in all_tiles(raw) if t=="LOCKED" and nq and quadrant(xy[0],xy[1],board_size)==nq]
    xy=nearest(locked)
    if xy is not None:
        seeds=raw["private"].get("seeds",{}) or {}
        for crop in CROPS:
            seed_cost=0.0 if int(seeds.get(crop,0) or 0)>0 else float(rules.CROPS[crop]["seed"])
            plant_day=(step+max(1,min(distance(p,xy) for p in pos)))//turns_per_day
            units=crop_future_units(crop,plant_day,day,final_day,None)
            if units>0:
                central=units*float(prices[crop])-lc-seed_cost
                jobs.append(Job(f"ext:expand_crop:{crop}:{xy[0]}:{xy[1]}","expand_crop","expansion",
                                {"tile":list(xy),"crop":crop},central,-lc-seed_cost))
        for animal in ANIMALS:
            placed_day=(step+2+distance(nearest_shed(xy,board_size),xy))//turns_per_day
            units=animal_future_units(animal,placed_day,placed_day,final_day)
            if units<=0: continue
            purchase=0.0 if owned(raw,animal)>0 else float(rules.ANIMALS[animal]["cost"])
            product=str(rules.ANIMALS[animal]["product"])
            feed_days=max(0,final_day-placed_day+1)
            central=units*float(prices[product])-lc-purchase-feed_days*float(prices["WHEAT"])
            jobs.append(Job(f"ext:expand_animal:{animal}:{xy[0]}:{xy[1]}","expand_animal","expansion",
                            {"tile":list(xy),"animal":animal},central,-lc-purchase))
    return jobs


def still_needed(spec:dict[str,Any],raw:dict[str,Any])->bool:
    base_kind=spec.get("base_kind")
    if base_kind:
        snapshot=bind_official_state(raw)
        for p in generate_plans(snapshot):
            if semantic_plan_match(p,kind=str(base_kind),target=dict(spec.get("base_target") or {})):
                return True
        return False

    kind=str(spec["kind"]); target=spec["target"]
    if kind in ("hire","sell"): return False
    if kind=="deliver":
        idx=int(target["unit_index"])
        return idx<len(inventories(raw)) and any(inventories(raw)[idx].values())
    xy=tuple(target.get("tile",()))
    tile=tile_at(raw,xy)
    if kind=="plant_crop":
        return tile is None or (isinstance(tile,dict) and tile.get("kind")=="WEED")
    if kind=="water":
        return isinstance(tile,dict) and tile.get("crop")==target["crop"] and not tile.get("watered_today")
    if kind=="harvest":
        return isinstance(tile,dict) and tile.get("crop")==target["crop"] and tile.get("yield_units",0)>0
    if kind=="feed_animal":
        return int(raw["day"])==int(target["day"]) and isinstance(tile,dict) and str(tile.get("animal"))==str(target["animal"]) and not bool(tile.get("fed_today",False))
    if kind=="care_animal":
        return int(raw["day"])==int(target["day"]) and isinstance(tile,dict) and str(tile.get("animal"))==str(target["animal"]) and not bool(tile.get("cared_today",False))
    if kind=="harvest_animal":
        return isinstance(tile,dict) and str(tile.get("animal"))==str(target["animal"]) and int(tile.get("yield_units",0) or 0)>0
    if kind in ("establish_animal","expand_animal"):
        if isinstance(tile,dict) and str(tile.get("animal"))==str(target["animal"]): return False
        struct=str(rules.ANIMALS[str(target["animal"])]["structure"])
        return tile in (None,"LOCKED") or (isinstance(tile,dict) and tile.get("kind")==struct and not tile.get("animal"))
    if kind=="expand_crop":
        crop=str(target["crop"])
        return not (isinstance(tile,dict) and tile.get("kind")=="PLANT" and str(tile.get("crop"))==crop)
    return False


def materialize_active(spec:dict[str,Any],raw:dict[str,Any])->Job|None:
    if not still_needed(spec,raw): return None
    if spec.get("base_kind"):
        snapshot=bind_official_state(raw)
        for p in generate_plans(snapshot):
            if semantic_plan_match(p,kind=str(spec["base_kind"]),target=dict(spec.get("base_target") or {})):
                j=_base_job(p,raw,720,24)
                j.key=str(spec["key"]); j.active=True
                return j
        return None
    return Job(str(spec["key"]),str(spec["kind"]),str(spec["category"]),copy.deepcopy(spec["target"]),0.0,0.0,True,False,None)


def preferred_units(job:Job,raw:dict[str,Any],free:set[int],board_size:int)->list[int]:
    pos=positions(raw); invs=inventories(raw)
    if job.kind=="deliver" or (job.base_plan is not None and job.base_plan.kind=="deliver_carried_to_shed"):
        idx=int(job.target["unit_index"])
        return [idx] if idx in free and idx<len(pos) else []

    xy=tuple(job.target.get("tile",()))
    if job.kind=="feed_animal":
        carriers=[i for i in free if int(invs[i].get("WHEAT",0) or 0)>0]
        if carriers:return sorted(carriers,key=lambda i:(distance(pos[i],xy),i))
        return sorted(free,key=lambda i:(distance(pos[i],nearest_shed(pos[i],board_size)),i))
    if job.kind in ("establish_animal","expand_animal"):
        a=str(job.target["animal"])
        carriers=[i for i in free if int(invs[i].get(a,0) or 0)>0]
        if carriers:return sorted(carriers,key=lambda i:(distance(pos[i],xy),i))
        if int((raw["private"].get("shed",{}) or {}).get(a,0) or 0)>0:
            return sorted(free,key=lambda i:(distance(pos[i],nearest_shed(pos[i],board_size)),i))
    return sorted(free,key=lambda i:(distance(pos[i],xy),i))


def unit_action(job:Job,raw:dict[str,Any],idx:int,board_size:int)->tuple[list[Any],list[list[Any]]]:
    pos=positions(raw)[idx]; inv=inventories(raw)[idx]

    if job.base_plan is not None:
        p=job.base_plan; kind=p.kind
        if kind=="deliver_carried_to_shed":
            access=nearest_shed(pos,board_size)
            return (["DROP"] if tuple(pos)==access else move_toward(pos,access),[])
        if kind=="establish_plant":
            xy=tuple(int(v) for v in p.target["tile"]); crop=str(p.target["crop"])
            return (["PLANT",crop] if tuple(pos)==xy else move_toward(pos,xy),[])
        if kind=="maintain_plant_today":
            xy=tuple(int(v) for v in p.target["tile"])
            return (["WATER"] if tuple(pos)==xy else move_toward(pos,xy),[])
        if kind=="collect_plant_output":
            xy=tuple(int(v) for v in p.target["tile"])
            return (["HARVEST"] if tuple(pos)==xy else move_toward(pos,xy),[])
        if kind=="prepare_surface_for_plant":
            xy=tuple(int(v) for v in p.target["tile"])
            return (["DIG"] if tuple(pos)==xy else move_toward(pos,xy),[])
        return (["PASS"],[])

    kind=job.kind; xy=tuple(job.target.get("tile",()))
    if kind=="deliver":
        access=nearest_shed(pos,board_size)
        return (["DROP"] if tuple(pos)==access else move_toward(pos,access),[])
    if kind=="water": return (["WATER"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="harvest": return (["HARVEST"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="plant_crop":
        crop=str(job.target["crop"]); tile=tile_at(raw,xy)
        orders=[]
        if int(raw["private"]["seeds"].get(crop,0))<=0:
            orders.append(["BUY_SEED",crop,1])
        if tuple(pos)!=xy: return move_toward(pos,xy),orders
        if isinstance(tile,dict) and tile.get("kind")=="WEED": return ["DIG"],orders
        return (["PLANT",crop] if not orders else ["PASS"]),orders
    if kind=="care_animal": return (["CARE"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="harvest_animal": return (["HARVEST"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="feed_animal":
        if int(inv.get("WHEAT",0) or 0)>0:
            return (["FEED"] if tuple(pos)==xy else move_toward(pos,xy),[])
        access=nearest_shed(pos,board_size)
        if int((raw["private"].get("shed",{}) or {}).get("WHEAT",0) or 0)>0:
            return (["PICKUP","WHEAT",1] if tuple(pos)==access else move_toward(pos,access),[])
        return (move_toward(pos,access) if tuple(pos)!=access else ["PASS"],[["BUY_PRODUCT","WHEAT",1]])

    if kind in ("establish_animal","expand_animal"):
        animal=str(job.target["animal"]); struct=str(rules.ANIMALS[animal]["structure"])
        tile=tile_at(raw,xy); market=[]
        if tile=="LOCKED": market.append(["BUY_LAND"])
        if owned(raw,animal)<=0: market.append(["BUY_ANIMAL",animal,1])
        if tile=="LOCKED": return (move_toward(pos,xy),market)
        if tile is None:
            op="BUILD_COOP" if struct=="COOP" else "BUILD_PASTURE"
            return ([op] if tuple(pos)==xy else move_toward(pos,xy),market)
        if isinstance(tile,dict) and tile.get("kind")==struct and not tile.get("animal"):
            if int(inv.get(animal,0) or 0)>0:
                return (["PLACE",animal,1] if tuple(pos)==xy else move_toward(pos,xy),market)
            access=nearest_shed(pos,board_size)
            if int((raw["private"].get("shed",{}) or {}).get(animal,0) or 0)>0:
                return (["PICKUP",animal,1] if tuple(pos)==access else move_toward(pos,access),market)
            return (move_toward(pos,access) if tuple(pos)!=access else ["PASS"],market)
        return (["PASS"],market)

    if kind=="expand_crop":
        crop=str(job.target["crop"]); tile=tile_at(raw,xy); market=[]
        if tile=="LOCKED": market.append(["BUY_LAND"])
        if int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)<=0:
            market.append(["BUY_SEED",crop,1])
        if tile=="LOCKED": return (move_toward(pos,xy),market)
        if tile is None and int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)>0:
            return (["PLANT",crop] if tuple(pos)==xy else move_toward(pos,xy),market)
        return (move_toward(pos,xy) if tuple(pos)!=xy else ["PASS"],market)

    return (["PASS"],[])


def market_only(job:Job)->list[list[Any]]:
    if job.base_plan is not None:
        if job.base_plan.kind=="realize_shed_stock_sale":
            return [["SELL",str(job.base_plan.target["item"]),int(job.base_plan.target["available_quantity"])]]
        if job.base_plan.kind=="prepare_for_plant":
            return [["BUY_SEED",str(job.base_plan.target["crop"]),int(job.base_plan.target["missing_seed_quantity"])]]
    if job.kind=="sell": return [["SELL",job.target["item"],job.target["quantity"]]]
    if job.kind=="hire": return [["HIRE"]]
    return []
