"""State-grounded work for Strong Model v0.

This module does not choose a global policy.  It exposes short jobs from the
Official State and can project one job into one turn.  Persistent purpose is
represented by Job.key/target; the next turn re-materializes that purpose from
the new Official State.

No fixed crop ratio, day cutoff, occupancy threshold, cash reserve, worker cap,
or weighted score is defined here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from kaggle_environments.envs.kaggriculture import kaggriculture as rules


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

    def spec(self)->dict[str,Any]:
        return {
            "key":self.key,
            "kind":self.kind,
            "category":self.category,
            "target":copy.deepcopy(self.target),
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


def _future_asset_value(raw:dict[str,Any],xy:tuple[int,int],tile:dict[str,Any],final_day:int)->float:
    day=int(raw["day"]); prices=raw["market"]["prices"]
    if tile.get("kind")=="PLANT":
        crop=str(tile["crop"])
        return crop_future_units(crop,int(tile.get("planted_day",day)),day,final_day,tile)*float(prices[crop])
    if tile.get("animal"):
        animal=str(tile["animal"]); product=str(rules.ANIMALS[animal]["product"])
        return animal_future_units(animal,int(tile.get("placed_day",day)),day,final_day)*float(prices[product])
    return 0.0


def fresh_jobs(raw:dict[str,Any],episode_steps:int,turns_per_day:int,board_size:int)->list[Job]:
    """Enumerate observable work.  Hints are direct Cash deltas from Official costs/timing."""
    day=int(raw["day"]); step=int(raw.get("step",day*turns_per_day+int(raw["hour"])))
    final_day=(episode_steps-2)//turns_per_day
    prices=raw["market"]["prices"]
    jobs:list[Job]=[]

    for xy,t in all_tiles(raw):
        if not isinstance(t,dict): continue

        if t.get("kind")=="PLANT":
            crop=str(t["crop"]); planted=int(t.get("planted_day",day))
            streak=int(t.get("consecutive_unwatered",0) or 0)
            if not bool(t.get("watered_today",False)):
                value=0.0
                spec=rules.CROPS[crop]; age=day-planted
                if streak>=1:
                    value=_future_asset_value(raw,xy,t,final_day)
                elif not spec["ongoing"]:
                    start=(int(spec["max_yield_day"])+1)//2
                    if start<=age<=int(spec["max_yield_day"]):
                        bonus=2 if int(t.get("fertilized_until_day",-1) or -1)>=day else 1
                        value=bonus*float(prices[crop])
                key=f"water:{day}:{xy[0]}:{xy[1]}:{crop}"
                jobs.append(Job(key,"water_plant","maintenance",{"tile":list(xy),"crop":crop,"day":day},value,0.0))

            yld=int(t.get("yield_units",0) or 0)
            if yld>0 and day-planted>=int(rules.CROPS[crop]["first_yield_day"]):
                now=yld*float(prices[crop])
                future=max(yld,crop_future_units(crop,planted,day,final_day,t))
                wait_gain=max(0,future-yld)*float(prices[crop]) if not rules.CROPS[crop]["ongoing"] else 0.0
                key=f"harvest:plant:{xy[0]}:{xy[1]}:{crop}"
                jobs.append(Job(key,"harvest","recovery",{"tile":list(xy),"item":crop},now-wait_gain,0.0))

        if t.get("animal"):
            animal=str(t["animal"]); product=str(rules.ANIMALS[animal]["product"])
            streak=int(t.get("consecutive_unfed",0) or 0)
            if not bool(t.get("fed_today",False)):
                value=_future_asset_value(raw,xy,t,final_day) if streak>=1 else 0.0
                key=f"feed:{day}:{xy[0]}:{xy[1]}:{animal}"
                jobs.append(Job(key,"feed_animal","maintenance",{"tile":list(xy),"animal":animal,"day":day},value,0.0))
            if bool(t.get("fed_today",False)) and not bool(t.get("cared_today",False)):
                units=animal_future_units(animal,int(t.get("placed_day",day)),day,final_day)
                value=float(prices[product]) if units>0 else 0.0
                key=f"care:{day}:{xy[0]}:{xy[1]}:{animal}"
                jobs.append(Job(key,"care_animal","maintenance",{"tile":list(xy),"animal":animal,"day":day},value,0.0))
            yld=int(t.get("yield_units",0) or 0)
            if yld>0:
                key=f"harvest:animal:{xy[0]}:{xy[1]}:{animal}"
                jobs.append(Job(key,"harvest","recovery",{"tile":list(xy),"item":product},yld*float(prices[product]),0.0))

    for i,inv in enumerate(inventories(raw)):
        value=sum(int(inv.get(item,0) or 0)*float(prices.get(item,0)) for item in PRODUCTS)
        if value>0:
            jobs.append(Job(f"deliver:{i}","deliver","recovery",{"unit_index":i},value,0.0))

    # One concrete production-start target per asset type.  Locked targets are
    # only in the next legal quadrant, so a single BUY_LAND can make them valid.
    pos=positions(raw)
    unlocked=[xy for xy,t in all_tiles(raw) if t is None]
    nq=next_quadrant(raw)
    locked=[xy for xy,t in all_tiles(raw) if t=="LOCKED" and nq and quadrant(xy[0],xy[1],board_size)==nq]
    def nearest(cells):
        return min(cells,key=lambda xy:(min(distance(p,xy) for p in pos),xy)) if cells else None

    seeds=raw["private"].get("seeds",{}) or {}
    for crop in CROPS:
        for category,cells in (("production_start",unlocked),("expansion",locked)):
            xy=nearest(cells)
            if xy is None: continue
            need_seed=int(seeds.get(crop,0) or 0)<=0
            dist=min(distance(p,xy) for p in pos)
            plant_step=step+max(dist,1 if need_seed else 0)
            plant_day=plant_step//turns_per_day
            units=crop_future_units(crop,plant_day,day,final_day,None)
            if units<=0: continue
            seed_cost=0.0 if not need_seed else float(rules.CROPS[crop]["seed"])
            lc=float(land_cost(raw) or 0) if category=="expansion" else 0.0
            central=units*float(prices[crop])-seed_cost-lc
            key=f"establish_crop:{crop}:{xy[0]}:{xy[1]}"
            jobs.append(Job(key,"establish_crop",category,{"tile":list(xy),"crop":crop},central,-seed_cost-lc))

    structures={"COOP":[],"PASTURE":[]}
    for xy,t in all_tiles(raw):
        if isinstance(t,dict) and t.get("kind") in structures and not t.get("animal"):
            structures[str(t["kind"])].append(xy)

    for animal in ANIMALS:
        struct=str(rules.ANIMALS[animal]["structure"])
        for category,cells in (("production_start",structures[struct] or unlocked),("expansion",locked)):
            xy=nearest(cells)
            if xy is None: continue
            purchase=0.0 if owned(raw,animal)>0 else float(rules.ANIMALS[animal]["cost"])
            lc=float(land_cost(raw) or 0) if category=="expansion" else 0.0
            placed_day=(step+2+distance(nearest_shed(xy,board_size),xy))//turns_per_day
            units=animal_future_units(animal,placed_day,placed_day,final_day)
            if units<=0: continue
            product=str(rules.ANIMALS[animal]["product"])
            feed_days=max(0,final_day-placed_day+1)
            feed_cost=feed_days*float(prices["WHEAT"])
            central=units*float(prices[product])-purchase-lc-feed_cost
            key=f"establish_animal:{animal}:{xy[0]}:{xy[1]}"
            jobs.append(Job(key,"establish_animal",category,{"tile":list(xy),"animal":animal},central,-purchase-lc-feed_cost))

    shed=raw["private"].get("shed",{}) or {}
    params=(raw.get("market",{}) or {}).get("params")
    for item in PRODUCTS:
        q=int(shed.get(item,0) or 0)
        if q<=0: continue

        # "Partial sale" candidates are placed at actual marginal-price block
        # boundaries of the Official rounded price curve, plus 1 and all stock.
        # No arbitrary half/percentage quantity is introduced.
        level=int(raw["market"]["inventory"][item])
        last_price=None
        revenue=0.0
        candidates={1,q}
        revenues={}
        for n in range(1,q+1):
            price=float(rules.market_price(item,level,params))
            if last_price is not None and price!=last_price:
                candidates.add(n-1)
            revenue+=price
            revenues[n]=revenue
            if price>rules.PRICE_FLOOR: level+=1
            last_price=price

        for n in sorted(candidates):
            key=f"sell:{item}:{n}"
            jobs.append(Job(key,"sell_stock","trade",{"item":item,"qty":n},
                            float(revenues[n]),n*float(rules.PRICE_FLOOR),False,True))
    return jobs


def still_needed(spec:dict[str,Any],raw:dict[str,Any])->bool:
    kind=str(spec["kind"]); target=spec["target"]
    if kind in ("sell_stock","hire"): return False
    if kind=="deliver":
        idx=int(target["unit_index"]); invs=inventories(raw)
        return idx<len(invs) and any(int(invs[idx].get(p,0) or 0)>0 for p in PRODUCTS)
    xy=tuple(target.get("tile",()))
    t=tile_at(raw,xy)
    if kind=="water_plant":
        return int(raw["day"])==int(target["day"]) and isinstance(t,dict) and t.get("kind")=="PLANT" and str(t.get("crop"))==str(target["crop"]) and not bool(t.get("watered_today",False))
    if kind=="feed_animal":
        return int(raw["day"])==int(target["day"]) and isinstance(t,dict) and str(t.get("animal"))==str(target["animal"]) and not bool(t.get("fed_today",False))
    if kind=="care_animal":
        return int(raw["day"])==int(target["day"]) and isinstance(t,dict) and str(t.get("animal"))==str(target["animal"]) and not bool(t.get("cared_today",False))
    if kind=="harvest":
        return isinstance(t,dict) and int(t.get("yield_units",0) or 0)>0
    if kind=="establish_crop":
        if isinstance(t,dict) and t.get("kind")=="PLANT" and str(t.get("crop"))==str(target["crop"]): return False
        return t in (None,"LOCKED") or (isinstance(t,dict) and t.get("kind")=="WEED")
    if kind=="establish_animal":
        if isinstance(t,dict) and str(t.get("animal"))==str(target["animal"]): return False
        struct=str(rules.ANIMALS[str(target["animal"])]["structure"])
        return t in (None,"LOCKED") or (isinstance(t,dict) and t.get("kind")==struct and not t.get("animal"))
    return False


def materialize_active(spec:dict[str,Any],raw:dict[str,Any])->Job|None:
    if not still_needed(spec,raw): return None
    return Job(str(spec["key"]),str(spec["kind"]),str(spec["category"]),copy.deepcopy(spec["target"]),0.0,0.0,True,False)


def preferred_units(job:Job,raw:dict[str,Any],free:set[int],board_size:int)->list[int]:
    pos=positions(raw); invs=inventories(raw)
    if job.kind=="deliver":
        idx=int(job.target["unit_index"])
        return [idx] if idx in free and idx<len(pos) else []
    xy=tuple(job.target.get("tile",()))
    if job.kind=="feed_animal":
        carriers=[i for i in free if int(invs[i].get("WHEAT",0) or 0)>0]
        if carriers:return sorted(carriers,key=lambda i:(distance(pos[i],xy),i))
        return sorted(free,key=lambda i:(distance(pos[i],nearest_shed(pos[i],board_size)),i))
    if job.kind=="establish_animal":
        a=str(job.target["animal"])
        carriers=[i for i in free if int(invs[i].get(a,0) or 0)>0]
        if carriers:return sorted(carriers,key=lambda i:(distance(pos[i],xy),i))
        if int((raw["private"].get("shed",{}) or {}).get(a,0) or 0)>0:
            return sorted(free,key=lambda i:(distance(pos[i],nearest_shed(pos[i],board_size)),i))
    return sorted(free,key=lambda i:(distance(pos[i],xy),i))


def unit_action(job:Job,raw:dict[str,Any],idx:int,board_size:int)->tuple[list[Any],list[list[Any]]]:
    pos=positions(raw)[idx]; inv=inventories(raw)[idx]; kind=job.kind
    if kind=="deliver":
        access=nearest_shed(pos,board_size)
        return (["DROP"] if tuple(pos)==access else move_toward(pos,access),[])
    xy=tuple(job.target.get("tile",()))
    if kind=="water_plant": return (["WATER"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="care_animal": return (["CARE"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="harvest": return (["HARVEST"] if tuple(pos)==xy else move_toward(pos,xy),[])
    if kind=="feed_animal":
        if int(inv.get("WHEAT",0) or 0)>0:
            return (["FEED"] if tuple(pos)==xy else move_toward(pos,xy),[])
        access=nearest_shed(pos,board_size)
        if int((raw["private"].get("shed",{}) or {}).get("WHEAT",0) or 0)>0:
            return (["PICKUP","WHEAT",1] if tuple(pos)==access else move_toward(pos,access),[])
        return (move_toward(pos,access) if tuple(pos)!=access else ["PASS"],[["BUY_PRODUCT","WHEAT",1]])
    if kind=="establish_crop":
        crop=str(job.target["crop"]); t=tile_at(raw,xy); market=[]
        if t=="LOCKED": market.append(["BUY_LAND"])
        if int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)<=0: market.append(["BUY_SEED",crop,1])
        if t=="LOCKED": return (move_toward(pos,xy),market)
        if isinstance(t,dict) and t.get("kind")=="WEED": return (["DIG"] if tuple(pos)==xy else move_toward(pos,xy),market)
        if t is None and int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)>0:
            return (["PLANT",crop] if tuple(pos)==xy else move_toward(pos,xy),market)
        return (move_toward(pos,xy) if tuple(pos)!=xy else ["PASS"],market)
    if kind=="establish_animal":
        animal=str(job.target["animal"]); struct=str(rules.ANIMALS[animal]["structure"]); t=tile_at(raw,xy); market=[]
        if t=="LOCKED": market.append(["BUY_LAND"])
        if owned(raw,animal)<=0: market.append(["BUY_ANIMAL",animal,1])
        if t=="LOCKED": return (move_toward(pos,xy),market)
        if t is None:
            op="BUILD_COOP" if struct=="COOP" else "BUILD_PASTURE"
            return ([op] if tuple(pos)==xy else move_toward(pos,xy),market)
        if isinstance(t,dict) and t.get("kind")==struct and not t.get("animal"):
            if int(inv.get(animal,0) or 0)>0:
                return (["PLACE",animal,1] if tuple(pos)==xy else move_toward(pos,xy),market)
            access=nearest_shed(pos,board_size)
            if int((raw["private"].get("shed",{}) or {}).get(animal,0) or 0)>0:
                return (["PICKUP",animal,1] if tuple(pos)==access else move_toward(pos,access),market)
            return (move_toward(pos,access) if tuple(pos)!=access else ["PASS"],market)
        return (["PASS"],market)
    return (["PASS"],[])


def market_only(job:Job)->list[list[Any]]:
    if job.kind=="sell_stock": return [["SELL",str(job.target["item"]),int(job.target["qty"])]]
    if job.kind=="hire": return [["HIRE"]]
    return []