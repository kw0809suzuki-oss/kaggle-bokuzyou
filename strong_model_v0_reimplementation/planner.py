"""Bundle formation and terminal-Cash comparison for Strong Model v0.

The planner compares:
- continuation: currently active work only
- one minimal new job from each available category
- one combined bundle containing those category representatives

Every bundle is projected for exactly one turn using Official actor/market/town
processing with no guessed opponent action.  The post-state is then evaluated
under the same terminal-Cash assumptions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from .jobs import (
    ANIMALS,CROPS,PRODUCTS,Job,all_tiles,animal_future_units,crop_future_units,
    distance,fresh_jobs,inventories,land_cost,market_only,materialize_active,
    nearest_shed,positions,preferred_units,unit_action,
)


@dataclass(frozen=True)
class Settings:
    episodeSteps:int=720
    turnsPerDay:int=24
    boardSize:int=10
    shedCapacity:int=100
    maxMarketOrdersPerTurn:int=10
    farmHandCostMult:int=1
    townShopSellInterval:int=4
    townCenterSellInterval:int=24


@dataclass(frozen=True)
class Envelope:
    strict_cash:float
    central_cash:float


@dataclass
class Bundle:
    action:dict[str,Any]
    scheduled:list[str]
    projected:dict[str,Any]
    envelope:Envelope
    immediate_cash:float


def settings_from(configuration:Any)->Settings:
    values={}
    for name in Settings.__dataclass_fields__:
        if configuration is None: continue
        if isinstance(configuration,dict) and name in configuration:
            values[name]=int(configuration[name])
        elif hasattr(configuration,name):
            values[name]=int(getattr(configuration,name))
    return Settings(**values)


def step_of(raw:dict[str,Any],cfg:Settings)->int:
    return int(raw.get("step",int(raw["day"])*cfg.turnsPerDay+int(raw["hour"])) or 0)


def empty_action(raw:dict[str,Any])->dict[str,Any]:
    p=int(raw["player"])
    return {
        "farmer":["PASS"],
        "hands":[["PASS"] for _ in (raw["farms"][p].get("hands",[]) or [])],
        "market":[],
    }


def set_unit(action:dict[str,Any],idx:int,a:list[Any])->None:
    if idx==0: action["farmer"]=a
    else: action["hands"][idx-1]=a


def _opposite_product_conflict(existing:list[list[Any]],orders:list[list[Any]])->bool:
    sells={str(o[1]) for o in existing if len(o)>=2 and o[0]=="SELL"}
    buys={str(o[1]) for o in existing if len(o)>=2 and o[0]=="BUY_PRODUCT"}
    for o in orders:
        if len(o)<2: continue
        item=str(o[1])
        if o[0]=="SELL" and item in buys:return True
        if o[0]=="BUY_PRODUCT" and item in sells:return True
    return False


def schedule(raw:dict[str,Any],cfg:Settings,jobs:list[Job],forced:set[str]|None=None)->tuple[dict[str,Any],list[str]]:
    """Compose simultaneous work without fixed job-type priority.

    Ordering is by terminal-Cash hint.  Active work wins only exact ties.
    A forced alternative is merely guaranteed consideration before the
    continuation work; the whole bundle is still rejected later if its
    terminal envelope is worse.
    """
    forced=forced or set()
    action=empty_action(raw)
    free=set(range(len(positions(raw))))
    scheduled=[]
    seed_claims={c:0 for c in CROPS}
    land_order_present=False

    def rank(j:Job):
        return (
            0 if j.key in forced else 1,
            -float(j.central_delta),
            -float(j.strict_delta),
            0 if j.active else 1,
            j.key,
        )

    for job in sorted(jobs,key=rank):
        if job.kind in ("sell_stock","hire"):
            orders=market_only(job)
            if len(action["market"])+len(orders)>cfg.maxMarketOrdersPerTurn: continue
            if _opposite_product_conflict(action["market"],orders): continue
            action["market"].extend(copy.deepcopy(orders))
            scheduled.append(job.key)
            continue

        candidates=preferred_units(job,raw,free,cfg.boardSize)
        if not candidates: continue
        for idx in candidates:
            a,orders=unit_action(job,raw,idx,cfg.boardSize)
            if len(action["market"])+len(orders)>cfg.maxMarketOrdersPerTurn: continue
            if _opposite_product_conflict(action["market"],orders): continue
            if any(o and o[0]=="BUY_LAND" for o in orders) and land_order_present: continue
            if a and a[0]=="PLANT":
                crop=str(a[1])
                available=int((raw["private"].get("seeds",{}) or {}).get(crop,0) or 0)
                if seed_claims[crop]+1>available: continue

            set_unit(action,idx,a)
            action["market"].extend(copy.deepcopy(orders))
            if any(o and o[0]=="BUY_LAND" for o in orders): land_order_present=True
            if a and a[0]=="PLANT": seed_claims[str(a[1])]+=1
            free.remove(idx)
            scheduled.append(job.key)
            break

    # Same-turn sales are permitted to fund later fixed-cost purchases.
    # Preserve relative order within SELL and non-SELL groups. Same-item
    # BUY_PRODUCT/SELL pairs were excluded above.
    sells=[o for o in action["market"] if o and o[0]=="SELL"]
    other=[o for o in action["market"] if not o or o[0]!="SELL"]
    action["market"]=(sells+other)[:cfg.maxMarketOrdersPerTurn]
    return action,scheduled


def _project_one_turn(raw:dict[str,Any],action:dict[str,Any],cfg:Settings)->dict[str,Any]:
    """Official one-turn self projection; opponent future action is left unknown."""
    out=copy.deepcopy(raw); p=int(raw["player"])
    farms=copy.deepcopy(raw["farms"])
    private=copy.deepcopy(raw["private"])
    market=copy.deepcopy(raw["market"])
    town=copy.deepcopy(raw["town"])
    day=int(raw["day"]); step=step_of(raw,cfg)

    acts=[action.get("farmer",["PASS"]),*list(action.get("hands",[]) or [])]
    demand={}
    for a in acts:
        if isinstance(a,list) and len(a)>=2 and a[0]=="PLANT":
            demand[str(a[1])]=demand.get(str(a[1]),0)+1
    blocked={c for c,n in demand.items() if n>int((private.get("seeds",{}) or {}).get(c,0) or 0)}
    for i,a in enumerate(acts):
        aa=["PASS"] if isinstance(a,list) and len(a)>=2 and a[0]=="PLANT" and str(a[1]) in blocked else a
        rules._apply_unit_action(farms[p],private,i,aa,cfg.boardSize,day,cfg.turnsPerDay,cfg.shedCapacity)

    states=[]
    for seat in range(len(farms)):
        priv=private if seat==p else rules._new_private()
        act=action if seat==p else {"farmer":["PASS"],"hands":[],"market":[]}
        obs=SimpleNamespace(market=market,farms=farms,private=priv,town=town)
        states.append(SimpleNamespace(observation=obs,action=act))
    env=SimpleNamespace(configuration=SimpleNamespace(
        boardSize=cfg.boardSize,
        maxMarketOrdersPerTurn=cfg.maxMarketOrdersPerTurn,
        farmHandCostMult=cfg.farmHandCostMult,
        shedCapacity=cfg.shedCapacity,
        townShopSellInterval=cfg.townShopSellInterval,
        townCenterSellInterval=cfg.townCenterSellInterval,
    ))
    rules._process_market(states,env)
    rules._town_consume(env,states,step)
    rules._decay_plants(farms[p],step)

    if (step+1)%cfg.turnsPerDay==0:
        rules._daily_refresh_plants(farms[p],day,cfg.turnsPerDay)
        rules._daily_refresh_animals(farms[p],day)
        rules._drop_inventories_to_shed(private,cfg.shedCapacity)
        farms[p]["farmer"]=list(rules._default_spawn(cfg.boardSize))
        farms[p]["hands"]=[]
        farms[p]["hires_today"]=0
        private["inventories"]=[{}]
        # Random future weeds and shop unlocks are intentionally not predicted.

    nxt=step+1
    out["farms"]=farms; out["private"]=private; out["market"]=market; out["town"]=town
    out["step"]=nxt; out["day"]=nxt//cfg.turnsPerDay; out["hour"]=nxt%cfg.turnsPerDay
    return out


def _marginal_sale(item:str,qty:int,inventory:int,params:dict[str,Any]|None)->float:
    level=int(inventory); total=0.0
    for _ in range(max(0,int(qty))):
        price=float(rules.market_price(item,level,params))
        total+=price
        if price>rules.PRICE_FLOOR: level+=1
    return total


def terminal_envelope(raw:dict[str,Any],cfg:Settings)->Envelope:
    """Common terminal-Cash range used for every bundle.

    Strict: current Cash plus already-owned sellable products at Official price
    floor when carried stock can still physically reach shed access.

    Central: current Cash plus current public-price marginal value of sellable
    stock and Official-timing future output from assets/seeds that can still
    mature. Future random shops and opponent actions are not predicted.
    """
    p=int(raw["player"]); farm=raw["farms"][p]; private=raw["private"]
    step=step_of(raw,cfg); final_action=cfg.episodeSteps-2
    final_day=final_action//cfg.turnsPerDay
    cash=float(farm.get("money",0) or 0)
    prices=raw["market"]["prices"]; market_inv=raw["market"]["inventory"]
    params=(raw.get("market",{}) or {}).get("params")

    strict_qty={item:int((private.get("shed",{}) or {}).get(item,0) or 0) for item in PRODUCTS}
    for pos,inv in zip(positions(raw),inventories(raw)):
        if step+distance(pos,nearest_shed(pos,cfg.boardSize))<=final_action:
            for item in PRODUCTS: strict_qty[item]+=int(inv.get(item,0) or 0)
    strict=cash+sum(strict_qty.values())*float(rules.PRICE_FLOOR)

    qty={item:strict_qty[item] for item in PRODUCTS}
    day=int(raw["day"])
    feed_days=0

    for xy,t in all_tiles(raw):
        if not isinstance(t,dict): continue
        if t.get("kind")=="PLANT":
            crop=str(t["crop"])
            future=crop_future_units(crop,int(t.get("planted_day",day)),day,final_day,t)
            # Current carried/shed does not include standing yield; future is the
            # whole remaining plant output under successful required maintenance.
            qty[crop]+=max(0,int(future))
        elif t.get("animal"):
            animal=str(t["animal"]); product=str(rules.ANIMALS[animal]["product"])
            units=animal_future_units(animal,int(t.get("placed_day",day)),day,final_day)
            qty[product]+=max(0,int(units))
            if units>0: feed_days+=max(0,final_day-day+(0 if bool(t.get("fed_today",False)) else 1))

    # Seeds receive no mark-to-market value. They contribute only when a concrete
    # observed empty tile exists and Official timing allows maturity.
    empty=[xy for xy,t in all_tiles(raw) if t is None]
    seeds=private.get("seeds",{}) or {}
    seed_rows=[]
    for crop in CROPS:
        seed_rows.extend([crop]*int(seeds.get(crop,0) or 0))
    for crop,xy in zip(seed_rows,empty):
        d=min(distance(pos,xy) for pos in positions(raw))
        plant_day=(step+d)//cfg.turnsPerDay
        qty[crop]+=crop_future_units(crop,plant_day,day,final_day,None)

    central=cash
    for item in PRODUCTS:
        central+=_marginal_sale(item,qty[item],int(market_inv[item]),params)

    # Feed is an opportunity cost. Current public WHEAT price is the common
    # central assumption; no future price path is invented.
    central-=feed_days*float(prices["WHEAT"])
    return Envelope(float(strict),float(central))


def plan_bundle(raw:dict[str,Any],cfg:Settings,jobs:list[Job],forced:set[str]|None=None)->Bundle:
    action,scheduled=schedule(raw,cfg,jobs,forced)
    projected=_project_one_turn(raw,action,cfg)
    env=terminal_envelope(projected,cfg)
    p=int(projected["player"])
    return Bundle(action,scheduled,projected,env,float(projected["farms"][p].get("money",0) or 0))


def better(candidate:Bundle,base:Bundle)->bool:
    if candidate.envelope.strict_cash>base.envelope.strict_cash:return True
    if candidate.envelope.central_cash>base.envelope.central_cash:return True
    if candidate.envelope.strict_cash==base.envelope.strict_cash and candidate.envelope.central_cash==base.envelope.central_cash:
        return candidate.immediate_cash>base.immediate_cash
    return False


def choose(raw:dict[str,Any],cfg:Settings,active_specs:dict[str,dict[str,Any]])->tuple[Bundle,Job|None,list[Job],Bundle]:
    all_fresh=fresh_jobs(raw,cfg.episodeSteps,cfg.turnsPerDay,cfg.boardSize)
    fresh_map={j.key:j for j in all_fresh}

    active=[]
    for key,spec in active_specs.items():
        if key in fresh_map:
            j=fresh_map.pop(key)
            j.active=True
            active.append(j)
        else:
            j=materialize_active(spec,raw)
            if j is not None: active.append(j)

    fresh=list(fresh_map.values())

    # One HIRE is a minimal candidate only when a concrete positive unit job is
    # currently blocked by unit count. No fixed worker cap is introduced.
    positive_units=[j for j in active+fresh if j.kind not in ("sell_stock","hire") and j.central_delta>0]
    if len(positive_units)>len(positions(raw)) and int(raw["hour"])<cfg.turnsPerDay-1:
        p=int(raw["player"]); n=int(raw["farms"][p].get("hires_today",0) or 0)
        cost=float(rules._hire_cost(n,cfg.farmHandCostMult))
        enabled=sorted(positive_units,key=lambda j:(-j.central_delta,-j.strict_delta,j.key))[len(positions(raw))]
        fresh.append(Job("hire:one","hire","workforce",{},enabled.central_delta-cost,enabled.strict_delta-cost,False,True))

    continuation=plan_bundle(raw,cfg,active)

    grouped={}
    for j in fresh: grouped.setdefault(j.category,[]).append(j)
    representatives=[]
    for category,rows in sorted(grouped.items()):
        representatives.append(max(rows,key=lambda j:(j.central_delta,j.strict_delta,j.key)))

    alternatives=[]
    for j in representatives:
        alternatives.append((j,plan_bundle(raw,cfg,active+[j],{j.key})))

    # Also evaluate the actual simultaneous bundle across categories. This is
    # how parallel production/recovery can beat a sequence of individually good
    # actions without introducing a fixed category ordering.
    if len(representatives)>1:
        combined=plan_bundle(raw,cfg,active+representatives,{j.key for j in representatives})
        alternatives.append((None,combined))

    eligible=[row for row in alternatives if better(row[1],continuation)]
    if not eligible:return continuation,None,active,continuation

    chosen_job,chosen=max(eligible,key=lambda row:(
        row[1].envelope.strict_cash,
        row[1].envelope.central_cash,
        row[1].immediate_cash,
        "" if row[0] is None else row[0].key,
    ))
    return chosen,chosen_job,active,continuation