"""Executable cash recovery forecast; approximations are documented in REPAIR_NOTES.md."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from .jobs import (
    ANIMALS,CROPS,PRODUCTS,Job,all_tiles,animal_future_units,crop_future_units,
    distance,fresh_jobs,inventories,land_cost,market_only,materialize_active,
    nearest_shed,positions,preferred_units,unit_action,still_needed,
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


def schedule(raw, cfg, jobs, forced=None):
    """Hints order scarce workers; terminal rollouts decide the proposal."""
    forced = forced or set()
    action = empty_action(raw)
    free = set(range(len(positions(raw))))
    scheduled, chosen, tiles = [], {}, set()
    def rank(j):
        xy = j.target.get("tile")
        d = min(distance(pos, xy) for pos in positions(raw)) if xy else 0
        return (j.key in forced, j.central_delta / (d + 1), j.active, j.key)
    for job in sorted(jobs, key=rank, reverse=True):
        orders = market_only(job)
        if orders and (job.one_turn or job.kind == "hire"):
            if len(action["market"]) + len(orders) <= cfg.maxMarketOrdersPerTurn:
                action["market"].extend(copy.deepcopy(orders))
                scheduled.append(job.key)
            continue
        xy = tuple(job.target.get("tile", ()))
        if xy and xy in tiles:
            continue
        units = preferred_units(job, raw, free, cfg.boardSize)
        if not units:
            continue
        idx = units[0]
        a, orders = unit_action(job, raw, idx, cfg.boardSize)
        chosen[idx] = (job, a, orders)
        free.remove(idx)
        if xy:
            tiles.add(xy)
    seeds = dict(raw["private"].get("seeds", {}))
    pickups = dict(raw["private"].get("shed", {}))
    requested, land = set(), False
    for idx, (job, a, orders) in sorted(chosen.items()):
        if a[0] == "PLANT":
            item = a[1]
            if seeds.get(item, 0) <= 0:
                a = ["PASS"]
            else:
                seeds[item] -= 1
        if a[0] == "PICKUP":
            item, qty = a[1], int(a[2])
            qty = min(qty, int(pickups.get(item, 0)))
            a = ["PICKUP", item, qty] if qty else ["PASS"]
            pickups[item] = pickups.get(item, 0) - qty
        for order in orders:
            if order[0] == "BUY_SEED":
                if order[1] in requested:
                    continue
                requested.add(order[1])
            if order[0] == "BUY_LAND":
                if land:
                    continue
                land = True
            action["market"].append(order)
        set_unit(action, idx, a)
        scheduled.append(job.key)
    orders = action["market"]
    action["market"] = ([o for o in orders if o[0] == "SELL"] +
                        [o for o in orders if o[0] != "SELL"])[:cfg.maxMarketOrdersPerTurn]
    return action, scheduled


def _project_one_turn(raw:dict[str,Any],action:dict[str,Any],cfg:Settings,in_place:bool=False)->dict[str,Any]:
    """Official one-turn self projection; opponent future action is left unknown."""
    out=raw if in_place else copy.deepcopy(raw); p=int(raw["player"])
    farms=out["farms"]
    private=out["private"]
    market=out["market"]
    town=out["town"]
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
    if action.get("market"):
        rules._process_market(states,env)
    if step%max(1,cfg.townShopSellInterval)==0 or step%max(1,cfg.townCenterSellInterval)==0:
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


def operating_jobs(raw, cfg, harvest_now=False):
    """Generate maintenance and recovery work without enumerating investments."""
    day = int(raw['day'])
    final_day = (cfg.episodeSteps - 2) // cfg.turnsPerDay
    prices = raw['market']['prices']
    jobs = []
    for xy, t in all_tiles(raw):
        if not isinstance(t, dict):
            continue
        if t.get('kind') == 'PLANT':
            crop = t['crop']; spec = rules.CROPS[crop]
            age = day - int(t['planted_day']); units = int(t.get('yield_units', 0))
            future = crop_future_units(crop, int(t['planted_day']), day, final_day, t)
            growth = (not spec['ongoing'] and (int(spec['max_yield_day'])+1)//2 <= age <= int(spec['max_yield_day']) and units < int(spec['max_yield']))
            survival = int(t.get('consecutive_unwatered', 0)) >= 1
            target = {'tile':list(xy), 'crop':crop}
            if not t.get('watered_today') and (survival or growth) and future > 0:
                value = (max(units, future) if survival else 1) * prices[crop]
                jobs.append(Job(f'water:{xy}', 'water', 'maintenance', target, value, 0))
            if age >= int(spec['first_yield_day']) and units > 0 and (harvest_now or spec['ongoing'] or age >= int(spec['max_yield_day']) or day >= final_day):
                if harvest_now or not (growth and not t.get('watered_today')):
                    jobs.append(Job(f'harvest:{xy}', 'harvest', 'recovery', target, units*prices[crop], 0))
        elif t.get('animal'):
            animal = t['animal']; product = rules.ANIMALS[animal]['product']
            future = animal_future_units(animal, int(t['placed_day']), day, final_day)
            target = {'tile':list(xy), 'animal':animal, 'day':day}
            if not t.get('fed_today') and future > 0:
                value = prices[product] * (max(1, future) if t.get('consecutive_unfed', 0)>=1 else 1)
                jobs.append(Job(f'feed:{xy}:{day}', 'feed_animal', 'maintenance', target, value, 0))
            if t.get('fed_today') and not t.get('cared_today') and future > 0:
                jobs.append(Job(f'care:{xy}:{day}', 'care_animal', 'maintenance', target, prices[product], 0))
            if t.get('yield_units', 0)>0:
                jobs.append(Job(f'animal_harvest:{xy}', 'harvest_animal', 'recovery', target, t['yield_units']*prices[product], 0))
    for idx, inv in enumerate(inventories(raw)):
        value = sum(q*prices.get(item, 0) for item,q in inv.items() if item in PRODUCTS)
        if value>0:
            jobs.append(Job(f'deliver:{idx}', 'deliver', 'recovery', {'unit_index':idx}, value, 0))
    return jobs


def investment_jobs(raw, cfg, fresh):
    result = {}
    for j in fresh:
        if j.kind in ('prepare_for_plant','establish_plant','prepare_surface_for_plant'):
            crop = j.target['crop']; x,y = j.target['tile']
            job = Job(f'plant:{crop}:{x}:{y}', 'plant_crop', 'production_start', {'tile':[x,y], 'crop':crop}, j.central_delta, j.strict_delta)
            result[job.key] = job
        elif j.kind in ('establish_animal','expand_crop','expand_animal'):
            result[j.key] = j
    for xy,t in all_tiles(raw):
        if isinstance(t,dict) and t.get('kind')=='WEED':
            for crop in CROPS:
                value = crop_future_units(crop, raw['day'], raw['day'], (cfg.episodeSteps-2)//cfg.turnsPerDay)*raw['market']['prices'][crop]-rules.CROPS[crop]['seed']
                j = Job(f'plant:{crop}:{xy[0]}:{xy[1]}', 'plant_crop', 'production_start', {'tile':list(xy),'crop':crop}, value, -rules.CROPS[crop]['seed'])
                result[j.key] = j
    return list(result.values())


def _service_action(raw, cfg, commitments, forced=None, harvest_now=False):
    live = [j for j in commitments if still_needed(j.spec(),raw)]
    keys = {j.key for j in live}
    jobs = [j for j in operating_jobs(raw,cfg,harvest_now) if j.key not in keys]
    required = set()
    for j in live+jobs:
        if j.kind=='feed_animal': required.add('WHEAT')
        if j.kind in ('establish_animal','expand_animal'): required.add(j.target['animal'])
    reserved = {i for i,inv in enumerate(inventories(raw)) if any(inv.get(item,0)>0 for item in required)}
    jobs = [j for j in jobs if not (j.kind=='deliver' and j.target['unit_index'] in reserved)]
    action, scheduled = schedule(raw,cfg,live+jobs,forced)
    private = raw['private']
    acts = [action['farmer'], *action['hands']]
    if any(a[0] in ('DROP','PICKUP') for a in acts):
        private = copy.deepcopy(private)
        for idx,a in enumerate(acts):
            if a[0] in ('DROP','PICKUP'):
                rules._apply_unit_action(raw['farms'][raw['player']],private,idx,a,cfg.boardSize,raw['day'],cfg.turnsPerDay,cfg.shedCapacity)
    feed_need = sum(j.kind=='feed_animal' for j in live+jobs)
    sales = []
    for item,q in private.get('shed',{}).items():
        if item in PRODUCTS:
            q = int(q)-(feed_need if item=='WHEAT' else 0)
            if q>0: sales.append(['SELL',item,q])
    action['market'] = (sales+action['market'])[:cfg.maxMarketOrdersPerTurn]
    return action,scheduled,live,jobs


def rollout(raw, cfg, commitments=(), first_action=None, harvest_now=False, stress=False, trace=False):
    world = copy.deepcopy(raw)
    records = []; first = True
    final_action = cfg.episodeSteps-2
    if stress:
        # One adverse supply shock from visible opponent yield, not a lower bound.
        for seat,farm in enumerate(world['farms']):
            if seat==world['player']: continue
            for row in farm['tiles']:
                for tile in row:
                    if not isinstance(tile,dict): continue
                    item = tile.get('crop') or (rules.ANIMALS[tile['animal']]['product'] if tile.get('animal') else None)
                    if item in PRODUCTS:
                        world['market']['inventory'][item] += int(tile.get('yield_units',0))
        for item in PRODUCTS:
            world['market']['prices'][item] = rules.market_price(item,world['market']['inventory'][item],world['market'].get('params'))
    while step_of(world,cfg)<=final_action:
        action,_,live,jobs = _service_action(world,cfg,commitments,harvest_now=harvest_now)
        commitments = live
        if first and first_action is not None: action = copy.deepcopy(first_action)
        if first and stress:
            action['farmer']=['PASS']; action['hands']=[['PASS'] for _ in action['hands']]
        first = False
        if trace: records.append({'step':step_of(world,cfg),'cash':world['farms'][world['player']]['money'],'action':copy.deepcopy(action)})
        _project_one_turn(world,action,cfg,in_place=True)
        if not live and not jobs and not any(world['private'].get('shed',{}).values()) and not any(any(inv.values()) for inv in inventories(world)):
            if not any(isinstance(t,dict) and (t.get('kind')=='PLANT' or t.get('animal')) for _,t in all_tiles(world)):
                break
            # A new day creates new work. Never skip it at the boundary itself.
            if world['hour']==0: continue
            while world['hour']!=0 and step_of(world,cfg)<=final_action:
                action=empty_action(world)
                if trace: records.append({'step':step_of(world,cfg),'cash':world['farms'][world['player']]['money'],'action':action})
                _project_one_turn(world,action,cfg,in_place=True)
    return float(world['farms'][world['player']]['money']),records


def terminal_envelope(raw,cfg):
    central,_ = rollout(raw,cfg)
    stress,_ = rollout(raw,cfg,stress=True)
    return Envelope(stress,central)


def plan_bundle(raw,cfg,jobs,forced=None):
    investments = investment_jobs(raw,cfg,jobs)
    commitments = investments+[j for j in jobs if j.category not in ('production_start','expansion')]
    action,scheduled,_,_ = _service_action(raw,cfg,commitments,forced)
    projected = _project_one_turn(raw,action,cfg)
    central,_ = rollout(raw,cfg,commitments,first_action=action)
    strict,_ = rollout(raw,cfg,commitments,first_action=action,stress=True)
    return Bundle(action,scheduled,projected,Envelope(strict,central),float(projected['farms'][raw['player']]['money']))


def better(candidate,base,minimal_commitment=True):
    return candidate.envelope.central_cash>base.envelope.central_cash


def choose(raw,cfg,active_specs):
    fresh = fresh_jobs(raw,cfg.episodeSteps,cfg.turnsPerDay,cfg.boardSize)
    investments = investment_jobs(raw,cfg,fresh)
    lookup = {j.key:j for j in investments+fresh+operating_jobs(raw,cfg)}
    active = []
    for key,spec in active_specs.items():
        if not still_needed(spec,raw): continue
        j = lookup.get(key) or materialize_active(spec,raw)
        if j is not None:
            j.active=True; active.append(j)
    action,scheduled,_,_ = _service_action(raw,cfg,active)
    candidates = [(None,active,action,scheduled,False)]
    if active:
        a,s,_,_ = _service_action(raw,cfg,[])
        candidates.append((None,[],a,s,False))
    active_keys={j.key for j in active}
    occupied={tuple(j.target['tile']) for j in active if 'tile' in j.target}
    groups={}
    for j in investments:
        if j.key in active_keys or tuple(j.target['tile']) in occupied: continue
        groups.setdefault((j.kind,j.target.get('crop',j.target.get('animal'))),[]).append(j)
    for rows in groups.values():
        j=min(rows,key=lambda j:(min(distance(pos,j.target['tile']) for pos in positions(raw)),j.key))
        planned=active+[j]
        a,s,_,_=_service_action(raw,cfg,planned,{j.key})
        candidates.append((j,planned,a,s,False))
    if raw['hour']<cfg.turnsPerDay-1 and (active or operating_jobs(raw,cfg)) and len(action['market'])<cfg.maxMarketOrdersPerTurn:
        a=copy.deepcopy(action); a['market'].append(['HIRE'])
        candidates.append((None,active,a,scheduled,False))
    for idx,order in enumerate(action['market']):
        if order[0]=='SELL':
            a=copy.deepcopy(action); a['market'].pop(idx)
            candidates.append((None,active,a,scheduled,False))
    scored=[]; seen=set()
    for representative,planned,a,s,early in candidates:
        key=(repr(a),tuple(j.key for j in planned),early)
        if key in seen: continue
        seen.add(key)
        value,_=rollout(raw,cfg,planned,first_action=a,harvest_now=early)
        projected=_project_one_turn(raw,a,cfg)
        b=Bundle(a,s,projected,Envelope(value,value),float(projected['farms'][raw['player']]['money']))
        b.commitments=[j.spec() for j in planned]
        keys={j.key for j in planned}
        b.commitments.extend(j.spec() for j in operating_jobs(raw,cfg,True) if j.key in s and j.key not in keys)
        scored.append((b,representative,planned,early))
    chosen=max(scored,key=lambda row:row[0].envelope.central_cash)
    # Kaggle runtime path: stress rollouts are diagnostic only and do not
    # participate in action selection. Keep the selected central forecast
    # as both envelope values to avoid two post-selection terminal rollouts.
    return chosen[0],chosen[1],active,scored[0][0]
