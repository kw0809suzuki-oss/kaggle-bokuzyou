#!/usr/bin/env python3
import argparse, copy, hashlib, json, subprocess, sys
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_kaggriculture

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"selfsrc"))
sys.path.insert(0,str(ROOT/"opponents"))

import whole_flow_control_agent as whole_flow
import seyamalam_v21 as opponent

SEEDS=[7001,7002,7003,7005,7006,7008,7009]
EXPANSION_OPS={"BUY_LAND","BUY_SEED","BUY_ANIMAL"}

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    return repr(x)

def stable_hash(obj):
    return hashlib.sha256(json.dumps(plain(obj),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def is_expansion(order):
    if not isinstance(order,(list,tuple)) or not order: return False
    if order[0] in EXPANSION_OPS: return True
    return order[0]=="BUY_PRODUCT" and len(order)>1 and order[1]=="COW"

def p1_filter(obs,native_action):
    if int(obs.get("day",0) or 0)<12 or not isinstance(native_action,dict):
        return copy.deepcopy(native_action),[]
    market=list(native_action.get("market",[]) or [])
    removed=[copy.deepcopy(o) for o in market if is_expansion(o)]
    revised=copy.deepcopy(native_action)
    revised["market"]=[copy.deepcopy(o) for o in market if not is_expansion(o)]
    return revised,removed

def reference_agent(arm):
    def agent(obs):
        native=whole_flow.agent(obs)
        if arm=="p1":
            submitted,_=p1_filter(obs,native)
            return submitted
        return native
    return agent

def run_reference(arm,seed):
    whole_flow.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.run([reference_agent(arm),opponent.agent])
    print(json.dumps({
        "kind":"reference","arm":arm,"seed":seed,
        "self":float(env.state[0].reward),
        "opponent":float(env.state[1].reward),
        "margin":float(env.state[0].reward)-float(env.state[1].reward),
        "steps":len(env.steps)-1,
    },separators=(",",":")))

def summarize_farm(farm,private,current_day):
    plants={}; placed_animals={}; structures={}; empty=0; weeds=0
    plant_details=[]
    for y,row in enumerate(farm.get("tiles",[])):
        for x,tile in enumerate(row):
            if tile is None:
                empty+=1; continue
            if tile=="LOCKED" or not isinstance(tile,dict): continue
            kind=tile.get("kind")
            if kind=="PLANT":
                crop=tile.get("crop")
                plants[crop]=plants.get(crop,0)+1
                cd=official_kaggriculture.CROPS.get(crop,{})
                planted_day=int(tile.get("planted_day",current_day) or 0)
                age_days=current_day-planted_day
                first_yield_day=int(cd.get("first_yield_day",0) or 0)
                max_yield_day=int(cd.get("max_yield_day",0) or 0)
                plant_details.append({
                    "x":x,"y":y,
                    "crop":crop,
                    "planted_day":planted_day,
                    "age_days":age_days,
                    "first_yield_day":first_yield_day,
                    "days_to_first_yield":max(0,first_yield_day-age_days),
                    "max_yield_day":max_yield_day,
                    "days_to_max_yield":max(0,max_yield_day-age_days),
                    "mature_for_first_yield":age_days>=first_yield_day,
                    "yield_units":int(tile.get("yield_units",0) or 0),
                    "watered_today":bool(tile.get("watered_today",False)),
                    "consecutive_unwatered":int(tile.get("consecutive_unwatered",0) or 0),
                    "fertilized_until_day":int(tile.get("fertilized_until_day",-1) or -1),
                    "max_lifespan_step":int(tile.get("max_lifespan_step",-1) or -1),
                })
            elif kind=="WEED":
                weeds+=1
            if "animal" in tile:
                a=tile.get("animal"); placed_animals[a]=placed_animals.get(a,0)+1
            if kind in ("COOP","PASTURE"):
                structures[kind]=structures.get(kind,0)+1
    shed=private.get("shed",{}) or {}
    return {
        "cash":float(farm.get("money",0) or 0),
        "land":list(farm.get("unlocked_quadrants",[]) or []),
        "land_count":len(farm.get("unlocked_quadrants",[]) or []),
        "hands":plain(farm.get("hands",[]) or []),
        "hands_count":len(farm.get("hands",[]) or []),
        "farmer":plain(farm.get("farmer")),
        "seeds":{k:int(v or 0) for k,v in (private.get("seeds",{}) or {}).items()},
        "shed":{k:int(v or 0) for k,v in shed.items() if int(v or 0)!=0},
        "inventories":plain(private.get("inventories",[]) or []),
        "plants":plants,
        "plant_details":plant_details,
        "placed_animals":placed_animals,
        "shed_animals":{k:int(shed.get(k,0) or 0) for k in ("GOOSE","COW","SHEEP")},
        "structures":structures,
        "empty_tiles":empty,
        "weeds":weeds,
    }

def snapshot(env):
    s0=env.state[0].observation
    s1=env.state[1].observation
    farms=s0.farms
    core={
        "day":int(s0.day),"hour":int(s0.hour),
        "self":summarize_farm(farms[0],s0.private,int(s0.day)),
        "opponent":summarize_farm(farms[1],s1.private,int(s0.day)),
        "market":plain(s0.market),
        "town":plain(s0.town),
    }
    full={
        "day":int(s0.day),"hour":int(s0.hour),
        "farms":plain(farms),"private0":plain(s0.private),"private1":plain(s1.private),
        "market":plain(s0.market),"town":plain(s0.town),
    }
    core["world_hash"]=stable_hash(full)
    return core

def install_world_hooks(env,exec_log,step_box):
    g=env.interpreter.__globals__
    orig_commit=g["_commit_unit"]
    orig_land=g["_do_buy_land"]
    orig_hire=g["_do_hire"]
    farm_ids={id(env.state[0].observation.farms[0]):0,id(env.state[0].observation.farms[1]):1}

    def player_for(farm): return farm_ids.get(id(farm),-1)

    def observed_commit(op,item,price,farm,private,market,shed_capacity=100):
        before={"money":float(farm.get("money",0) or 0),
                "seed":int((private.get("seeds",{}) or {}).get(item,0) or 0),
                "shed":int((private.get("shed",{}) or {}).get(item,0) or 0),
                "market_inventory":int((market.get("inventory",{}) or {}).get(item,0) or 0)}
        ok=orig_commit(op,item,price,farm,private,market,shed_capacity)
        after={"money":float(farm.get("money",0) or 0),
               "seed":int((private.get("seeds",{}) or {}).get(item,0) or 0),
               "shed":int((private.get("shed",{}) or {}).get(item,0) or 0),
               "market_inventory":int((market.get("inventory",{}) or {}).get(item,0) or 0)}
        exec_log.append({"step":step_box["step"],"player":player_for(farm),"kind":"market_unit",
                         "op":op,"item":item,"quoted_price":float(price),"success":bool(ok),
                         "before":before,"after":after})
        return ok

    def observed_land(farm,board_size):
        before={"money":float(farm.get("money",0) or 0),"land":list(farm.get("unlocked_quadrants",[]) or [])}
        out=orig_land(farm,board_size)
        after={"money":float(farm.get("money",0) or 0),"land":list(farm.get("unlocked_quadrants",[]) or [])}
        exec_log.append({"step":step_box["step"],"player":player_for(farm),"kind":"atomic_market",
                         "op":"BUY_LAND","success":before!=after,"before":before,"after":after})
        return out

    def observed_hire(farm,private,board_size,mult=1):
        before={"money":float(farm.get("money",0) or 0),"hands":plain(farm.get("hands",[]) or [])}
        out=orig_hire(farm,private,board_size,mult)
        after={"money":float(farm.get("money",0) or 0),"hands":plain(farm.get("hands",[]) or [])}
        exec_log.append({"step":step_box["step"],"player":player_for(farm),"kind":"atomic_market",
                         "op":"HIRE","success":before!=after,"before":before,"after":after})
        return out

    g["_commit_unit"]=observed_commit
    g["_do_buy_land"]=observed_land
    g["_do_hire"]=observed_hire
    return g,orig_commit,orig_land,orig_hire

def restore_world_hooks(h):
    g,a,b,c=h
    g["_commit_unit"]=a; g["_do_buy_land"]=b; g["_do_hire"]=c

def run_trace(arm,seed):
    whole_flow.reset_telemetry()
    env=make("kaggriculture",configuration={"seed":seed},debug=False)
    env.reset(num_agents=2)
    exec_log=[]; step_box={"step":None}
    hooks=install_world_hooks(env,exec_log,step_box)
    trace=[]; step=0
    try:
        while not env.done:
            pre=snapshot(env)

            # Exactly mirror Environment.run agent input surface:
            # each agent receives its own copied/shared observation view.
            state0=env._Environment__get_shared_state(0)
            state1=env._Environment__get_shared_state(1)
            obs_self=state0["observation"]
            obs_opp=state1["observation"]

            native_self=plain(whole_flow.agent(obs_self))
            if arm=="p1":
                submitted_self,removed=p1_filter(obs_self,native_self)
            else:
                submitted_self=copy.deepcopy(native_self)
                _,removed=p1_filter(obs_self,native_self)  # hypothetical P1 removal at same state
            submitted_opp=plain(opponent.agent(obs_opp))

            exec_log.clear()
            step_box["step"]=step
            env.step([submitted_self,submitted_opp])
            post=snapshot(env)

            trace.append({
                "step":step,"day":pre["day"],"hour":pre["hour"],
                "pre":pre,
                "native_self_action":native_self,
                "submitted_self_action":plain(submitted_self),
                "removed_by_p1_rule":plain(removed),
                "opponent_action":submitted_opp,
                "official_world_execution":[copy.deepcopy(e) for e in exec_log],
                "post":post,
            })
            step+=1
    finally:
        restore_world_hooks(hooks)

    print(json.dumps({
        "kind":"trace","arm":arm,"seed":seed,
        "self":float(env.state[0].reward),
        "opponent":float(env.state[1].reward),
        "margin":float(env.state[0].reward)-float(env.state[1].reward),
        "steps":len(trace),
        "trace":trace,
    },ensure_ascii=False,separators=(",",":")))

def child(mode,arm,seed):
    cp=subprocess.run([sys.executable,__file__,"--mode",mode,"--arm",arm,"--seed",str(seed)],
                      check=True,capture_output=True,text=True)
    return json.loads(cp.stdout.strip().splitlines()[-1])

def metric_delta(p1,base):
    out={"cash":p1["cash"]-base["cash"],
         "land_count":p1["land_count"]-base["land_count"],
         "hands_count":p1["hands_count"]-base["hands_count"],
         "empty_tiles":p1["empty_tiles"]-base["empty_tiles"],
         "weeds":p1["weeds"]-base["weeds"]}
    for group in ("seeds","plants","placed_animals","shed_animals","structures"):
        keys=sorted(set(base.get(group,{}))|set(p1.get(group,{})))
        out[group]={k:p1.get(group,{}).get(k,0)-base.get(group,{}).get(k,0) for k in keys}
    return out

def action_key(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)

def self_exec(events):
    return [e for e in events if e.get("player")==0]

def summarize_removed_execution(removed,events):
    result=[]
    for order in removed:
        if not order: continue
        op=order[0]
        item=None
        qty=1
        if op in ("BUY_SEED","BUY_PRODUCT","BUY_ANIMAL") and len(order)>=3:
            item=order[1]; qty=int(order[2])
        matches=[]
        for e in events:
            if e.get("player")!=0: continue
            if e.get("op")!=op: continue
            if item is not None and e.get("item")!=item: continue
            matches.append(e)
        result.append({
            "order":order,
            "requested_qty":qty,
            "execution_events":matches,
            "successful_units":sum(1 for e in matches if e.get("success")),
            "failed_units":sum(1 for e in matches if not e.get("success")),
        })
    return result


def wheat_plants(state_self):
    return [p for p in state_self.get("plant_details",[]) if p.get("crop")=="WHEAT"]

def wheat_plant_keys(state_self):
    return {(p.get("x"),p.get("y"),p.get("planted_day")) for p in wheat_plants(state_self)}

def wheat_plant_actions(action):
    if not isinstance(action,dict):
        return []
    out=[]
    farmer=action.get("farmer")
    if isinstance(farmer,list) and len(farmer)>=2 and farmer[0]=="PLANT" and farmer[1]=="WHEAT":
        out.append({"unit":"farmer","action":farmer})
    for i,a in enumerate(action.get("hands",[]) or []):
        if isinstance(a,list) and len(a)>=2 and a[0]=="PLANT" and a[1]=="WHEAT":
            out.append({"unit":"hand_"+str(i+1),"action":a})
    return out

def build_wheat_conversion_chain(base,p1):
    bt,pt=base["trace"],p1["trace"]
    n=min(len(bt),len(pt))
    first_action=next((i for i in range(n)
                       if action_key(bt[i]["submitted_self_action"])!=action_key(pt[i]["submitted_self_action"])),None)
    if first_action is None:
        return None

    # This probe is only for cases whose first removed action is BUY_SEED WHEAT.
    removed=pt[first_action].get("removed_by_p1_rule",[])
    if not any(isinstance(o,list) and len(o)>=2 and o[0]=="BUY_SEED" and o[1]=="WHEAT" for o in removed):
        return None

    chain=[]
    first_plant_action_divergence=None
    first_productive_divergence=None
    first_seed_delta_change=None

    initial_seed_delta=(pt[first_action]["post"]["self"]["seeds"].get("WHEAT",0)
                        - bt[first_action]["post"]["self"]["seeds"].get("WHEAT",0))

    for i in range(first_action, n):
        b,p=bt[i],pt[i]
        bpre=b["pre"]["self"]; ppre=p["pre"]["self"]
        bpost=b["post"]["self"]; ppost=p["post"]["self"]

        bpre_keys=wheat_plant_keys(bpre); ppre_keys=wheat_plant_keys(ppre)
        bpost_keys=wheat_plant_keys(bpost); ppost_keys=wheat_plant_keys(ppost)

        b_added=sorted(list(bpost_keys-bpre_keys))
        p_added=sorted(list(ppost_keys-ppre_keys))

        row={
            "step":i,"day":b["day"],"hour":b["hour"],
            "baseline_pre_wheat_seed":bpre["seeds"].get("WHEAT",0),
            "p1_pre_wheat_seed":ppre["seeds"].get("WHEAT",0),
            "pre_seed_delta_p1_minus_baseline":ppre["seeds"].get("WHEAT",0)-bpre["seeds"].get("WHEAT",0),
            "baseline_native_wheat_plant_actions":wheat_plant_actions(b["native_self_action"]),
            "p1_native_wheat_plant_actions":wheat_plant_actions(p["native_self_action"]),
            "baseline_submitted_wheat_plant_actions":wheat_plant_actions(b["submitted_self_action"]),
            "p1_submitted_wheat_plant_actions":wheat_plant_actions(p["submitted_self_action"]),
            "baseline_post_wheat_seed":bpost["seeds"].get("WHEAT",0),
            "p1_post_wheat_seed":ppost["seeds"].get("WHEAT",0),
            "post_seed_delta_p1_minus_baseline":ppost["seeds"].get("WHEAT",0)-bpost["seeds"].get("WHEAT",0),
            "baseline_pre_wheat_plants":len(bpre_keys),
            "p1_pre_wheat_plants":len(ppre_keys),
            "baseline_post_wheat_plants":len(bpost_keys),
            "p1_post_wheat_plants":len(ppost_keys),
            "post_wheat_plant_delta_p1_minus_baseline":len(ppost_keys)-len(bpost_keys),
            "baseline_new_wheat_plants":b_added,
            "p1_new_wheat_plants":p_added,
            "baseline_full_action":b["submitted_self_action"],
            "p1_full_action":p["submitted_self_action"],
            "baseline_post_state":b["post"],
            "p1_post_state":p["post"],
        }

        if first_seed_delta_change is None and row["post_seed_delta_p1_minus_baseline"] != initial_seed_delta:
            first_seed_delta_change=i
        if first_plant_action_divergence is None and (
            action_key(row["baseline_submitted_wheat_plant_actions"])
            != action_key(row["p1_submitted_wheat_plant_actions"])
        ):
            first_plant_action_divergence=i
        if first_productive_divergence is None and len(bpost_keys)!=len(ppost_keys):
            first_productive_divergence=i

        # Preserve the whole path until productive WHEAT first diverges.
        chain.append(row)
        if first_productive_divergence is not None:
            break

    return {
        "seed":base["seed"],
        "first_buy_seed_divergence_step":first_action,
        "initial_post_seed_delta_p1_minus_baseline":initial_seed_delta,
        "first_seed_delta_change_step":first_seed_delta_change,
        "first_wheat_plant_action_divergence_step":first_plant_action_divergence,
        "first_productive_wheat_divergence_step":first_productive_divergence,
        "baseline_extra_seed_reached_productive_difference":first_productive_divergence is not None,
        "chain":chain,
    }

def build_packet(base,p1,validation):
    bt,pt=base["trace"],p1["trace"]
    n=min(len(bt),len(pt))
    first_action=next((i for i in range(n) if action_key(bt[i]["submitted_self_action"])!=action_key(pt[i]["submitted_self_action"])),None)
    first_world=next((i for i in range(first_action or 0,n) if bt[i]["post"]["world_hash"]!=pt[i]["post"]["world_hash"]),None)
    first_follow=None
    if first_world is not None:
        first_follow=next((i for i in range(first_world+1,n) if action_key(bt[i]["native_self_action"])!=action_key(pt[i]["native_self_action"])),None)

    packet={
        "seed":base["seed"],
        "validation":validation,
        "terminal":{
            "baseline_self":base["self"],"p1_self":p1["self"],"delta_self":p1["self"]-base["self"],
            "baseline_opponent":base["opponent"],"p1_opponent":p1["opponent"],
            "baseline_margin":base["margin"],"p1_margin":p1["margin"],
        },
        "first_action_divergence":None,
        "first_world_divergence":None,
        "first_downstream_current_divergence":None,
        "followup_3_turns_after_world_divergence":[],
    }

    if first_action is not None:
        b,p=bt[first_action],pt[first_action]
        packet["first_action_divergence"]={
            "step":first_action,"day":b["day"],"hour":b["hour"],
            "pre_state_equal":b["pre"]["world_hash"]==p["pre"]["world_hash"],
            "baseline_pre_state":b["pre"],"p1_pre_state":p["pre"],
            "baseline_native_action":b["native_self_action"],"baseline_submitted_action":b["submitted_self_action"],
            "p1_native_action":p["native_self_action"],"p1_submitted_action":p["submitted_self_action"],
            "p1_removed_action":p["removed_by_p1_rule"],
            "baseline_official_world_execution":self_exec(b["official_world_execution"]),
            "p1_official_world_execution":self_exec(p["official_world_execution"]),
            "baseline_removed_action_execution":summarize_removed_execution(p["removed_by_p1_rule"],b["official_world_execution"]),
            "baseline_post_state":b["post"],"p1_post_state":p["post"],
            "immediate_self_state_delta_p1_minus_baseline":metric_delta(p["post"]["self"],b["post"]["self"]),
        }

    if first_world is not None:
        b,p=bt[first_world],pt[first_world]
        packet["first_world_divergence"]={
            "step":first_world,"day":b["day"],"hour":b["hour"],
            "pre_state_equal":b["pre"]["world_hash"]==p["pre"]["world_hash"],
            "baseline_action":b["submitted_self_action"],"p1_action":p["submitted_self_action"],
            "baseline_official_world_execution":self_exec(b["official_world_execution"]),
            "p1_official_world_execution":self_exec(p["official_world_execution"]),
            "baseline_post_state":b["post"],"p1_post_state":p["post"],
            "self_state_delta_p1_minus_baseline":metric_delta(p["post"]["self"],b["post"]["self"]),
        }
        for j in range(first_world+1,min(first_world+4,n)):
            packet["followup_3_turns_after_world_divergence"].append({
                "step":j,
                "baseline_pre_state":bt[j]["pre"],"p1_pre_state":pt[j]["pre"],
                "baseline_native_action":bt[j]["native_self_action"],"p1_native_action":pt[j]["native_self_action"],
                "baseline_submitted_action":bt[j]["submitted_self_action"],"p1_submitted_action":pt[j]["submitted_self_action"],
                "baseline_official_world_execution":self_exec(bt[j]["official_world_execution"]),
                "p1_official_world_execution":self_exec(pt[j]["official_world_execution"]),
                "baseline_post_state":bt[j]["post"],"p1_post_state":pt[j]["post"],
            })

    if first_follow is not None:
        b,p=bt[first_follow],pt[first_follow]
        packet["first_downstream_current_divergence"]={
            "step":first_follow,"day":b["day"],"hour":b["hour"],
            "baseline_pre_state":b["pre"],"p1_pre_state":p["pre"],
            "pre_state_delta_p1_minus_baseline":metric_delta(p["pre"]["self"],b["pre"]["self"]),
            "baseline_native_action":b["native_self_action"],"p1_native_action":p["native_self_action"],
            "baseline_submitted_action":b["submitted_self_action"],"p1_submitted_action":p["submitted_self_action"],
            "baseline_official_world_execution":self_exec(b["official_world_execution"]),
            "p1_official_world_execution":self_exec(p["official_world_execution"]),
            "baseline_post_state":b["post"],"p1_post_state":p["post"],
        }

    fa=packet["first_action_divergence"]
    world_capture_ok=bool(fa and fa["baseline_removed_action_execution"] and
                          all("successful_units" in x for x in fa["baseline_removed_action_execution"]))
    packet["observation_checks"]={
        "action_difference_observed":fa is not None,
        "pre_and_post_state_observed":bool(fa and fa["baseline_pre_state"] and fa["baseline_post_state"] and fa["p1_post_state"]),
        "official_world_result_observed":world_capture_ok,
        "baseline_and_p1_symmetric":bool(fa and fa["baseline_pre_state"] and fa["p1_pre_state"] and
                                         fa["baseline_submitted_action"] is not None and fa["p1_submitted_action"] is not None),
        "current_followup_connected":packet["first_downstream_current_divergence"] is not None,
        "instrumented_replay_matches_reference":validation["all_match"],
    }
    packet["observation_complete"]=all(packet["observation_checks"].values())
    return packet

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["reference","trace"])
    ap.add_argument("--arm",choices=["baseline","p1"])
    ap.add_argument("--seed",type=int)
    args=ap.parse_args()

    if args.mode=="reference":
        run_reference(args.arm,args.seed); return
    if args.mode=="trace":
        run_trace(args.arm,args.seed); return

    packets=[]
    conversion_chains=[]
    for seed in SEEDS:
        rb=child("reference","baseline",seed)
        rp=child("reference","p1",seed)
        tb=child("trace","baseline",seed)
        tp=child("trace","p1",seed)
        validation={
            "baseline_self_match":tb["self"]==rb["self"],
            "baseline_opponent_match":tb["opponent"]==rb["opponent"],
            "p1_self_match":tp["self"]==rp["self"],
            "p1_opponent_match":tp["opponent"]==rp["opponent"],
        }
        validation["all_match"]=all(validation.values())
        packet=build_packet(tb,tp,validation)
        packets.append(packet)
        conversion=build_wheat_conversion_chain(tb,tp)
        if conversion is not None:
            conversion["terminal_delta_self"]=tp["self"]-tb["self"]
            conversion["reference_match"]=validation["all_match"]
            conversion_chains.append(conversion)
            final_row=conversion["chain"][-1] if conversion["chain"] else None
            print("CONVERSION "+json.dumps({
                "seed":seed,
                "delta_self":conversion["terminal_delta_self"],
                "first_buy_seed_step":conversion["first_buy_seed_divergence_step"],
                "first_seed_delta_change_step":conversion["first_seed_delta_change_step"],
                "first_wheat_plant_action_divergence_step":conversion["first_wheat_plant_action_divergence_step"],
                "first_productive_wheat_divergence_step":conversion["first_productive_wheat_divergence_step"],
                "final_post_seed_delta":None if final_row is None else final_row["post_seed_delta_p1_minus_baseline"],
                "final_post_wheat_plant_delta":None if final_row is None else final_row["post_wheat_plant_delta_p1_minus_baseline"],
                "reference_match":validation["all_match"],
            },separators=(",",":")))
        fa=packet["first_action_divergence"]
        fills=None
        if fa:
            fills=[{"order":x["order"],"successful_units":x["successful_units"],"failed_units":x["failed_units"]}
                   for x in fa["baseline_removed_action_execution"]]
        print("PACKET "+json.dumps({
            "seed":seed,
            "reference_match":validation["all_match"],
            "first_action_step":None if fa is None else fa["step"],
            "first_world_step":None if packet["first_world_divergence"] is None else packet["first_world_divergence"]["step"],
            "first_followup_step":None if packet["first_downstream_current_divergence"] is None else packet["first_downstream_current_divergence"]["step"],
            "baseline_removed_execution":fills,
            "delta_self":packet["terminal"]["delta_self"],
            "complete":packet["observation_complete"],
        },separators=(",",":")))
        if fa and fa["p1_removed_action"] and fa["p1_removed_action"][0][:2]==["BUY_SEED","WHEAT"]:
            pre=fa["baseline_pre_state"]["self"]
            wheat=[p for p in pre.get("plant_details",[]) if p.get("crop")=="WHEAT"]
            strawberry=[p for p in pre.get("plant_details",[]) if p.get("crop")=="STRAWBERRY"]
            follow=packet["first_downstream_current_divergence"]
            print("WHEAT_CASE "+json.dumps({
                "seed":seed,
                "day":fa["day"],"hour":fa["hour"],
                "delta_self":packet["terminal"]["delta_self"],
                "pre_cash":pre["cash"],
                "pre_wheat_seed":pre["seeds"].get("WHEAT",0),
                "pre_wheat_plants":len(wheat),
                "pre_strawberry_plants":len(strawberry),
                "wheat_plant_details":wheat,
                "strawberry_plant_details":strawberry,
                "immediate_post_delta":fa["immediate_self_state_delta_p1_minus_baseline"],
                "baseline_next_native_action":None if follow is None else follow["baseline_native_action"],
                "p1_next_native_action":None if follow is None else follow["p1_native_action"],
            },ensure_ascii=False,separators=(",",":")))

    out={
        "schema":"p1-wheat-seed-to-productive-v0",
        "purpose":"follow the seven WHEAT suppression cases from seed difference through PLANT action to productive WHEAT state",
        "baseline":"whole_flow_control_agent",
        "p1":"whole_flow + D12 expansion closure",
        "world":"Official Kaggriculture; hooks installed into the exact interpreter globals used by env.step",
        "opponent":"Seyamalam pinned 8b8c421e...",
        "seeds":SEEDS,
        "guard":"instrumented manual replay must exactly match an independent env.run reference for self and opponent terminal reward",
        "observation_completion_conditions":[
            "Action difference observed",
            "Pre-State and Post-State observed",
            "Official World execution result observed",
            "Baseline and P1 observed symmetrically",
            "Current follow-up from the next State connected",
        ],
        "packets":packets,
        "wheat_conversion_chains":conversion_chains,
    }
    Path("p1_wheat_seed_to_productive_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps({
        "packets":len(packets),
        "reference_matched":sum(p["validation"]["all_match"] for p in packets),
        "complete":sum(p["observation_complete"] for p in packets),
        "improved":sum(p["terminal"]["delta_self"]>0 for p in packets),
        "worsened":sum(p["terminal"]["delta_self"]<0 for p in packets),
        "conversion_chains":len(conversion_chains),
        "productive_divergence_found":sum(x["first_productive_wheat_divergence_step"] is not None for x in conversion_chains),
    },separators=(",",":")))

if __name__=="__main__":
    main()
