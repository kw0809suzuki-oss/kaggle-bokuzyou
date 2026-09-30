#!/usr/bin/env python3
"""Search PASS actors adjacent to mature crops while expansion continues.

Observation only. This is a possible two-turn spare-labor return opportunity:
PASS -> move onto mature crop -> HARVEST, without replacing an active productive action.
"""

from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_circulation_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
DIRS=[(-1,0,"WEST"),(1,0,"EAST"),(0,-1,"NORTH"),(0,1,"SOUTH")]

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def op(a): return str(a[0]) if isinstance(a,(list,tuple)) and a else "PASS"

def positions(farm): return [plain(farm.get("farmer"))]+[plain(x) for x in (farm.get("hands",[]) or [])]

def actions(bundle,n):
    xs=[plain(bundle.get("farmer",["PASS"]))]+plain(bundle.get("hands",[]) or [])
    while len(xs)<n: xs.append(["PASS"])
    return xs[:n]

def is_expansion(a):
    o=op(a)
    if o in {"PLANT","BUILD_COOP","BUILD_PASTURE"}: return True
    return o=="PLACE" and len(a)>=2 and str(a[1]) in {"GOOSE","COW","SHEEP"}

def mature_crop(tile,day):
    if not (isinstance(tile,dict) and tile.get("kind")=="PLANT"): return False
    crop=str(tile.get("crop"))
    if crop not in CROPS or float(tile.get("yield_units",0) or 0)<=0: return False
    return int(day)-int(tile.get("planted_day",day) or day) >= int(CROPS[crop]["first_yield_day"])

def tile_at(farm,x,y):
    tiles=farm.get("tiles",[]) or []
    if 0<=y<len(tiles) and 0<=x<len(tiles[y]): return plain(tiles[y][x])
    return None

def scan(seed):
    m=load(MODEL,f"adj_m_{seed}"); o=load(OPP,f"adj_o_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False); env.reset(num_agents=2)
    found=[]
    while not env.done:
        ob=shared(env,0); oo=shared(env,1)
        a=plain(m.agent(ob,env.configuration)); ao=plain(o.agent(oo))
        farm=ob["farms"][0]; pos=positions(farm); acts=actions(a,len(pos)); day=int(ob["day"])
        expansion=[{"actor_index":i,"action":ac} for i,ac in enumerate(acts) if is_expansion(ac)]
        if expansion and not any(op(ac)=="HARVEST" for ac in acts):
            for i,(p,ac) in enumerate(zip(pos,acts)):
                if op(ac)!="PASS": continue
                x,y=int(p[0]),int(p[1])
                for dx,dy,dname in DIRS:
                    tx,ty=x+dx,y+dy
                    tile=tile_at(farm,tx,ty)
                    if mature_crop(tile,day):
                        found.append({
                            "step":int(ob["step"]),"day":day,"hour":int(ob["hour"]),
                            "actor_index":i,"actor_position":[x,y],
                            "move_action":[dname],
                            "target_position":[tx,ty],"target_tile":tile,
                            "expansion_actions":expansion,
                            "cash":float(farm.get("money",0) or 0),
                            "action_bundle":a
                        })
                        break
        env.step([a,ao])
    return {"seed":seed,"count":len(found),"first":found[0] if found else None,"examples":found[:5]}

def main():
    rows=[scan(s) for s in SEEDS]
    summary={
        "seed_count":sum(r["count"]>0 for r in rows),
        "total":sum(r["count"] for r in rows),
        "seeds":[r["seed"] for r in rows if r["count"]>0],
    }
    out={"schema":"adaptive-circulation-adjacent-pass-search-v0","summary":summary,"cases":rows,
         "boundary":"Observation only; a two-turn route is not yet a candidate policy."}
    Path("adaptive_circulation_adjacent_pass_search_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    for r in rows:
        print("CASE "+json.dumps({"seed":r["seed"],"count":r["count"],"first":r["first"]},ensure_ascii=False,separators=(",",":")))

if __name__=="__main__": main()
