#!/usr/bin/env python3
"""Observe whole-farm return queue at step250.

Predeclared economic surfaces:
- current market-valued carry across all workers
- current market-valued shed stock
- WHEAT share of ready-to-sell value
- non-WHEAT ready value
- shed occupancy / free capacity
- selected actor's carried value

No fitted classifier. No intervention.
"""

from __future__ import annotations
import importlib.util, json, math, sys
from pathlib import Path
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture.kaggriculture import PRODUCTS

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_circulation_runtime_v0.py"
OPPONENT=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
D={92802001:-7657,92802002:-104,92802003:-9578,92802004:-50,92802005:-794,92802006:895,92802007:190,92802008:-53,92802009:20312,92802010:21568}

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"):return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):return [plain(v) for v in x]
    return x

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m)
    if hasattr(m,"reset_agent"):m.reset_agent()
    return m

def shared(env,seat): return plain(env._Environment__get_shared_state(seat)["observation"])

def value(inv,prices):
    return sum(float(n or 0)*float(prices.get(item,0) or 0) for item,n in (inv or {}).items() if item in PRODUCTS)

def units(inv):
    return sum(float(n or 0) for item,n in (inv or {}).items() if item in PRODUCTS)

def capture(seed):
    model=load(MODEL,f"rq_m_{seed}"); opp=load(OPPONENT,f"rq_o_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    while not env.done:
        o=shared(env,0)
        if int(o.get("step",0) or 0)==250:
            p=o.get("private") or {}; prices=(o.get("market") or {}).get("prices") or {}
            shed=p.get("shed") or {}; invs=p.get("inventories") or []
            carry_val=sum(value(inv,prices) for inv in invs if isinstance(inv,dict))
            carry_units=sum(units(inv) for inv in invs if isinstance(inv,dict))
            shed_val=value(shed,prices); shed_units=units(shed)
            wheat_price=float(prices.get("WHEAT",0) or 0)
            wheat_carry=sum(float((inv or {}).get("WHEAT",0) or 0) for inv in invs if isinstance(inv,dict))
            wheat_shed=float(shed.get("WHEAT",0) or 0)
            wheat_ready_val=(wheat_carry+wheat_shed)*wheat_price
            total_ready_val=carry_val+shed_val
            non_wheat_ready=total_ready_val-wheat_ready_val
            actor3_inv=invs[3] if len(invs)>3 and isinstance(invs[3],dict) else {}
            actor3_val=value(actor3_inv,prices)
            f={
              "carry_all_value":carry_val,
              "shed_all_value":shed_val,
              "ready_all_value":total_ready_val,
              "non_wheat_ready_value":non_wheat_ready,
              "wheat_ready_value":wheat_ready_val,
              "wheat_ready_share":(wheat_ready_val/total_ready_val) if total_ready_val>0 else 0,
              "carry_all_units":carry_units,
              "shed_all_units":shed_units,
              "ready_all_units":carry_units+shed_units,
              "shed_free_capacity":100.0-shed_units,
              "actor3_carried_value":actor3_val,
              "actor3_carried_units":units(actor3_inv),
              "actor3_wheat_units":float(actor3_inv.get("WHEAT",0) or 0),
              "wheat_price":wheat_price,
            }
            return f
        o1=shared(env,1);env.step([plain(model.agent(o,env.configuration)),plain(opp.agent(o1))])
    raise RuntimeError(seed)

def perfect(rows,key):
    vals=[]
    for r in rows:
        v=r["features"].get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):return []
        vals.append((float(v),r["label"],r["seed"]))
    uniq=sorted(set(v for v,_,_ in vals));ans=[]
    for a,b in zip(uniq,uniq[1:]):
        cut=(a+b)/2
        for orient in ("GT_HARVEST","LE_HARVEST"):
            ok=True
            for v,lab,_ in vals:
                pred="HARVEST_GT_PASS" if ((v>cut) if orient=="GT_HARVEST" else (v<=cut)) else "PASS_GT_HARVEST"
                if pred!=lab:ok=False;break
            if ok:ans.append({"feature":key,"threshold":cut,"orientation":orient})
    return ans

def main():
    rows=[]
    for seed in SEEDS:
        f=capture(seed); lab="HARVEST_GT_PASS" if D[seed]>0 else "PASS_GT_HARVEST"
        row={"seed":seed,"label":lab,"delta_harvest_minus_pass":D[seed],"features":f};rows.append(row)
        print("CASE "+json.dumps(row,separators=(",",":")))
    keys=sorted(rows[0]["features"]);pos=[r for r in rows if r["label"]=="HARVEST_GT_PASS"];neg=[r for r in rows if r["label"]=="PASS_GT_HARVEST"]
    dis=[];ths=[]
    for k in keys:
        pv={r["features"][k] for r in pos};nv={r["features"][k] for r in neg}
        if pv.isdisjoint(nv):dis.append({"feature":k,"harvest_values":[r["features"][k] for r in pos],"pass_values":[r["features"][k] for r in neg]})
        ths.extend(perfect(rows,k))
    s={"n":10,"features_checked":keys,"disjoint_feature_count":len(dis),"disjoint_features":dis,"perfect_threshold_count":len(ths),"perfect_thresholds":ths}
    print("SUMMARY "+json.dumps(s,separators=(",",":")))
    Path("adaptive_circulation_return_queue_probe_v0_result.json").write_text(json.dumps({"schema":"adaptive-circulation-return-queue-probe-v0","summary":s,"cases":rows,
    "boundary":["No compound classifier.","Fixed10 separation is observational only.","No rescue if simple return-queue surfaces fail."]},indent=2)+"\n",encoding="utf-8")
if __name__=="__main__":main()
