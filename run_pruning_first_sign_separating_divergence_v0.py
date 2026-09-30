#!/usr/bin/env python3
import copy, importlib.util, json, os, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"adaptive_replay_contract_runtime_v0.py"
CAND=ROOT/"adaptive_replay_pruning_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"

SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]
SIGN={
  92802001:"improved",92802002:"improved",92802003:"improved",92802004:"improved",92802005:"worsened",
  92802006:"improved",92802007:"worsened",92802008:"worsened",92802009:"improved",92802010:"worsened",
}
TARGET={"step":0,"market_index":2,"order":["BUY_PRODUCT","WHEAT",5]}
ITEMS=["WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER"]
ANIMALS=["COW","SHEEP","CHICKEN"]

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)): return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"): mod.reset_agent()
    return mod

def shared(env,seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])

def action_sig(a):
    return json.dumps(a,sort_keys=True,separators=(",",":"))

def get_num(d,*path,default=0):
    cur=d
    try:
        for p in path: cur=cur[p]
        return float(cur or 0)
    except Exception:
        return float(default)

def count_kind(obj, kind):
    if isinstance(obj,dict):
        c=1 if str(obj.get("kind",""))==kind else 0
        return c+sum(count_kind(v,kind) for v in obj.values())
    if isinstance(obj,list):
        return sum(count_kind(v,kind) for v in obj)
    return 0

def count_inventory_mentions(obj, item):
    # diagnostic count across visible farm structure only; not interpreted as canonical inventory
    if isinstance(obj,dict):
        total=0
        for k,v in obj.items():
            if str(k)==item and isinstance(v,(int,float)):
                total+=float(v)
            else:
                total+=count_inventory_mentions(v,item)
        return total
    if isinstance(obj,list):
        return sum(count_inventory_mentions(v,item) for v in obj)
    return 0

def feature_delta(ob, oc, ab, ac, aob, aoc):
    fb=ob["farms"][0]; fc=oc["farms"][0]
    obf=ob["farms"][1]; ocf=oc["farms"][1]
    out={
      "self_money": get_num(fc,"money")-get_num(fb,"money"),
      "opp_money": get_num(ocf,"money")-get_num(obf,"money"),
      "self_hands": len(fc.get("hands",[]) or [])-len(fb.get("hands",[]) or []),
      "opp_hands": len(ocf.get("hands",[]) or [])-len(obf.get("hands",[]) or []),
      "self_hires_today": get_num(fc,"hires_today")-get_num(fb,"hires_today"),
      "opp_hires_today": get_num(ocf,"hires_today")-get_num(obf,"hires_today"),
      "self_action_equal": action_sig(ab)==action_sig(ac),
      "opp_action_equal": action_sig(aob)==action_sig(aoc),
    }
    mb=ob.get("market",{}) or {}; mc=oc.get("market",{}) or {}
    for item in ITEMS:
        out[f"market_inventory_{item}"]=get_num(mc,"inventory",item)-get_num(mb,"inventory",item)
        out[f"market_price_{item}"]=get_num(mc,"prices",item)-get_num(mb,"prices",item)
        out[f"self_visible_{item}"]=count_inventory_mentions(fc,item)-count_inventory_mentions(fb,item)
    tb=ob.get("town",{}) or {}; tc=oc.get("town",{}) or {}
    out["unlocked_shop_count"]=len(tc.get("unlocked_shops",[]) or [])-len(tb.get("unlocked_shops",[]) or [])
    for k in ["PLANT","ANIMAL","PASTURE","WEED"]:
        out[f"self_kind_{k}"]=count_kind(fc,k)-count_kind(fb,k)
    return out

def init_pair(seed):
    eb=make("kaggriculture",configuration={"seed":seed},debug=False); eb.reset(2)
    ec=make("kaggriculture",configuration={"seed":seed},debug=False); ec.reset(2)
    b=load(BASE,f"b_{seed}_{os.getpid()}")
    c=load(CAND,f"c_{seed}_{os.getpid()}"); c.configure(TARGET["step"],TARGET["market_index"],TARGET["order"]); c.reset_agent()
    ob=load(OPP,f"ob_{seed}_{os.getpid()}")
    oc=load(OPP,f"oc_{seed}_{os.getpid()}")
    return eb,ec,b,c,ob,oc

def run(seed):
    eb,ec,b,c,obm,ocm=init_pair(seed)
    rows=[]
    while not eb.done and not ec.done:
        sob=shared(eb,0); soc=shared(ec,0)
        ab=plain(b.agent(copy.deepcopy(sob))); ac=plain(c.agent(copy.deepcopy(soc)))
        aob=plain(obm.agent(shared(eb,1))); aoc=plain(ocm.agent(shared(ec,1)))
        step=int(sob["step"])
        feat=feature_delta(sob,soc,ab,ac,aob,aoc)
        rows.append({"step":step,"features":feat})
        eb.step([ab,aob]); ec.step([ac,aoc])
    return rows

def group_uniform_separator(per_seed):
    improved=[s for s in SEEDS if SIGN[s]=="improved"]
    worsened=[s for s in SEEDS if SIGN[s]=="worsened"]
    max_steps=min(len(per_seed[s]) for s in SEEDS)
    hits=[]
    for t in range(max_steps):
        keys=sorted(per_seed[SEEDS[0]][t]["features"].keys())
        for key in keys:
            iv=[per_seed[s][t]["features"][key] for s in improved]
            wv=[per_seed[s][t]["features"][key] for s in worsened]
            if len({json.dumps(v,sort_keys=True) for v in iv})==1 and len({json.dumps(v,sort_keys=True) for v in wv})==1:
                if iv[0] != wv[0]:
                    hits.append({
                      "step":t,"feature":key,
                      "improved_value":iv[0],"worsened_value":wv[0],
                      "improved_seeds":improved,"worsened_seeds":worsened,
                    })
        if hits:
            return hits
    return []

def main():
    per_seed={}
    for s in SEEDS:
        per_seed[s]=run(s)
        print("SIGN_TRACE_DONE",s,SIGN[s],len(per_seed[s]))
    hits=group_uniform_separator(per_seed)
    first_step=hits[0]["step"] if hits else None
    snapshot={}
    if first_step is not None:
        for s in SEEDS:
            snapshot[str(s)]={
              "sign":SIGN[s],
              "features":per_seed[s][first_step]["features"],
            }
    out={
      "schema":"pruning-first-sign-separating-divergence-v0",
      "target":TARGET,
      "definition":"first turn where at least one candidate-minus-baseline observable feature is uniform within improved6, uniform within worsened4, and differs between groups",
      "first_step":first_step,
      "separating_features":hits,
      "first_step_snapshot":snapshot,
      "boundary":"diagnostic group separator only; no causal claim and no policy promotion",
    }
    Path("pruning_first_sign_separating_divergence_v0.json").write_text(
      json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("SIGN_SEPARATOR_SUMMARY",json.dumps({
      "first_step":first_step,
      "feature_count":len(hits),
      "features":[{"feature":h["feature"],"improved":h["improved_value"],"worsened":h["worsened_value"]} for h in hits],
    },separators=(",",":")))

if __name__=="__main__":
    main()
