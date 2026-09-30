#!/usr/bin/env python3
"""World Lens Contact Map v0.

Observe a full Adaptive Replay episode and map places where the agent touches
shared/external world state. No behavior changes.

Surfaces:
- SELL market actions
- BUY_* market actions
- market actions immediately after a Town/shop change
- market-touch turns annotated with seat/order context

Purpose:
Find where Replay acts on shared world while a Current-World check may matter.
"""

from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
from kaggle_environments import make

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"adaptive_replay_contract_runtime_v0.py"
OPP=ROOT/"astra_flow_vendor"/"seyamalam_v21.py"
SEEDS=[92802001,92802002,92802003,92802004,92802005,92802006,92802007,92802008,92802009,92802010]

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    if isinstance(x,(str,int,float,bool)) or x is None:return x
    if hasattr(x,"items"): return {str(k):plain(v) for k,v in x.items()}
    if hasattr(x,"__iter__") and not isinstance(x,(str,bytes)):return [plain(v) for v in x]
    return x

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    if hasattr(mod,"reset_agent"):mod.reset_agent()
    return mod

def shared(env,seat):
    return plain(env._Environment__get_shared_state(seat)["observation"])

def market_ops(bundle):
    out=[]
    for a in bundle.get("market",[]) or []:
        if isinstance(a,(list,tuple)) and a:
            out.append(list(a))
    return out

def classify(op):
    if not op:return "OTHER"
    x=str(op[0])
    if x=="SELL": return "SELL"
    if x.startswith("BUY_"): return "BUY"
    if x=="HIRE": return "HIRE"
    return "OTHER"

def run(seed):
    m=load(MODEL,f"model_{seed}");o=load(OPP,f"opp_{seed}")
    env=make("kaggriculture",configuration={"seed":seed},debug=False);env.reset(num_agents=2)
    prev_town=None; contacts=[]
    while not env.done:
        a=shared(env,0);b=shared(env,1)
        step=int(a.get("step",0) or 0)
        act0=plain(m.agent(a,env.configuration)); act1=plain(o.agent(b))
        town=plain(a.get("town") or {})
        town_changed = prev_town is not None and town!=prev_town
        ops=market_ops(act0)
        if ops or town_changed:
            prices=plain((a.get("market") or {}).get("prices") or {})
            inv=plain((a.get("market") or {}).get("inventory") or {})
            contacts.append({
                "step":step,
                "day":int(a.get("day",0) or 0),
                "hour":int(a.get("hour",0) or 0),
                "seat":0,
                "processing_order_hint":"seat0_then_seat1_in_env.step_input",
                "town_changed":town_changed,
                "town":town,
                "market_ops":ops,
                "op_classes":[classify(x) for x in ops],
                "prices":prices,
                "inventory":inv,
            })
        prev_town=town
        env.step([act0,act1])
    return contacts

def main():
    rows=[]; summary={"SELL":0,"BUY":0,"HIRE":0,"OTHER":0,"town_change_contacts":0,"market_touch_turns":0}
    for seed in SEEDS:
        cs=run(seed)
        rows.append({"seed":seed,"contacts":cs})
        touched=sum(1 for c in cs if c["market_ops"])
        summary["market_touch_turns"]+=touched
        summary["town_change_contacts"]+=sum(1 for c in cs if c["town_changed"])
        for c in cs:
            for k in c["op_classes"]:
                summary[k]=summary.get(k,0)+1
        print("SEED "+json.dumps({
            "seed":seed,
            "contacts":len(cs),
            "market_touch_turns":touched,
            "town_changes":sum(1 for c in cs if c["town_changed"]),
            "sell_ops":sum(k=="SELL" for c in cs for k in c["op_classes"]),
            "buy_ops":sum(k=="BUY" for c in cs for k in c["op_classes"]),
            "hire_ops":sum(k=="HIRE" for c in cs for k in c["op_classes"]),
        },separators=(",",":")))
    print("SUMMARY "+json.dumps(summary,separators=(",",":")))
    Path("world_lens_contact_map_v0_result.json").write_text(
      json.dumps({"schema":"world-lens-contact-map-v0","summary":summary,"cases":rows,
      "boundary":[
        "Observation only; no action behavior changed.",
        "Order annotation is execution-input order, not yet proof of internal shared-market mutation order.",
        "Contact counts identify observation surfaces, not policy rules."
      ]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )

if __name__=="__main__":main()
