#!/usr/bin/env python3
import json
from pathlib import Path
import requests

EPISODE_ID=115303987
SOURCE_SEAT=0
URL=f"https://www.kaggle.com/competitions/episodes/{EPISODE_ID}/replay.json"

def plain(x):
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,list): return [plain(v) for v in x]
    return x

r=requests.get(URL,timeout=30)
meta={"status_code":r.status_code,"content_type":r.headers.get("content-type"),"url":r.url}
try:
    replay=r.json()
except Exception:
    out={"schema":"decem-source-step34-market-v0","episode_id":EPISODE_ID,"error":"non_json_response","http":meta,"preview":r.text[:2000]}
    Path("decem_source_step34_market_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SOURCE_STEP34_MARKET "+json.dumps(out,separators=(",",":")))
    raise SystemExit(0)

if isinstance(replay,dict) and "replay" in replay and isinstance(replay["replay"],str):
    replay=json.loads(replay["replay"])
steps=replay.get("steps") if isinstance(replay,dict) else None
if not isinstance(steps,list):
    out={"schema":"decem-source-step34-market-v0","episode_id":EPISODE_ID,"error":"steps_not_found","http":meta,"keys":list(replay.keys()) if isinstance(replay,dict) else str(type(replay))}
else:
    def summarize(idx):
        s=plain(steps[idx][SOURCE_SEAT])
        obs=s.get("observation")
        if isinstance(obs,str):
            try: obs=json.loads(obs)
            except Exception: pass
        return {"action":s.get("action"),"observation":obs,"reward":s.get("reward"),"status":s.get("status")}
    out={"schema":"decem-source-step34-market-v0","episode_id":EPISODE_ID,"seat":SOURCE_SEAT,"http":meta,"num_steps":len(steps),"step33":summarize(33),"step34":summarize(34),"step35":summarize(35)}
Path("decem_source_step34_market_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SOURCE_STEP34_MARKET "+json.dumps(out,separators=(",",":")))
