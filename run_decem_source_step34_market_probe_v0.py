#!/usr/bin/env python3
import json
from pathlib import Path
from kaggle_environments import api

EPISODE_ID = 115303987
SOURCE_SEAT = 0

def find_replay(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() == "replay":
                if isinstance(v, str):
                    try:
                        return json.loads(v)
                    except Exception:
                        pass
                if isinstance(v, (dict, list)):
                    return v
        for v in x.values():
            r = find_replay(v)
            if r is not None:
                return r
    elif isinstance(x, list):
        for v in x:
            r = find_replay(v)
            if r is not None:
                return r
    return None

def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k,v in x.items()}
    if isinstance(x, list):
        return [plain(v) for v in x]
    return x

resp = api.get_episode_replay(EPISODE_ID)
replay = find_replay(resp)
if replay is None:
    # Some API versions return the replay document directly.
    if isinstance(resp, dict) and "steps" in resp:
        replay = resp
    else:
        out = {
            "schema":"decem-source-step34-market-v0",
            "episode_id":EPISODE_ID,
            "error":"replay_not_found",
            "response_keys":list(resp.keys()) if isinstance(resp,dict) else str(type(resp)),
            "response_preview":str(resp)[:2000],
        }
        Path("decem_source_step34_market_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print("SOURCE_STEP34_MARKET "+json.dumps(out,separators=(",",":")))
        raise SystemExit(0)

steps = replay.get("steps") if isinstance(replay,dict) else None
if not isinstance(steps,list):
    out = {
        "schema":"decem-source-step34-market-v0",
        "episode_id":EPISODE_ID,
        "error":"steps_not_found",
        "replay_keys":list(replay.keys()) if isinstance(replay,dict) else str(type(replay)),
    }
else:
    def seat_state(idx):
        if idx >= len(steps): return None
        row=steps[idx]
        if not isinstance(row,list) or SOURCE_SEAT >= len(row): return None
        return plain(row[SOURCE_SEAT])

    s33=seat_state(33); s34=seat_state(34); s35=seat_state(35)
    def summarize(s):
        if not isinstance(s,dict): return s
        obs=s.get("observation")
        if isinstance(obs,str):
            try: obs=json.loads(obs)
            except Exception: pass
        return {
            "action":plain(s.get("action")),
            "observation":plain(obs),
            "reward":s.get("reward"),
            "status":s.get("status"),
        }
    out = {
        "schema":"decem-source-step34-market-v0",
        "episode_id":EPISODE_ID,
        "seat":SOURCE_SEAT,
        "num_steps":len(steps),
        "step33":summarize(s33),
        "step34":summarize(s34),
        "step35":summarize(s35),
    }

Path("decem_source_step34_market_v0.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("SOURCE_STEP34_MARKET "+json.dumps(out,separators=(",",":")))
