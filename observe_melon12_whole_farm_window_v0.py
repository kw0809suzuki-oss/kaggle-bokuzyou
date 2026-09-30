#!/usr/bin/env python3
import json, gzip, zipfile, collections, sys

ZIP = sys.argv[1]
START, END = 257, 322
TARGET_NAME = "melon12_v1_92801801_candidate_v1_seat0_battle.json.gz"
DAY0 = {(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,3),(0,4)}

with zipfile.ZipFile(ZIP) as z:
    raw = z.read(TARGET_NAME)
battle = json.loads(gzip.decompress(raw).decode("utf-8"))
turns = [t for t in battle["turns"] if START <= t["step"] <= END]

def op(a):
    return a[0] if isinstance(a, list) and a else "NONE"

def workers(t):
    farm=t["pre"]["public"]["farms"][0]
    a=t["action_seat0"]
    out=[("farmer",0,farm["farmer"],a.get("farmer"))]
    for i,act in enumerate(a.get("hands",[]) or []):
        pos=(farm.get("hands") or [])[i] if i < len(farm.get("hands") or []) else None
        out.append(("hand",i,pos,act))
    return out

def mature_coords(t):
    farm=t["pre"]["public"]["farms"][0]
    out=[]
    for x,y in DAY0:
        tile=farm["tiles"][y][x]
        if isinstance(tile,dict) and tile.get("kind")=="PLANT" and tile.get("crop")=="MELON" and tile.get("planted_day")==0 and int(tile.get("yield_units",0) or 0)>=6:
            out.append([x,y])
    return sorted(out)

def carry(t):
    invs=t["pre"]["private_seat0"].get("inventories",[]) or []
    out=[]
    for idx,inv in enumerate(invs):
        n=int((inv or {}).get("MELON",0) or 0)
        if n:
            out.append({"unit_index":idx,"melon":n})
    return out

action_counts=collections.Counter()
active_counts=collections.Counter()
movement=0
records=[]
mature_presence=collections.Counter()
mature_tile_contacts=[]
for t in turns:
    ms=mature_coords(t)
    ws=workers(t)
    cs=carry(t)
    for _,_,_,a in ws:
        action_counts[op(a)] += 1
        if op(a) not in ("PASS","NONE"):
            active_counts[op(a)] += 1
        if op(a) in ("NORTH","SOUTH","EAST","WEST"):
            movement += 1
    for xy in ms:
        mature_presence[tuple(xy)] += 1
    for who,i,pos,a in ws:
        if pos and list(pos) in ms:
            mature_tile_contacts.append({
                "step":t["step"],"coord":list(pos),"worker":who,"index":i,"action":a
            })
    records.append({
        "step":t["step"],
        "day":t["day"],"hour":t["hour"],
        "mature_unharvested":ms,
        "worker_count":len(ws),
        "actions":collections.Counter(op(a) for _,_,_,a in ws),
        "carry_melon":cs,
        "shed_melon":int(t["pre"]["private_seat0"].get("shed",{}).get("MELON",0) or 0),
    })

summary={
    "window":[START,END],
    "steps":len(turns),
    "action_counts":dict(action_counts),
    "mature_presence_steps":{f"{x},{y}":n for (x,y),n in sorted(mature_presence.items())},
    "steps_with_mature_backlog":sum(1 for r in records if r["mature_unharvested"]),
    "max_mature_backlog":max((len(r["mature_unharvested"]) for r in records),default=0),
    "steps_with_any_melon_carry":sum(1 for r in records if r["carry_melon"]),
    "max_total_carried_melon":max((sum(x["melon"] for x in r["carry_melon"]) for r in records),default=0),
    "mature_tile_contacts":mature_tile_contacts,
    "records":records,
}
print(json.dumps(summary,ensure_ascii=False,indent=2))
