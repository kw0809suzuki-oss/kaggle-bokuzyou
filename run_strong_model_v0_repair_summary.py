"""Verify terminal summaries against raw replays and aggregate one paired probe."""
import gzip,json,collections,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'battle-results/repair-probe'
rows=[]; trajectories={}
for file in sorted(out.glob('*.summary.json')):
 s=json.loads(file.read_text()); replay=json.loads(gzip.decompress(file.with_name(file.name.replace('.summary.json','.replay.json.gz')).read_bytes()))
 seat=s['seat']; last=replay['steps'][-1]
 assert s['statuses']==[x['status'] for x in last]
 # Shared fields are stored once (seat 0) in Kaggle replay.
 farms=last[0]['observation']['farms']
 assert farms[seat]['money']==s['self_cash'] and farms[1-seat]['money']==s['opponent_cash']
 orders=collections.Counter();farmer=collections.Counter();hands=collections.Counter()
 snapshots=[]
 for step,state in enumerate(replay['steps']):
  if step:
   action=state[seat].get('action') or {}
   farmer[action.get('farmer',['PASS'])[0]]+=1
   for a in action.get('hands',[]):hands[a[0]]+=1
   for order in action.get('market',[]):orders[order[0]]+=1
  obs=state[0]['observation'];farm=obs['farms'][seat]
  if step%24==0 or step==719:
   tiles=collections.Counter(t.get('crop') or t.get('animal') or t.get('kind') for row in farm['tiles'] for t in row if isinstance(t,dict))
   snapshots.append({'step':step,'cash':farm['money'],'assets':dict(tiles),'quadrants':len(farm['unlocked_quadrants'])})
 assert dict(farmer)==s['farmer_actions'],(s['label'],seat,farmer,s['farmer_actions'])
 trajectories[f"{s['label']}-seat{seat}"]={'orders_requested':dict(orders),'farmer_actions':dict(farmer),'hand_actions':dict(hands),'snapshots':snapshots}
 rows.append({k:s[k] for k in ('label','seed','seat','self_cash','opponent_cash','margin','statuses','steps','mean_seconds','max_seconds','remaining_overage')})
assert len(rows)==6, 'Run all three models in both seats before summarizing.'
result={'scope':'One seed, both seats; not independent seeds or adoption benchmark','seed':92804001,'implementation_commit':'0ba1f3f2eb2c5bf85376ec86f257769b20699880','baseline_commit':'6dcfcb8280027e4b086ba5bca845c5e49382ab83','official_commit':'d7729da06cc1382eb742d6980dc3180aa85caa28','independent_source_sha256':hashlib.sha256((root/'astra_flow_independent_distilled_v0.py').read_bytes()).hexdigest(),'rows':rows,'trajectories':trajectories,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.name.endswith(('.replay.json.gz','.summary.json','.trace.json'))}}
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
