"""Reproducible full-season comparison with raw replay and timing evidence."""
import argparse
import collections
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',default=str(Path(__file__).resolve().parent))
parser.add_argument('--seed',type=int,default=92804001)
parser.add_argument('--seat',type=int,choices=(0,1),default=0)
parser.add_argument('--label',default='repaired')
parser.add_argument('--model',choices=('strong','independent'),default='strong')
parser.add_argument('--out-dir',default='battle-results/repair-probe')
args=parser.parse_args()
root=Path(args.source_root).resolve();sys.path.insert(0,str(root))
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules
from strong_model_v0_reimplementation.agent import agent,reset_agent,debug_state
if args.model=='independent':
    from astra_flow_independent_distilled_v0 import agent,reset_agent
opponent_path=root/'astra_flow_vendor'/'seyamalam_v21.py'
spec=importlib.util.spec_from_file_location('probe_opponent',opponent_path)
opponent=importlib.util.module_from_spec(spec);spec.loader.exec_module(opponent)
if hasattr(opponent,'reset_agent'):opponent.reset_agent()
reset_agent()
out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
name=f'{args.label}-{args.seed}-seat{args.seat}'
latencies=[];trace=[];counts=collections.Counter()
def measured(obs,cfg):
    started=time.perf_counter();action=agent(obs,cfg);latencies.append(time.perf_counter()-started)
    counts[action['farmer'][0]]+=1
    if obs['step']<15 or obs['hour']==0:
        row={'step':obs['step'],'cash':obs['farms'][args.seat]['money'],'action':action,'seconds':latencies[-1]}
        if args.model=='strong':row['debug']=debug_state(args.seat)
        trace.append(row)
        (out/f'{name}.progress.json').write_text(json.dumps(row,default=str))
    return action

env=make('kaggriculture',configuration={'seed':args.seed},debug=False)
agents=[opponent.agent,opponent.agent];agents[args.seat]=measured
env.run(agents)
self_cash=float(env.state[args.seat].observation.farms[args.seat]['money'])
opp_cash=float(env.state[args.seat].observation.farms[1-args.seat]['money'])
hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'label':args.label,'model':args.model,'seed':args.seed,'seat':args.seat,'self_cash':self_cash,'opponent_cash':opp_cash,'margin':self_cash-opp_cash,'statuses':[str(s.status) for s in env.state],'steps':len(env.steps)-1,'farmer_actions':dict(counts),'mean_seconds':statistics.mean(latencies),'max_seconds':max(latencies),'remaining_overage':env.state[args.seat].observation.get('remainingOverageTime'),'source_hashes':{str(p.relative_to(root)):hash_file(p) for p in (root/'strong_model_v0_reimplementation').glob('*.py')},'official_rules_sha256':hash_file(rules.__file__),'opponent_sha256':hash_file(opponent_path)}
(out/f'{name}.summary.json').write_text(json.dumps(result,indent=2)+'\n')
(out/f'{name}.trace.json').write_text(json.dumps(trace,indent=2,default=str)+'\n')
with gzip.open(out/f'{name}.replay.json.gz','wt') as f:json.dump(env.toJSON(),f,default=str)
print(json.dumps(result))
