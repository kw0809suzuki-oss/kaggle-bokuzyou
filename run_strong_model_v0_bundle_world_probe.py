"""Run exactly the frozen diagnostic opening against the pinned World/opponent."""
from __future__ import annotations
import copy,gzip,hashlib,json,statistics,time
from importlib import import_module
from pathlib import Path
from typing import Any
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules
from run_strong_model_v0_reimplementation_contract import plain
strong_module=import_module("strong_model_v0_reimplementation.agent")
from strong_model_v0_reimplementation.planner import settings_from
from strong_model_v0_bundle_probe import SEED,build_diagnostic_bundle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'battle-results'/'bundle-probe';OUT.mkdir(parents=True,exist_ok=True)
OPPONENT_PATH=ROOT/'astra_flow_vendor'/'seyamalam_v21.py'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def obs_for(env,seat):return plain(env._Environment__get_shared_state(seat)['observation'])
def load_opponent():
 import importlib.util
 spec=importlib.util.spec_from_file_location('bundle_probe_opponent',OPPONENT_PATH)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def selection():
 env=make('kaggriculture',configuration={'seed':SEED},debug=False);env.reset(2)
 raw=obs_for(env,0);diagnostic=build_diagnostic_bundle(raw)
 result={'schema':'strong-model-v0-bundle-selection-v0','seed':SEED,'seat':0,'step':raw['step'],'starting_cash':raw['farms'][0]['money'],'current_state':raw,'diagnostic':diagnostic,
  'interpretation_boundary':'Candidate contents are one diagnostic opening derived from the user-provided same-seed Independent observation. Rankings use the existing limited rollout, which does not model automatic future rehire/reinvestment.'}
 (OUT/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 return result

def battle(seat:int,diagnostic:dict[str,Any]):
 opponent=load_opponent();reset=getattr(opponent,'reset_agent',None)
 if callable(reset):reset()
 strong_module.reset_agent()
 env=make('kaggriculture',configuration={'seed':SEED},debug=False)
 latencies=[]; trace=[]; action_counts={}; injected=[]; baseline_cash=None
 def probe_agent(obs,cfg):
  nonlocal baseline_cash
  started=time.perf_counter()
  if int(obs['step'])==0:
   # Install the frozen proposal and commitments; ordinary Strong resumes at step 1.
   rt=strong_module.Runtime(settings_from(cfg));rt.last_step=0
   rt.active={s['key']:copy.deepcopy(s) for s in diagnostic['commitments']}
   strong_module._RUNTIMES[int(obs['player'])]=rt
   action=copy.deepcopy(diagnostic['action'])
   baseline_cash=float(obs['farms'][seat]['money'])
   injected.append({'step':0,'action':copy.deepcopy(action),'cash_before':baseline_cash})
  else:action=strong_module.agent(obs,cfg)
  elapsed=time.perf_counter()-started;latencies.append(elapsed)
  for a in [action.get('farmer',['PASS']),*action.get('hands',[])]:
   key=a[0];action_counts[key]=action_counts.get(key,0)+1
  if int(obs['step'])<15 or int(obs['hour'])==0:
   trace.append({'step':int(obs['step']),'cash':obs['farms'][seat]['money'],'hands':len(obs['farms'][seat].get('hands',[])),'action':copy.deepcopy(action),'elapsed_seconds':elapsed,
    'debug':None if int(obs['step'])==0 else strong_module.debug_state(seat)})
   (OUT/f'bundle-seat{seat}.progress.json').write_text(json.dumps(trace[-1],default=str))
  return action
 agents=[opponent.agent,opponent.agent];agents[seat]=probe_agent
 env.run(agents)
 self_cash=float(env.state[seat].observation.farms[seat]['money']);opp_cash=float(env.state[1-seat].observation.farms[1-seat]['money'])
 result={'model':'Strong with one diagnostic initial bundle','seed':SEED,'seat':seat,'self_cash':self_cash,'opponent_cash':opp_cash,'margin':self_cash-opp_cash,'statuses':[str(x.status) for x in env.state],'steps':len(env.steps)-1,
  'injected':injected,'farmer_and_hand_actions':action_counts,'mean_decision_seconds':statistics.mean(latencies),'max_decision_seconds':max(latencies),'remaining_overage':env.state[seat].observation.get('remainingOverageTime'),
  'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'strong_model_v0_reimplementation/agent.py',ROOT/'strong_model_v0_reimplementation/jobs.py',ROOT/'strong_model_v0_reimplementation/planner.py',ROOT/'strong_model_v0_bundle_probe.py']},
  'official_rules_sha256':sha(rules.__file__),'opponent_sha256':sha(OPPONENT_PATH)}
 name=f'bundle-seat{seat}'
 (OUT/f'{name}.summary.json').write_text(json.dumps(result,indent=2)+'\n')
 (OUT/f'{name}.trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2,default=str)+'\n')
 with gzip.open(OUT/f'{name}.replay.json.gz','wt') as f:json.dump(env.toJSON(),f,default=str)
 return result

def main():
 diagnostic=selection()['diagnostic']
 controls=[]
 for seat in (0,1):
  p=ROOT/'battle-results'/'repair-probe'/f'repaired-92804001-seat{seat}.summary.json'
  control=json.loads(p.read_text())
  for key,want in [('seed',SEED),('seat',seat)]:assert control[key]==want,(p,key,control[key])
  assert control['source_hashes']['strong_model_v0_reimplementation/agent.py']==sha(ROOT/'strong_model_v0_reimplementation/agent.py')
  assert control['source_hashes']['strong_model_v0_reimplementation/jobs.py']==sha(ROOT/'strong_model_v0_reimplementation/jobs.py')
  assert control['source_hashes']['strong_model_v0_reimplementation/planner.py']==sha(ROOT/'strong_model_v0_reimplementation/planner.py')
  controls.append({'seat':seat,'self_cash':control['self_cash'],'opponent_cash':control['opponent_cash'],'margin':control['margin'],'status':control['statuses'],'steps':control['steps']})
 results=[battle(seat,diagnostic) for seat in (0,1)]
 for r in results:assert r['statuses']==['DONE','DONE'] and r['steps']==719,r
 report={'schema':'strong-model-v0-bundle-world-probe-v0','seed':SEED,'control_source_commit':'0ba1f3f2eb2c5bf85376ec86f257769b20699880','bundle_results':results,'matched_controls':controls,
  'boundary':'One seed with seat swap (2 episodes), not an independent-seed strength benchmark. Initial candidate is changed once at step 0; the existing Strong resumes thereafter. No tuning follows the observed terminal result.'}
 (OUT/'world-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'selection':diagnostic,'world':report},ensure_ascii=False))
if __name__=='__main__':main()
