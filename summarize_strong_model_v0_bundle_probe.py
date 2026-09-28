"""Validate paired bundle-probe artifacts and report observations without tuning."""
import gzip,json,hashlib
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'battle-results'/'bundle-probe'
BASE=ROOT/'battle-results'/'repair-probe'
selection=json.loads((OUT/'selection.json').read_text())
assert selection['seed']==92804001 and selection['step']==0
candidate=selection['diagnostic'];assert candidate['diagnostic_beats_current_by_central_estimate']
expected={'strong_model_v0_reimplementation/agent.py':'ec818185babc2c0ba8a784d41b029ccf495a3445da3dc2583f4b526eb60bd13e',
 'strong_model_v0_reimplementation/jobs.py':'f9da3e34af3e21172893145156bda258e4184eb20a9d0fd1499396aa1b574913',
 'strong_model_v0_reimplementation/planner.py':'d42b70521236316351276067675eb7a2b0234e30686074207fcb35459e7f6163'}
production={'BUILD_COOP','BUILD_PASTURE','PLANT','WATER','HARVEST','FEED','CARE','PLACE'}
movement={'NORTH','SOUTH','EAST','WEST'}
rows=[];throughput=[];snapshots={}
for seat in (0,1):
 run=json.loads((OUT/f'bundle-seat{seat}.summary.json').read_text())
 control=json.loads((BASE/f'repaired-92804001-seat{seat}.summary.json').read_text())
 assert run['seed']==control['seed']==92804001 and run['seat']==control['seat']==seat
 assert run['statuses']==control['statuses']==['DONE','DONE'] and run['steps']==control['steps']==719
 assert run['official_rules_sha256']==control['official_rules_sha256']
 assert run['opponent_sha256']==control['opponent_sha256']
 for k,v in expected.items():assert run['source_sha256'][k]==control['source_hashes'][k]==v,(seat,k)
 delta={'self_cash':run['self_cash']-control['self_cash'],'opponent_cash':run['opponent_cash']-control['opponent_cash'],'margin':run['margin']-control['margin']}
 rows.append({'seat':seat,'control_self':control['self_cash'],'bundle_self':run['self_cash'],'self_delta':delta['self_cash'],
              'control_opponent':control['opponent_cash'],'bundle_opponent':run['opponent_cash'],'opponent_delta':delta['opponent_cash'],
              'control_margin':control['margin'],'bundle_margin':run['margin'],'margin_delta':delta['margin'],'bundle_status':run['statuses'],'steps':run['steps']})
 bt=json.loads((OUT/f'bundle-seat{seat}.trace.json').read_text())[:15]
 ct=json.loads((BASE/f'repaired-92804001-seat{seat}.trace.json').read_text())[:15]
 def counts(trace):
  c=Counter();slots=work=prod=move=0
  for r in trace:
   acts=[r['action'].get('farmer',['PASS']),*r['action'].get('hands',[])]
   slots+=len(acts);work+=sum(a!=['PASS'] for a in acts)
   prod+=sum(a[0] in production for a in acts);move+=sum(a[0] in movement for a in acts)
   c.update(a[0] for a in acts)
  return {'action_slots':slots,'nonpass_actions':work,'production_actions':prod,'movement_actions':move,'actions':dict(c)}
 throughput.append({'seat':seat,'control':counts(ct),'bundle':counts(bt)})
 replay=json.loads(gzip.decompress((OUT/f'bundle-seat{seat}.replay.json.gz').read_bytes()))
 obs=replay['steps'][1][seat]['observation']
 farm=obs['farms'][seat]
 assert obs.get('step',1)==1 and len(farm['hands'])==2
 assert farm['money']==2058
 assert obs['private']['seeds'].get('MELON')==6 and obs['private']['seeds'].get('WHEAT')==6
 assert obs['private']['shed'].get('COW')==1
 for step in (1,2,24,48,120,192,336,504,696,719):
  state=replay['steps'][step][0]['observation']; f=state['farms'][seat]
  assets=Counter(t.get('crop') or t.get('animal') or t.get('kind') for line in f['tiles'] for t in line if isinstance(t,dict))
  snapshots.setdefault(str(seat),[]).append({'step':step,'self_cash':f['money'],'hands':len(f['hands']),'quadrants':len(f['unlocked_quadrants']),'assets':dict(assets)})

comparison={'schema':'strong-model-v0-bundle-probe-result-v0','seed':92804001,'seat_swap':True,'episodes':2,
 'selection':{'incumbent_action':candidate['current_choice']['action'],'incumbent_central_estimate':candidate['current_choice']['terminal_cash'],
             'bundle_action':candidate['action'],'bundle_central_estimate':candidate['central_diagnostic_cash'],
             'bundle_minus_incumbent_estimate':candidate['central_diagnostic_cash']-candidate['current_choice']['terminal_cash'],
             'bundle_ranked_higher':candidate['diagnostic_beats_current_by_central_estimate']},
 'world':rows,'opening_throughput_first15':throughput,'bundle_daily_snapshots':snapshots,
 'boundary':'One seed with two seat placements. The existing control artifacts were run at the same seed, seats, Official rules, opponent and core source hashes. This is not an independent-seed strength benchmark. The forecast omits automatic future rehire/reinvestment. No model tuning followed the terminal observations.'}
(OUT/'world-comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
report='''# Strong Model｜Bundle Probe 結果\n\n## 目的と候補\n\n親目的はStrong Modelを強くすること。このProbeでは、現行比較に労働・資産・実行をまとめた初期運転を一件加えたときのSelectionとWorldの応答を観測した。Independentは候補内容の参照にだけ使用。候補束はstep 0で HIRE×2、COW×1、MELON seed×6、WHEAT seed×6、Pasture建設とし、12作物仕事・COW配置仕事を運転予定に含めた。以後は既存Strongが毎turn再計画した。\n\n## Selection｜同じstep 0 State\n\n| 候補 | 中央終局Cash見積り | 初手 |\n|---|---:|---|\n| 現行Strongの選択 | 8,628 | BUILD_PASTURE、BUY_ANIMAL SHEEP×1 |\n| 診断Bundle | 13,440 | BUILD_PASTURE、HIRE×2、BUY_ANIMAL COW×1、MELON/WHEAT seed各6 |\n\nBundleは現行選択より4,812高く評価され、現行候補を残した比較で上位になった。初手後は市場処理され、step 1の観測Cash2,058、雇用Hands2、COW在庫1、MELON/WHEAT seed各6、Pasture1面を確認。\n\n## World｜同seed・席入替\n\n| 席 | 対照self | Bundle self | 差 | 対照margin | Bundle margin | 差 |\n|---:|---:|---:|---:|---:|---:|---:|\n'''
for r in rows:report+=f"| {r['seat']} | {r['control_self']:,.0f} | {r['bundle_self']:,.0f} | {r['self_delta']:+,.0f} | {r['control_margin']:,.0f} | {r['bundle_margin']:,.0f} | {r['margin_delta']:+,.0f} |\n"
report+='''\n両席ともDONE、719行動で終了。Bundleは両席でself Cashが下がり、marginは両席で改善した。改善を勝利とは扱えない。どちらのmarginも負のまま。Selection上位とWorldのself方向は一致しなかった。\n\nstep 0〜14の実行記録では、席0の非PASS行動枠は対照27からBundle42、生産系行動は9から13へ変化。診断Bundle後も、step 1に既存StrongはBUY_LANDとSHEEP購入を発行し、step 2時点で2面目が解放されていた。初期Bundleが実行されたことと、以後の運転全体がBundleに置き換わったことは同じではない。\n\n## このEvidenceの境界\n\n確認できたのは、一件の候補束が同一Stateの比較に入り、中央見積り上位となったこと、初手の購入・雇用・建設が成立したこと、席入替二戦の終端応答。selfとmarginは異なる向きに動いた。\n\n一seed二席なので一般的な強さは未確定。bundle評価は将来の自動再雇用・再投資を含まない。候補の各要素が結果へ与えた寄与や、途中のどの差が終端差を作ったかはこのProbeでは判定していない。結果を見てモデルや候補を追加修正していない。\n\n## 成果物\n\n- `selection.json`：step 0 raw State、現行選択、束評価\n- `bundle-seat0/1.summary.json`：終端・実行時間・hash\n- `bundle-seat0/1.trace.json`：初期15turnと各日の判断\n- `bundle-seat0/1.replay.json.gz`：Raw Replay\n- `world-comparison.json`：対照との比較・日次State\n- `run_strong_model_v0_bundle_world_probe.py`：実行runner\n- `summarize_strong_model_v0_bundle_probe.py`：artifact照合と集計\n- `test_strong_model_v0_bundle_probe.py`：候補の最小Contract\n'''
(OUT/'RESULT.md').write_text(report)
print(json.dumps({'selection':comparison['selection'],'world':rows,'throughput':throughput},ensure_ascii=False))
