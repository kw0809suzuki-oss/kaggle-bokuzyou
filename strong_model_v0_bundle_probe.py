"""One-state diagnostic: add the observed initial operating bundle as one candidate."""
from __future__ import annotations
import copy
from typing import Any
from strong_model_v0_reimplementation.jobs import Job, all_tiles, distance, positions
from strong_model_v0_reimplementation.planner import (
    Settings, choose, empty_action, rollout,
)

LABEL='labor_assets_execution_diagnostic'
SEED=92804001


def _candidate(raw:dict[str,Any]):
    p=int(raw['player']); farm=raw['farms'][p]
    available=[xy for xy,t in all_tiles(raw) if t is None]
    animal_tile=min(available,key=lambda xy:(min(distance(pos,xy) for pos in positions(raw)),xy))
    available.remove(animal_tile)
    targets=sorted(available,key=lambda xy:(min(distance(pos,xy) for pos in positions(raw)),xy))[:12]
    if len(targets)<12: raise ValueError('diagnostic bundle needs 12 unlocked empty crop tiles')
    commitments=[]
    for crop in ('MELON','WHEAT'):
        for xy in targets[:6] if crop=='MELON' else targets[6:12]:
            commitments.append(Job(f'probe:plant:{crop}:{xy[0]}:{xy[1]}','plant_crop','production_start',{'tile':list(xy),'crop':crop},100,0))
    commitments.append(Job(f'probe:cow:{animal_tile[0]}:{animal_tile[1]}','establish_animal','production_start',{'tile':list(animal_tile),'animal':'COW'},100,0))
    action=empty_action(raw)
    action['market']=[['HIRE'],['HIRE'],['BUY_ANIMAL','COW',1],['BUY_SEED','MELON',6],['BUY_SEED','WHEAT',6]]
    action['farmer']=['BUILD_PASTURE']
    return commitments,action


def build_diagnostic_bundle(raw:dict[str,Any],cfg:Settings|None=None):
    cfg=cfg or Settings()
    commitments,action=_candidate(raw)
    current,representative,active,continuation=choose(raw,cfg,{})
    value,_=rollout(raw,cfg,commitments,first_action=action)
    stressed,_=rollout(raw,cfg,commitments,first_action=action,stress=True)
    return {
        'label':LABEL,'seed':SEED,'step':int(raw.get('step',0)),
        'candidate_origin':'user-provided same-seed Independent opening, used only to specify diagnostic contents',
        'action':copy.deepcopy(action),'commitments':[j.spec() for j in commitments],
        'strict_diagnostic_cash':stressed,'central_diagnostic_cash':value,
        'current_choice':{
            'action':copy.deepcopy(current.action),
            'representative':None if representative is None else representative.spec(),
            'active':[j.spec() for j in active],
            'terminal_cash':current.envelope.central_cash,
            'strict_cash':current.envelope.strict_cash,
            'continuation_cash':continuation.envelope.central_cash,
        },
        'diagnostic_beats_current_by_central_estimate':value>current.envelope.central_cash,
        'alternatives':'Original choose() alternatives remain intact; the reported current_choice is the choice without this diagnostic candidate.',
    }
