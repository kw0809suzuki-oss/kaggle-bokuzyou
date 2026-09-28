"""Cash must be earned by an executable sequence, including the final sale."""
import copy
import json
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules
from run_strong_model_v0_reimplementation_contract import plain
from strong_model_v0_reimplementation.jobs import Job, fresh_jobs
from strong_model_v0_reimplementation.planner import Settings, terminal_envelope, investment_jobs, rollout, schedule, _project_one_turn


def world():
    env=make('kaggriculture',configuration={'seed':92809901,'weedSpawnChance':0,'townShopUnlockInterval':1000},debug=False)
    env.reset(2)
    return env,plain(env._Environment__get_shared_state(0)['observation'])


def main():
    cfg=Settings(); env,raw=world()
    unworked=copy.deepcopy(raw); unworked['private']['seeds']['MELON']=1
    assert terminal_envelope(unworked,cfg).central_cash==3000
    investments=investment_jobs(raw,cfg,fresh_jobs(raw,720,24,10))
    wheat=next(j for j in investments if j.kind=='plant_crop' and j.target=={'tile':[4,4],'crop':'WHEAT'})
    cash,trace=rollout(raw,cfg,[wheat],trace=True)
    actions={a['action']['farmer'][0] for a in trace}
    orders={o[0] for a in trace for o in a['action']['market']}
    assert {'PLANT','WATER','HARVEST'}<=actions, actions
    assert {'BUY_SEED','SELL'}<=orders,orders
    assert sum(row['action']['farmer'][0]=='PLANT' for row in trace)==1
    assert sum(o[0]=='BUY_SEED' for row in trace for o in row['action']['market'])==1
    for row in trace:
        assert row['step']==len(env.steps)-1
        env.step([row['action'],{'farmer':['PASS'],'hands':[],'market':[]}])
    actual=env.state[0].observation.farms[0]['money']
    assert cash==actual and cash>3000,(cash,actual)
    terminal={}
    for step,pos in [(718,[4,4]),(718,[3,4]),(719,[4,4])]:
        end=copy.deepcopy(raw);end.update(step=step,day=step//24,hour=step%24)
        end['farms'][0]['farmer']=pos;end['private']['inventories']=[{'WHEAT':4}]
        value,_=rollout(end,cfg)
        terminal[f'{step}:{pos}']=value
        assert (value>3000)==(step==718 and pos==[4,4]),terminal
    env,raw=world()
    for turn in range(49):
        action={'farmer':['PASS'],'hands':[['PASS'] for _ in raw['farms'][0]['hands']],'market':[]}
        if turn==0: action['market']=[['BUY_SEED','WHEAT',3],['HIRE']]
        if turn==1: action['farmer']=['PLANT','WHEAT']
        if turn in (2,24,48):action['farmer']=['WATER']
        if turn==47:action['market']=[['BUY_PRODUCT','WHEAT',2]]
        forecast=_project_one_turn(raw,action,cfg)
        env.step([action,{'farmer':['PASS'],'hands':[],'market':[]}])
        observed=plain(env._Environment__get_shared_state(0)['observation'])
        for key in ('private','market','town','day','hour'):
            assert forecast[key]==observed[key],(turn,key,forecast[key],observed[key])
        assert forecast['farms'][0]==observed['farms'][0],turn
        raw=observed
    _,raw=world();raw['farms'][0]['hands']=[[3,4]];raw['private']['inventories']=[{},{}];raw['private']['seeds']['WHEAT']=1
    jobs=[Job(f'p{i}','plant_crop','production_start',{'tile':xy,'crop':'WHEAT'},100,0) for i,xy in enumerate(([4,4],[3,4]))]
    action,_=schedule(raw,cfg,jobs)
    assert sum(a[0]=='PLANT' for a in [action['farmer'],*action['hands']])==1,action
    sell=Job('s','sell','trade',{'item':'WHEAT','quantity':1},100,0,False,True)
    feed=Job('f','feed_animal','maintenance',{'tile':[0,0],'animal':'COW','day':0},100,0)
    action,_=schedule(raw,cfg,[sell,feed])
    assert ['SELL','WHEAT',1] in action['market'] and ['BUY_PRODUCT','WHEAT',1] in action['market'],action
    print(json.dumps({'seed_without_work':3000,'wheat_forecast':cash,'wheat_official':actual,'official_transitions_matched':49,'terminal':terminal,'seed_claims':'passed','market_pair':'passed'}))

if __name__=='__main__':main()
