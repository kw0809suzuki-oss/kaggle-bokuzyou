"""One runtime-only Official World episode. Does not print or persist scores."""
from copy import deepcopy
import json
import time
from kaggle_environments import make
from astra_flow_terminal_model_v0 import AstraFlowAgent


def main():
    env = make('kaggriculture', configuration={'seed':7001}, debug=False)
    env.reset(num_agents=2)
    model = AstraFlowAgent()
    turns, maximum = 0, 0.0
    while not env.done:
        obs = env._Environment__get_shared_state(0)['observation']
        before = deepcopy(obs)
        start = time.perf_counter()
        action = model.act(obs)
        maximum = max(maximum, time.perf_counter()-start)
        assert obs == before, 'input mutation'
        assert len(action['hands']) == len(obs['farms'][0]['hands'])
        assert len(action['market']) <= model.cfg.maxMarketOrdersPerTurn
        env.step([action, {'farmer':['PASS'], 'hands':[], 'market':[]}])
        turns += 1
    assert turns == 719, turns
    assert all(s.status == 'DONE' for s in env.state)
    print(json.dumps({'runtime_only':True, 'turns':turns,
                      'statuses':[s.status for s in env.state],
                      'max_act_seconds':round(maximum,6),
                      'scores_not_evaluated':True}))


if __name__ == '__main__':
    main()
