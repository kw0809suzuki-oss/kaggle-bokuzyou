"""Correctness checks only. No score comparison or selection of parameters."""
from copy import deepcopy
import unittest
from kaggle_environments import make
from astra_flow_terminal_model_v0 import AstraFlowAgent, demand, Settings


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.env = make('kaggriculture', configuration={'seed': 7001}, debug=False)
        self.env.reset(num_agents=2)
        self.obs = deepcopy(self.env._Environment__get_shared_state(0)['observation'])
        self.model = AstraFlowAgent()

    def test_input_isolation_and_reset(self):
        original = deepcopy(self.obs)
        first = self.model.act(self.obs)
        self.assertEqual(self.obs, original)
        self.model.reset()
        self.assertEqual(first, self.model.act(self.obs))
        other = AstraFlowAgent()
        self.assertIsNot(other.teacher, self.model.teacher)
        self.assertIsNot(other.teacher._V18_SELECTED_MARKET, self.model.teacher._V18_SELECTED_MARKET)

    def test_quantity_cash_curve_and_terminal(self):
        obs = self.obs
        obs['town']['unlocked_shops'] = []
        obs['step'] = 1  # no town event inside the one-step horizon
        self.assertEqual(self.model._quantity(obs, 'WOOL', 10, 0, 2), 10)
        obs['town']['unlocked_shops'] = ['YARN_STORE'] * 5
        self.assertLess(self.model._quantity(obs, 'WOOL', 10, 0, 100), 10)
        self.assertEqual(self.model._quantity(obs, 'WOOL', 10, 7, 1), 10)
        self.assertGreaterEqual(self.model._quantity(obs, 'WOOL', 10, 7, 100), 7)
        obs['market']['inventory']['WOOL'] = 20000
        self.assertIn(self.model._quantity(obs, 'WOOL', 100, 0, 100), range(101))

    def test_terminal_drop_then_sell_and_no_purchase(self):
        obs = self.obs
        obs['step'], obs['day'], obs['hour'] = 718, 29, 22
        obs['farms'][0]['farmer'] = [4, 4]
        obs['private']['inventories'][0] = {'WOOL': 3}
        self.model.teacher._V18S_BASE_AGENT = lambda obs: {
            'farmer': ['PASS'], 'hands': [], 'market': [['BUY_SEED','WHEAT',1]]}
        action = self.model.act(obs)
        self.assertEqual(action['farmer'], ['DROP'])
        self.assertIn(['SELL', 'WOOL', 3], action['market'])
        self.assertTrue(all(o[0] == 'SELL' for o in action['market']))

    def test_funding_sales_and_impossible_investment(self):
        obs = self.obs
        obs['private']['shed']['WOOL'] = 10
        obs['town']['unlocked_shops'] = ['YARN_STORE'] * 5
        self.model.teacher._V18S_BASE_AGENT = lambda obs: {
            'farmer':['PASS'], 'hands':[],
            'market':[['SELL','WOOL',7], ['BUY_SEED','WHEAT',1]]}
        action = self.model.act(obs)
        self.assertGreaterEqual(next(o[2] for o in action['market'] if o[:2] == ['SELL','WOOL']), 7)
        obs['step'],obs['day'],obs['hour'] = 717,29,21
        action = self.model.act(obs)
        self.assertNotIn(['BUY_SEED','WHEAT',1], action['market'])

    def test_known_demand_exact_event_boundaries(self):
        self.obs['town']['unlocked_shops'] = ['YARN_STORE', 'YARN_STORE']
        self.assertEqual(demand(self.obs, 0, 1, 'WOOL', Settings()), 5)
        self.assertEqual(demand(self.obs, 1, 4, 'WOOL', Settings()), 0)
        self.assertEqual(demand(self.obs, 4, 5, 'WOOL', Settings()), 4)


if __name__ == '__main__':
    unittest.main()
