"""Minimal contract for the diagnostic labor+assets+execution bundle."""
import unittest
from kaggle_environments import make
from run_strong_model_v0_reimplementation_contract import plain
from strong_model_v0_bundle_probe import build_diagnostic_bundle

class BundleProbeContract(unittest.TestCase):
    def test_bundle_is_one_alternative_and_preserves_existing_selection(self):
        env=make('kaggriculture',configuration={'seed':92804001},debug=False)
        env.reset(2)
        raw=plain(env._Environment__get_shared_state(0)['observation'])
        bundle=build_diagnostic_bundle(raw)
        self.assertEqual(bundle['label'],'labor_assets_execution_diagnostic')
        self.assertEqual(len(bundle['commitments']),13)
        self.assertEqual(bundle['action']['market'],[
            ['HIRE'],['HIRE'],['BUY_ANIMAL','COW',1],
            ['BUY_SEED','MELON',6],['BUY_SEED','WHEAT',6],
        ])
        self.assertEqual(bundle['action']['farmer'],['BUILD_PASTURE'])
        self.assertEqual(bundle['action']['hands'],[])
        tiles=[tuple(j['target']['tile']) for j in bundle['commitments'] if 'tile' in j['target']]
        self.assertEqual(len(tiles),13)
        self.assertEqual(len(set(tiles)),13)
        self.assertGreaterEqual(len(bundle['alternatives']),1)
        self.assertIsNotNone(bundle['current_choice'])

if __name__=='__main__':unittest.main()
