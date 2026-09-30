import copy
import unittest

import adaptive_replay_contract_runtime_v0 as model


class StandingOnReturnTests(unittest.TestCase):
    def setUp(self):
        model.reset_agent()

    def test_worker_standing_on_mature_crop_harvests_before_replay_movement(self):
        original = {
            "farmer": ["NORTH"],
            "hands": [["PASS"]],
            "market": [["SELL", "MILK", 1]],
        }
        model.replay.agent = lambda obs, configuration=None: copy.deepcopy(original)

        tiles = [[None for _ in range(5)] for _ in range(5)]
        tiles[1][0] = {
            "kind": "PLANT",
            "crop": "MELON",
            "yield_units": 6,
            "watered_today": True,
            "planted_day": 0,
        }
        obs = {
            "step": 300,
            "player": 0,
            "farms": [{
                "money": 1000,
                "hires_today": 0,
                "farmer": [0, 1],
                "hands": [[4, 4]],
                "tiles": tiles,
            }],
        }

        action = model.agent(obs)

        self.assertEqual(action["farmer"], ["HARVEST"])
        self.assertEqual(action["hands"], original["hands"])
        self.assertEqual(action["market"], original["market"])


if __name__ == "__main__":
    unittest.main()
