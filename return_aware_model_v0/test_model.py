"""Behavior contracts against the pinned Official World and existing Strong."""
import copy
import random
import unittest
from unittest.mock import patch

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from run_strong_model_v0_reimplementation_contract import plain
from strong_model_v0_reimplementation.jobs import Job
from strong_model_v0_reimplementation.agent import Runtime as StrongRuntime
from strong_model_v0_reimplementation.planner import Settings, _service_action, choose as strong_choose, rollout
from return_aware_model_v0.model import generate_candidates, choose
from return_aware_model_v0.agent import Runtime


def state(step=714, seat=0):
    env = make("kaggriculture", configuration={"seed": 92804001}, debug=False)
    env.reset(2)
    raw = plain(env._Environment__get_shared_state(seat)["observation"])
    raw.update(step=step, day=step // 24, hour=step % 24)
    farm = raw["farms"][seat]
    farm["farmer"] = [4, 4]
    farm["hands"] = []
    raw["private"]["inventories"] = [{"MILK": 3}]
    return raw


class ModelTests(unittest.TestCase):
    def test_carry_is_planned_before_action_and_preserves_other_commitments(self):
        raw = state()
        job = Job("animal:SHEEP", "establish_animal", "production_start",
                  {"tile": [4, 4], "animal": "SHEEP"}, 10000)
        before = copy.deepcopy(raw)
        rng = random.getstate()
        rows = generate_candidates(raw, Settings(), [job], max_candidates=20)
        carry = next(c for c in rows if c.family == "carry_to_shed")
        self.assertEqual(carry.action["farmer"], ["DROP"])
        self.assertIn(["SELL", "MILK", 3], carry.action["market"])
        self.assertIn(job.key, [j.key for j in carry.commitments])
        self.assertEqual(raw, before)
        self.assertEqual(random.getstate(), rng)
        self.assertFalse(job.active)

    def test_hold_investment_keeps_maintenance(self):
        raw = state()
        tile = rules._new_plant("MELON", 20, 24)
        raw["farms"][0]["tiles"][0][0] = tile
        water = Job("water:(0, 0)", "water", "maintenance",
                    {"tile": [0, 0], "crop": "MELON"}, 1000)
        buy = Job("new:COW", "establish_animal", "production_start",
                  {"tile": [4, 4], "animal": "COW"}, 2000)
        rows = generate_candidates(raw, Settings(), [water, buy])
        hold = next(c for c in rows if c.family == "hold_investment")
        self.assertIn(water.key, [j.key for j in hold.commitments])
        self.assertNotIn(buy.key, [j.key for j in hold.commitments])
        self.assertFalse(any(o[0] in ("BUY_ANIMAL", "BUY_SEED", "BUY_LAND")
                             for o in hold.action["market"]))

    def test_location_changes_cash_recovery_at_deadline(self):
        cfg = Settings()
        shed = state(718)
        shed["private"]["inventories"] = [{}]
        shed["private"]["shed"]["MILK"] = 3
        cow = copy.deepcopy(shed)
        cow["private"]["shed"]["MILK"] = 0
        animal = rules._new_animal("COW", 0)
        animal["yield_units"] = 3
        cow["farms"][0]["tiles"][4][4] = animal
        shed_cash, _ = rollout(shed, cfg)
        cow_cash, _ = rollout(cow, cfg)
        self.assertGreater(shed_cash, cow_cash)

    def test_disabled_extension_matches_strong_exactly(self):
        raw = state()
        expected, _, _, _ = strong_choose(copy.deepcopy(raw), Settings(), {})
        result = choose(raw, Settings(), {}, max_candidates=0)
        self.assertEqual(result.bundle.action, expected.action)
        self.assertEqual(result.bundle.commitments, expected.commitments)
        self.assertEqual(result.bundle.envelope, expected.envelope)
        self.assertEqual(result.report["chosen_family"], "strong")

    def test_selection_keeps_baseline_and_does_not_mutate_observation(self):
        raw = state()
        before = copy.deepcopy(raw)
        result = choose(raw, Settings(), {})
        self.assertGreaterEqual(result.bundle.envelope.central_cash,
                                result.report["baseline_terminal_cash"])
        self.assertEqual(raw, before)
        self.assertIn("generated_count", result.report)
        self.assertIn("action_changed", result.report)

    def test_deadline_carry_candidate_is_selected_over_unrecoverable_harvest(self):
        raw = state(717)
        raw["private"]["inventories"] = [{"MILK": 1}]
        melon = rules._new_plant("MELON", 17, 24)
        melon.update(yield_units=6, watered_today=True)
        raw["farms"][0]["tiles"][3][4] = melon
        result = choose(raw, Settings(), {})
        self.assertEqual(result.report["chosen_family"], "carry_to_shed")
        self.assertTrue(result.report["action_changed"])
        self.assertEqual(result.bundle.action["farmer"], ["DROP"])
        self.assertGreater(result.report["chosen_terminal_cash"],
                           result.report["baseline_terminal_cash"])

    def test_selected_carry_increases_official_terminal_cash_in_constructed_state(self):
        totals = []
        for runtime_type in (StrongRuntime, Runtime):
            env = make("kaggriculture", configuration={"seed": 92804001}, debug=False)
            env.reset(2)
            raw = state(717)
            raw["private"]["inventories"] = [{"MILK": 1}]
            melon = rules._new_plant("MELON", 17, 24)
            melon.update(yield_units=6, watered_today=True)
            raw["farms"][0]["tiles"][3][4] = melon
            env.state[0].observation.update(raw)
            # Synthetic deadline fixture, not a replay or evidence of strength.
            env.steps = [env.state] * 718
            runtime = runtime_type(Settings())
            for _ in range(2):
                obs = env._Environment__get_shared_state(0)["observation"]
                env.step([runtime.act(obs), {"farmer": ["PASS"], "hands": [], "market": []}])
            self.assertTrue(env.done)
            self.assertEqual([s.status for s in env.state], ["DONE", "DONE"])
            totals.append(env.state[0].observation.farms[0].money)
        self.assertGreater(totals[1], totals[0])

    def test_more_time_preserves_higher_value_harvest_over_immediate_cash(self):
        raw = state(715)
        raw["private"]["inventories"] = [{"MILK": 1}]
        melon = rules._new_plant("MELON", 17, 24)
        melon.update(yield_units=6, watered_today=True)
        raw["farms"][0]["tiles"][3][4] = melon
        result = choose(raw, Settings(), {})
        carry = next(c for c in result.report["candidates"] if c["family"] == "carry_to_shed")
        self.assertGreater(result.report["chosen_terminal_cash"], carry["terminal_cash"])
        self.assertNotEqual(result.bundle.action["farmer"], ["DROP"])

    def test_reconstruction_preserves_active_work_priority(self):
        raw = state(715)
        raw["private"]["inventories"] = [{}]
        for x, y in ((3, 4), (4, 3)):
            animal = rules._new_animal("COW", 0)
            animal["yield_units"] = 1
            raw["farms"][0]["tiles"][y][x] = animal
        active = Job("animal_harvest:(3, 4)", "harvest_animal", "recovery",
                     {"tile": [3, 4], "animal": "COW", "day": 29}, 160, active=True)
        with patch("return_aware_model_v0.model.generate_candidates", wraps=generate_candidates) as observed:
            choose(raw, Settings(), {active.key: active.spec()})
        # Observe the real boundary without replacing generation or scoring.
        planned = observed.call_args.args[2]
        action, _, _, _ = _service_action(raw, Settings(), planned)
        self.assertEqual(action["farmer"], ["WEST"])
        rows = generate_candidates(raw, Settings(), planned)
        same_work = next(c for c in rows if active.key in c.forced)
        retained = next(j for j in same_work.commitments if j.key == active.key)
        self.assertTrue(retained.active)

    def test_budget_and_terminal_boundary(self):
        self.assertEqual(generate_candidates(state(), Settings(), [], max_candidates=0), [])
        self.assertEqual(generate_candidates(state(719), Settings(), []), [])
        with self.assertRaises(ValueError):
            generate_candidates(state(), Settings(), [], max_candidates=-1)

    def test_runtime_reentry_and_seat_do_not_share_state(self):
        cfg = Settings()
        first = Runtime(cfg, max_candidates=0)
        second = Runtime(cfg, max_candidates=0)
        raw = state(718, 1)
        action = first.act(raw)
        self.assertEqual(action, first.act(raw))
        self.assertEqual(second.active, {})
        self.assertEqual(first.last_choice["step"], 718)


if __name__ == "__main__":
    unittest.main()
