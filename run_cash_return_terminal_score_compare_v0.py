#!/usr/bin/env python3
from pathlib import Path
import importlib.util
from kaggle_environments import make

from cash_return_wheat_v0 import make_candidate
from plan_generator_entrance_v0 import bind_official_state

ROOT = Path(__file__).resolve().parent
SEED = 92801101


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    raise TypeError(type(x).__name__)


def load_opponent():
    path = ROOT / "opponents" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location("seyamalam_v21_score_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def main():
    candidate = make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )
    opponent = load_opponent()

    env = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        a0 = plain(candidate.act(shared_obs(env, 0)))
        a1 = plain(opponent.agent(shared_obs(env, 1)))
        env.step([a0, a1])

    raw = bind_official_state(env.state[0].observation).raw()
    self_terminal = float(raw["farms"][0]["money"])
    opponent_terminal = float(raw["farms"][1]["money"])

    print({
        "seed": SEED,
        "candidate_code_commit": "5278b25325d119ed71372c137e418d4a17fb37b2",
        "self_terminal": self_terminal,
        "opponent_terminal": opponent_terminal,
        "margin": self_terminal - opponent_terminal,
        "opponent_over_self": (opponent_terminal / self_terminal) if self_terminal else None,
    })


if __name__ == "__main__":
    main()
