#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

from kaggle_environments import make

from seyamalam_teacher_body_v0 import make_teacher_body
from cash_return_wheat_v0 import make_candidate

ROOT = Path(__file__).resolve().parent
SEED = 92801101
TEACHER_COMMIT = "8b8c421eb10634c756583ce10c75189f50c83a72"
WHEAT_COMMIT = "5278b25325d119ed71372c137e418d4a17fb37b2"
OFFICIAL_COMMIT = "d7729da06cc1382eb742d6980dc3180aa85caa28"


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shared_obs(env, seat):
    return env._Environment__get_shared_state(seat)["observation"]


def state_signature(env):
    obs = env.state[0].observation
    return json.dumps(plain(obs), sort_keys=True, separators=(",", ":"))


def make_wheat():
    return make_candidate(
        crop="WHEAT",
        max_active_plants=1,
        harvest_age_days=4,
        season_days=30,
        turns_per_day=24,
        parallel_market=True,
    )


def main():
    raw_teacher = load_module(ROOT / "opponents" / "seyamalam_raw.py", "seyamalam_raw_parity")
    wrapped_module = load_module(ROOT / "opponents" / "seyamalam_wrapped.py", "seyamalam_wrapped_parity")
    wrapped_teacher = make_teacher_body(wrapped_module)

    raw_wheat = make_wheat()
    wrapped_wheat = make_wheat()

    env_raw = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env_wrapped = make("kaggriculture", configuration={"seed": SEED}, debug=False)
    env_raw.reset(num_agents=2)
    env_wrapped.reset(num_agents=2)

    action_mismatches = []
    state_mismatches = []

    turn = 0
    while not env_raw.done and not env_wrapped.done:
        if state_signature(env_raw) != state_signature(env_wrapped):
            state_mismatches.append(turn)
            break

        obs_raw_0 = shared_obs(env_raw, 0)
        obs_raw_1 = shared_obs(env_raw, 1)
        obs_wrap_0 = shared_obs(env_wrapped, 0)
        obs_wrap_1 = shared_obs(env_wrapped, 1)

        a_raw = plain(raw_teacher.agent(obs_raw_0))
        a_wrap = plain(wrapped_teacher.act(obs_wrap_0))
        b_raw = plain(raw_wheat.act(obs_raw_1))
        b_wrap = plain(wrapped_wheat.act(obs_wrap_1))

        if a_raw != a_wrap:
            action_mismatches.append({
                "turn": turn,
                "raw": a_raw,
                "wrapped": a_wrap,
            })
            break

        if b_raw != b_wrap:
            action_mismatches.append({
                "turn": turn,
                "side": "wheat_control",
                "raw": b_raw,
                "wrapped": b_wrap,
            })
            break

        env_raw.step([a_raw, b_raw])
        env_wrapped.step([a_wrap, b_wrap])
        turn += 1

    final_raw = env_raw.state[0].observation
    final_wrap = env_wrapped.state[0].observation

    result = {
        "schema": "seyamalam-teacher-wrapper-parity-v0",
        "seed": SEED,
        "provenance": {
            "teacher_commit": TEACHER_COMMIT,
            "wheat_commit": WHEAT_COMMIT,
            "official_commit": OFFICIAL_COMMIT,
        },
        "turns_compared": turn,
        "action_mismatch_count": len(action_mismatches),
        "state_mismatch_count": len(state_mismatches),
        "terminal": {
            "raw_teacher": float(final_raw["farms"][0]["money"]),
            "wrapped_teacher": float(final_wrap["farms"][0]["money"]),
            "raw_wheat": float(final_raw["farms"][1]["money"]),
            "wrapped_wheat": float(final_wrap["farms"][1]["money"]),
        },
        "exact_parity": (
            len(action_mismatches) == 0
            and len(state_mismatches) == 0
            and state_signature(env_raw) == state_signature(env_wrapped)
        ),
        "action_mismatches": action_mismatches,
        "state_mismatches": state_mismatches,
    }

    Path("seyamalam_teacher_wrapper_parity_seed92801101.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
