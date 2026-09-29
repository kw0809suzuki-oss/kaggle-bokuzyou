#!/usr/bin/env python3
import json
from pathlib import Path
import requests
from kaggle_environments import make

EPISODE_ID = 115303987
SOURCE_SEED = 1305581493
URL = f"https://www.kaggle.com/competitions/episodes/{EPISODE_ID}/replay.json"

def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "items"):
        return {str(k): plain(v) for k, v in x.items()}
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [plain(v) for v in x]
    return x

def parse_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def obs_from(step_row, seat):
    o = parse_obj(step_row[seat].get("observation"))
    return plain(o)

def action_from(step_row, seat):
    return plain(parse_obj(step_row[seat].get("action")))

def canonical_obs(o):
    o = plain(o)
    return {
        "step": o.get("step"),
        "day": o.get("day"),
        "hour": o.get("hour"),
        "farms": o.get("farms"),
        "private": o.get("private"),
        "market": o.get("market"),
        "town": o.get("town"),
    }

def diff_top(a, b):
    return [k for k in ("step","day","hour","farms","private","market","town") if a.get(k) != b.get(k)]

def main():
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    replay = r.json()
    if isinstance(replay, dict) and isinstance(replay.get("replay"), str):
        replay = json.loads(replay["replay"])

    steps = replay["steps"]
    config = replay.get("configuration") if isinstance(replay, dict) else None
    env_config = {}
    if isinstance(config, dict):
        # Reuse source configuration where possible, but seed is authoritative.
        env_config.update(config)
    env_config["seed"] = SOURCE_SEED

    env = make("kaggriculture", configuration=env_config, debug=False)
    env.reset(num_agents=2)

    initial_actual = canonical_obs(env._Environment__get_shared_state(0)["observation"])
    initial_source = canonical_obs(obs_from(steps[0], 0))
    first_div = None
    action_count = 0

    for i in range(len(steps) - 1):
        # Kaggle replay stores the action that produced state i+1 on steps[i+1].
        a0 = action_from(steps[i + 1], 0)
        a1 = action_from(steps[i + 1], 1)
        env.step([a0, a1])
        action_count += 1

        actual = canonical_obs(env._Environment__get_shared_state(0)["observation"]) if not env.done else canonical_obs(env.state[0].observation)
        source = canonical_obs(obs_from(steps[i + 1], 0))
        ds = diff_top(actual, source)
        if ds and first_div is None:
            first_div = {
                "issued_turn": i,
                "observed_source_step_index": i + 1,
                "diff_fields": ds,
                "actions": [a0, a1],
                "actual": actual,
                "source": source,
            }

    final0 = canonical_obs(env.state[0].observation)
    final_source0 = canonical_obs(obs_from(steps[-1], 0))
    source_terminal_self = float(final_source0["farms"][0]["money"])
    reproduced_terminal_self = float(final0["farms"][0]["money"])

    out = {
        "schema": "decem-source-replay-reproduction-v0",
        "episode_id": EPISODE_ID,
        "source_seed": SOURCE_SEED,
        "source_num_states": len(steps),
        "replayed_action_count": action_count,
        "source_configuration": config,
        "initial_diff_fields": diff_top(initial_actual, initial_source),
        "first_divergence": first_div,
        "source_terminal_self": source_terminal_self,
        "reproduced_terminal_self": reproduced_terminal_self,
        "terminal_delta": reproduced_terminal_self - source_terminal_self,
        "final_diff_fields": diff_top(final0, final_source0),
    }
    Path("decem_source_replay_reproduction_v0.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SOURCE_REPLAY_REPRODUCTION " + json.dumps(out, separators=(",", ":")))

if __name__ == "__main__":
    main()
