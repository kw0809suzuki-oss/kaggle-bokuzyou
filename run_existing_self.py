import sys
from pathlib import Path

from kaggle_environments import make

SELF_ROOT = Path(__file__).resolve().parent / "selfsrc"
sys.path.insert(0, str(SELF_ROOT))

import whole_flow_control_agent as existing_self


def pass_agent(obs):
    hands = obs.get("private", {}).get("farmHands", [])
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }


def main():
    print("self_module:", existing_self.__file__)
    env = make("kaggriculture", debug=True)
    env.run([existing_self.agent, pass_agent])

    rewards = [state.reward for state in env.state]
    print("terminal_rewards:", rewards)
    print("steps:", len(env.steps))

    if any(r is None for r in rewards):
        raise RuntimeError("Existing Self did not reach terminal rewards")


if __name__ == "__main__":
    main()
