import sys
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent
SELF_ROOT = ROOT / "selfsrc"
OPP_ROOT = ROOT / "opponents"
sys.path.insert(0, str(SELF_ROOT))
sys.path.insert(0, str(OPP_ROOT))

import whole_flow_control_agent as existing_self
import seyamalam_v21 as opponent


def main():
    seed = 7001
    env = make("kaggriculture", configuration={"seed": seed}, debug=True)
    env.run([existing_self.agent, opponent.agent])

    rewards = [float(state.reward) for state in env.state]
    print("seed:", seed)
    print("self_module:", existing_self.__file__)
    print("opponent_module:", opponent.__file__)
    print("terminal_rewards:", rewards)
    print("self:", rewards[0])
    print("opponent:", rewards[1])
    print("margin:", rewards[0] - rewards[1])
    print("steps:", len(env.steps))

    if any(r is None for r in rewards):
        raise RuntimeError("Battle did not reach terminal rewards")


if __name__ == "__main__":
    main()
