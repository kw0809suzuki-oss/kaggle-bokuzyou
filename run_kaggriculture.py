from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture


def pass_agent(obs):
    hands = obs.get("private", {}).get("farmHands", [])
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }


def main():
    print("official_world:", kaggriculture.__file__)
    env = make("kaggriculture", debug=True)
    env.run([pass_agent, pass_agent])
    rewards = [state.reward for state in env.state]
    print("terminal_rewards:", rewards)
    print("steps:", len(env.steps))

    if any(r is None for r in rewards):
        raise RuntimeError("Kaggriculture did not reach terminal rewards")


if __name__ == "__main__":
    main()
