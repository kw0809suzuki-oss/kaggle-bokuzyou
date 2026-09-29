#!/usr/bin/env python3
import json, os
import run_decem_first_source_divergence_v0 as base

def compact(obs):
    f = obs["farms"][0]
    p = obs["private"]
    return {
        "step": obs["step"],
        "day": obs["day"],
        "hour": obs["hour"],
        "money": f["money"],
        "farmer": f["farmer"],
        "hands": f["hands"],
        "hires_today": f["hires_today"],
        "unlocked_quadrants": f["unlocked_quadrants"],
        "shed": p["shed"],
        "seeds": p["seeds"],
        "inventories": p["inventories"],
    }

def main():
    seed = int(os.environ["SEED"])
    model = base.load(base.MODEL, f"m_pre24_{seed}")
    opp = base.load(base.OPP, f"o_pre24_{seed}")
    env = base.make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(num_agents=2)

    while not env.done:
        pre = base.plain(base.shared(env, 0))
        step = int(pre["step"])
        action = base.plain(model.agent(pre))
        if step == 24:
            expected = base.EXPECTED[24]
            out = {
                "schema": "decem-step24-prestate-probe-v0",
                "seed": seed,
                "actual_pre": compact(pre),
                "expected_pre": {
                    "step": expected["step"],
                    "day": expected["day"],
                    "hour": expected["hour"],
                    "money": expected["money"],
                    "farmer": expected["farmer"],
                    "hands": expected["hands"],
                    "hires_today": expected["hires"],
                    "unlocked_quadrants": expected["unlocked"],
                    "shed": expected["shed"],
                    "seeds": expected["seeds"],
                    "inventories": expected["inventories"],
                },
                "issued_action": action,
            }
            fn = f"decem_step24_prestate_seed{seed}.json"
            open(fn, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
            print("STEP24_PRESTATE " + json.dumps(out, ensure_ascii=False, separators=(",", ":")))
            return
        oa = base.plain(opp.agent(base.shared(env, 1)))
        env.step([action, oa])

if __name__ == "__main__":
    main()
