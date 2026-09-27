"""Pre-registered Fresh20 paired Official World screening; no mechanism probe."""
import copy
import hashlib
import json
import statistics
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import kaggle_environments
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official

import relationship_surface_body_v0 as frozen
from minimal_terminal_candidate_v0 import make_candidate
from plan_generator_entrance_v0 import bind_official_state, generate_plans
from run_exclude_confirmed_blocked_ab_v0 import load_opponent, OPP_PATH

ROOT = Path(__file__).resolve().parent
SPEC = json.loads((ROOT / "minimal_terminal_candidate_v0_spec.json").read_text())


def run_pair(seed):
    baseline = frozen.RelationshipSurfaceBody(policy_seed=SPEC["policy_seed"])
    candidate = make_candidate()
    agents = [baseline, candidate]
    opponents = [load_opponent(f"minimal_{seed}_{arm}") for arm in range(2)]
    envs = [make("kaggriculture", configuration={"seed": seed}, debug=False)
            for _ in range(2)]
    for env in envs:
        env.reset(num_agents=2)
    assert envs[0].configuration == envs[1].configuration
    turns = 0
    replacement_checks = None
    next_call_checked = False
    while not all(env.done for env in envs):
        assert not any(env.done for env in envs), "paired episode lengths differ"
        self_obs = [env._Environment__get_shared_state(0)["observation"] for env in envs]
        opp_obs = [env._Environment__get_shared_state(1)["observation"] for env in envs]
        was_intervened = candidate.intervened
        if not was_intervened:
            assert bind_official_state(self_obs[0]).canonical_hash == bind_official_state(self_obs[1]).canonical_hash
        # Exactly one next-call parity check proves control returned to ordinary Body.
        reference = copy.deepcopy(candidate.body) if was_intervened and not next_call_checked else None
        actions = [agent.act(obs) for agent, obs in zip(agents, self_obs)]
        if reference is not None:
            assert actions[1] == reference.act(self_obs[1])
            assert candidate.body.rng.getstate() == reference.rng.getstate()
            assert candidate.body.active == reference.active
            assert candidate.body.active_steps == reference.active_steps
            next_call_checked = True
        if not was_intervened:
            assert baseline.rng.getstate() == candidate.body.rng.getstate()
            if candidate.intervened:
                log = candidate.intervention_log
                assert log["baseline_projected_action"] == actions[0]
                assert log["lens_projected_action"] == actions[1]
                assert actions[0] != actions[1]
                pre = bind_official_state(self_obs[1])
                plans = generate_plans(pre)
                selectable = [p for p in plans if not frozen._is_blocked(pre, p)]
                by_id = {p.candidate_id: p for p in selectable}
                assert set(by_id) == set(log["selectable_candidate_ids"])
                assert log["baseline_selected_candidate"] in by_id
                replacement = by_id[log["lens_selected_candidate"]]
                assert replacement.kind == "realize_shed_stock_sale"
                assert not frozen._is_blocked(pre, replacement)
                raw = pre.raw()
                item = replacement.target["item"]
                qty = replacement.target["available_quantity"]
                assert item in official.PRODUCTS and 0 < qty <= raw["private"]["shed"][item]
                assert log["body_rng_unchanged"]
                replacement_checks = {"same_selectable_set": True, "not_blocked": True,
                                      "official_sell_item_and_stock_valid": True,
                                      "action_changed": True, "body_rng_unchanged": True}
            else:
                assert actions[0] == actions[1]
        opponent_actions = [frozen._plain(op.agent(obs)) for op, obs in zip(opponents, opp_obs)]
        if not was_intervened:
            assert opponent_actions[0] == opponent_actions[1]
        for env, action, opponent_action in zip(envs, actions, opponent_actions):
            env.step([frozen._plain(action), opponent_action])
            assert all(s.status in ("ACTIVE", "DONE") for s in env.state)
        candidate.observe_post(envs[1].state[0].observation)
        assert candidate.intervention_count <= 1
        turns += 1
    assert all(all(s.status == "DONE" for s in env.state) for env in envs)
    assert turns == int(envs[0].configuration.episodeSteps) - 1
    terminals = []
    for env in envs:
        raw = bind_official_state(env.state[0].observation).raw()
        money = [float(f["money"]) for f in raw["farms"]]
        assert all(s.reward is not None for s in env.state)
        terminals.append({"terminal_self": money[0], "terminal_opponent": money[1],
                          "margin": money[0] - money[1]})
    assert not candidate.intervened or next_call_checked or candidate.intervention_log["turn"] == turns - 1
    return {"seed": seed, "self_seat": 0, "policy_seed": SPEC["policy_seed"],
            "turns": turns, "baseline": terminals[0], "candidate": terminals[1],
            "delta_terminal_self": terminals[1]["terminal_self"] - terminals[0]["terminal_self"],
            "intervention": candidate.summary(), "replacement_checks": replacement_checks,
            "next_call_frozen_body_parity": next_call_checked,
            "ran_to_terminal": True}


def main():
    # All tracked files in the base must remain byte-identical. Only additions are allowed.
    base = SPEC["base_commit"]
    subprocess.run(["git", "diff", "--exit-code", base, "--", *subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", base], text=True).splitlines()], check=True)
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(run_pair, SPEC["environment_seeds"]):
            rows.append(row)
            print("PAIR " + json.dumps({"seed": row["seed"], "delta": row["delta_terminal_self"],
                  "intervened": row["intervention"]["intervention_count"]}), flush=True)
    ds = [r["delta_terminal_self"] for r in rows]
    mean = statistics.fmean(ds)
    result = {"fixed_before_results": SPEC,
              "provenance": {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                             "official_version": kaggle_environments.__version__,
                             "official_module_sha256": hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest(),
                             "opponent_sha256": hashlib.sha256(OPP_PATH.read_bytes()).hexdigest()},
              "aggregate": {"intervention_reached": sum(r["intervention"]["intervention_count"] for r in rows),
                            "improved": sum(d > 0 for d in ds), "worse": sum(d < 0 for d in ds),
                            "same": sum(d == 0 for d in ds), "mean_delta_terminal_self": mean,
                            "median_delta_terminal_self": statistics.median(ds),
                            "min": min(ds), "max": max(ds),
                            "decision": "SURVIVE" if mean > 0 else "DISCARD"},
              "integrity": {"frozen_body_unchanged": True, "max_one_intervention": True,
                            "paired_conditions_match": True, "all_terminal": True},
              "per_seed": rows}
    (ROOT / "minimal_terminal_candidate_v0_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SUMMARY " + json.dumps(result["aggregate"]), flush=True)


if __name__ == "__main__":
    main()
