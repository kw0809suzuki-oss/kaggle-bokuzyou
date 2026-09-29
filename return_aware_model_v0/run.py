"""Small paired Official World run; use --steps 719 for terminal comparison."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as rules

from strong_model_v0_reimplementation.agent import Runtime as StrongRuntime
from strong_model_v0_reimplementation.planner import settings_from
from .agent import Runtime


ROOT = Path(__file__).resolve().parents[1]


def _opponent():
    path = ROOT / "astra_flow_vendor" / "seyamalam_v21.py"
    spec = importlib.util.spec_from_file_location("return_aware_opponent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def _fingerprint(env):
    observations = [env._Environment__get_shared_state(i)["observation"] for i in (0, 1)]
    world = {k: observations[0][k] for k in ("farms", "market", "town", "day", "hour")}
    world["private"] = [o["private"] for o in observations]
    return hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()


def run_one(seed, seat, steps, extension, max_candidates=6):
    env = make("kaggriculture", configuration={"seed": seed}, debug=False)
    env.reset(2)
    cfg = settings_from(env.configuration)
    runtime = Runtime(cfg, max_candidates) if extension else StrongRuntime(cfg)
    timings = []

    def act(obs, configuration):
        start = perf_counter()
        action = runtime.act(obs)
        timings.append(perf_counter() - start)
        return action

    agents = [None, None]
    agents[seat], agents[1 - seat] = act, _opponent()
    runner = env._Environment__agent_runner(agents)
    records = []
    for step in range(steps):
        if env.done:
            break
        actions, logs = runner.act()
        env.step(actions, logs)
        statuses = [str(s.status) for s in env.state]
        if any(s not in ("ACTIVE", "DONE") for s in statuses):
            raise RuntimeError(f"Official World rejected step {step}: {statuses}; logs={logs}")
        decision = runtime.last_choice if extension else None
        records.append({
            "step": step,
            "self_action": actions[seat],
            "world_hash": _fingerprint(env),
            "self_cash": env.state[0].observation.farms[seat].money,
            "decision": decision,
        })
    terminal = env.done and all(str(s.status) == "DONE" for s in env.state)
    money = [float(f.money) for f in env.state[0].observation.farms]
    return {
        "model": "return_aware" if extension else "strong",
        "seed": seed, "seat": seat, "steps": len(records),
        "statuses": [str(s.status) for s in env.state],
        "terminal_reached": terminal,
        "observed_self_cash": money[seat],
        "terminal_self": money[seat] if terminal else None,
        "terminal_margin": money[seat] - money[1 - seat] if terminal else None,
        "mean_decision_seconds": sum(timings) / len(timings) if timings else 0,
        "max_decision_seconds": max(timings, default=0),
        "records": records,
    }


def run_pair(seed=92804001, seat=0, steps=24, max_candidates=6):
    if seat not in (0, 1) or not 1 <= steps <= 719 or max_candidates < 0:
        raise ValueError("seat must be 0/1, steps 1..719, budget nonnegative")
    baseline = run_one(seed, seat, steps, False)
    candidate = run_one(seed, seat, steps, True, max_candidates)
    paired = list(zip(baseline["records"], candidate["records"]))
    terminal = baseline["terminal_reached"] and candidate["terminal_reached"]
    summary = {
        "seed": seed, "seat": seat, "steps": steps,
        "generated_count": sum(r["decision"]["generated_count"] for r in candidate["records"]),
        "evaluated_count": sum(r["decision"]["evaluated_count"] for r in candidate["records"]),
        "selected_count": sum(r["decision"]["extension_selected"] for r in candidate["records"]),
        "first_action_difference": next((b["step"] for b, c in paired if b["self_action"] != c["self_action"]), None),
        "first_world_difference": next((b["step"] for b, c in paired if b["world_hash"] != c["world_hash"]), None),
        "terminal_reached": terminal,
        "terminal_self_delta": candidate["terminal_self"] - baseline["terminal_self"] if terminal else None,
        "terminal_margin_delta": candidate["terminal_margin"] - baseline["terminal_margin"] if terminal else None,
    }
    return {
        "schema": "return-aware-model-v0-paired-run",
        "model_sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__), Path(__file__).with_name("agent.py"),
                Path(__file__).with_name("model.py"),
                ROOT / "strong_model_v0_reimplementation" / "planner.py",
                ROOT / "strong_model_v0_reimplementation" / "jobs.py",
                ROOT / "astra_flow_vendor" / "seyamalam_v21.py",
            )
        },
        "world_source_sha256": hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest(),
        "opponent": "Seyamalam v21",
        "max_extra_candidates": max_candidates,
        "summary": summary,
        "boundary": "Short runs verify execution only; forecast cash is not observed terminal cash. No adoption claim.",
        "baseline": baseline,
        "candidate": candidate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=92804001)
    parser.add_argument("--seat", type=int, choices=(0, 1), default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("smoke_result.json"))
    args = parser.parse_args()
    result = run_pair(args.seed, args.seat, args.steps, args.max_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
