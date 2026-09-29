#!/usr/bin/env python3
"""Idle Recovery single-shot World Gate v1.

Baseline: exact Strong selection state.
Candidate: at the first turn where a recovery job can be added while preserving
every baseline-busy unit action and activating at least one baseline-PASS unit,
inject exactly one such recovery commitment. After that action, return to normal
Strong on the resulting World state.

This is a World gate only, not a terminal strength verdict.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import run_terminal_return_world_gate_v0 as gate

OUT=Path(__file__).with_name("idle_recovery_single_shot_world_gate_v1_result.json")


def _unit_actions(action: dict[str, Any]) -> list[Any]:
    return [action.get("farmer"), *(action.get("hands", []) or [])]


class IdleRecoverySingleShotRuntime:
    def __init__(self, cfg: Any):
        self.cfg=cfg
        self.active: dict[str, dict[str, Any]]={}
        self.last_step=-1
        self.intervened=False
        self.last_allocation: dict[str, Any] | None=None

    def reset(self):
        self.active={}
        self.last_step=-1
        self.intervened=False
        self.last_allocation=None

    def act(self, obs: Any) -> dict[str, Any]:
        snapshot=gate.bind_official_state(obs)
        raw=snapshot.raw()
        raw["step"]=gate.step_of(raw,self.cfg)
        step=int(raw["step"])
        if step <= self.last_step:
            self.reset()

        self.active={
            key:spec for key,spec in self.active.items()
            if gate.still_needed(spec,raw)
        }

        chosen,selection=gate._capture_selected_state(raw,self.cfg,self.active)
        baseline_action=copy.deepcopy(chosen.action)
        baseline_units=_unit_actions(baseline_action)
        base_specs=copy.deepcopy(chosen.commitments)
        base_keys={str(spec["key"]) for spec in base_specs}

        picked=None
        picked_action=None
        picked_scheduled=[]
        candidate_rows=[]

        if not self.intervened:
            recovery=[
                j for j in gate.operating_jobs(raw,self.cfg,harvest_now=True)
                if j.category=="recovery" and j.key not in base_keys
            ]
            for job in recovery:
                planned=list(selection["planned"])
                planned.append(job)
                forced=set(selection["forced"])
                forced.add(job.key)
                action,scheduled,_,_=gate._service_action(
                    raw,self.cfg,planned,
                    forced=forced or None,
                    harvest_now=selection["harvest_now"],
                )
                action=gate._apply_native_postprocess(action,selection["extra"])
                cand_units=_unit_actions(action)
                if len(cand_units)!=len(baseline_units):
                    continue
                busy_preserved=all(
                    b==["PASS"] or c==b
                    for b,c in zip(baseline_units,cand_units)
                )
                idle_activated=any(
                    b==["PASS"] and c!=["PASS"]
                    for b,c in zip(baseline_units,cand_units)
                )
                actually_scheduled=job.key in scheduled
                if busy_preserved and idle_activated and actually_scheduled:
                    candidate_rows.append((job,action,scheduled))

            if candidate_rows:
                picked,picked_action,picked_scheduled=max(
                    candidate_rows,
                    key=lambda row:(float(row[0].central_delta),row[0].key)
                )

        if picked is not None:
            action=copy.deepcopy(picked_action)
            self.intervened=True
            recovery_scheduled=True
        else:
            action=baseline_action
            recovery_scheduled=False

        # Never carry the probe-only job. Continue from Strong's own commitments.
        self.active={
            str(spec["key"]):copy.deepcopy(spec)
            for spec in base_specs
            if gate.still_needed(spec,raw)
        }
        self.last_allocation={
            "step":step,
            "selection_state_reconstructed":True,
            "selection_mode":selection["mode"],
            "matching_modes":selection["matching_modes"],
            "base_commitments":[s["key"] for s in base_specs],
            "baseline_action":baseline_action,
            "eligible_idle_recovery_count":len(candidate_rows),
            "added_recovery":None if picked is None else picked.spec(),
            "scheduled":list(picked_scheduled) if picked is not None else [],
            "recovery_scheduled":recovery_scheduled,
            "single_shot_already_fired":self.intervened,
        }
        self.last_step=step
        return action


def _run_variant() -> dict[str, Any]:
    opponent=gate._load_opponent()
    gate._reset(opponent)
    gate.strong_module.reset_agent()
    holder={"runtime":None}
    trace=[]

    def self_agent(obs,configuration):
        step=int(obs["step"])
        raw_obs=copy.deepcopy(obs)
        if step >= gate.GATE_ACTIONS:
            action=gate._pass_action(obs)
            if step==gate.GATE_ACTIONS:
                trace.append({
                    "step":step,
                    "phase":"gate_state",
                    "metrics":gate._metrics(raw_obs),
                    "world_signature":gate._world_signature(raw_obs),
                    "world_digest":gate._digest(gate._world_signature(raw_obs)),
                    "action":action,
                    "allocation":None,
                })
            return action
        cfg=gate.settings_from(configuration)
        rt=holder["runtime"]
        if rt is None or rt.cfg!=cfg:
            rt=IdleRecoverySingleShotRuntime(cfg)
            holder["runtime"]=rt
        action=rt.act(obs)
        trace.append({
            "step":step,
            "phase":"decision",
            "metrics":gate._metrics(raw_obs),
            "world_signature":gate._world_signature(raw_obs),
            "world_digest":gate._digest(gate._world_signature(raw_obs)),
            "action":copy.deepcopy(action),
            "allocation":copy.deepcopy(rt.last_allocation),
        })
        return action

    env=gate.make("kaggriculture",configuration={"seed":gate.SEED},debug=False)
    agents=[opponent.agent,opponent.agent]
    agents[gate.SEAT]=self_agent
    env.run(agents)
    gate_row=next(r for r in trace if r["step"]==gate.GATE_ACTIONS)
    return {
        "label":"idle_recovery_single_shot_first24",
        "seed":gate.SEED,
        "seat":gate.SEAT,
        "gate_actions":gate.GATE_ACTIONS,
        "gate_metrics":gate_row["metrics"],
        "trace":trace,
    }


def main():
    baseline=gate._run("baseline_strong_first24",variant=False)
    variant=_run_variant()
    comparison=gate._compare(baseline,variant)
    comparison["eligible_idle_recovery_steps"]=[
        int(r["step"]) for r in variant["trace"]
        if r.get("phase")=="decision"
        and int(r.get("allocation",{}).get("eligible_idle_recovery_count",0))>0
    ]
    result={
        "schema":"idle-recovery-single-shot-world-gate-v1",
        "conditions":{
            "seed":gate.SEED,
            "seat":gate.SEAT,
            "opponent":"Seyamalam pinned v21",
            "official_world_commit":"d7729da06cc1382eb742d6980dc3180aa85caa28",
            "baseline":"Strong Model v0 exact selected Action",
            "variant":"one recovery allocation only when all baseline-busy unit actions are preserved and at least one baseline-PASS unit is activated; then return to normal Strong",
        },
        "comparison":comparison,
        "baseline":baseline,
        "variant":variant,
        "boundary":{
            "world_gate_only":True,
            "not_terminal_strength_test":True,
            "single_seed_single_seat":True,
        },
    }
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+"\n",encoding="utf-8")
    print(json.dumps({
        "world_gate_pass":comparison["world_gate_pass"],
        "intervention_count":comparison["intervention_count"],
        "scheduled_intervention_count":comparison["scheduled_intervention_count"],
        "first_action_divergence_step":comparison["first_action_divergence_step"],
        "first_world_divergence_step":comparison["first_world_divergence_step"],
        "eligible_idle_recovery_steps":comparison["eligible_idle_recovery_steps"],
        "gate_metric_delta":comparison["gate_metric_delta_variant_minus_baseline"],
    },ensure_ascii=False))


if __name__=="__main__":
    main()
