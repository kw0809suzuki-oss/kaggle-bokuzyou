#!/usr/bin/env python3
"""Adaptive Replay Runtime v0.1 release gate.

No performance exploration. This gate checks:
  1) one active Contract only;
  2) release behavior files are byte-identical to the frozen v0 blobs recorded
     by Git blob SHA in the release manifest;
  3) Effect Observer is non-intervening on a representative record;
  4) existing fixed10 exact-equivalence gate still passes.
"""
from __future__ import annotations
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"adaptive_replay_runtime_v0_1_evidence_manifest.json"
CONTRACTS=ROOT/"adaptive_replay_effect_contracts_v0.json"
OBSERVER=ROOT/"adaptive_replay_effect_observer_v0.py"

EXPECTED_GIT_BLOB_SHA={
    "adaptive_replay_contract_runtime_v0.py":"c94afd2b833cf2b8a6134b8a64ac6aff5da9cfcb",
    "adaptive_replay_effect_contracts_v0.json":"3d612360de4f207142935ba74da4a0396ccc4007",
    "adaptive_replay_world_adapter_v0.py":"b627bc763eba11c5d8f199f5011923b4cbba88a0",
}


def git_blob_sha(path: Path) -> str:
    data=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    contracts=json.loads(CONTRACTS.read_text(encoding="utf-8"))
    active=[c for c in contracts.get("contracts",[]) if c.get("status")=="active"]
    if len(active)!=1 or active[0].get("id")!="step24-hire-realized-plus1-v0":
        raise SystemExit("release gate failed: active Contract set changed")

    hashes={}
    for rel,expected in EXPECTED_GIT_BLOB_SHA.items():
        actual=git_blob_sha(ROOT/rel)
        hashes[rel]=actual
        if actual!=expected:
            raise SystemExit(f"release gate failed: frozen behavior file changed: {rel}")

    observer=load(OBSERVER,"_release_effect_observer")
    requested={"farm":[["PASS"]],"market":[["HIRE"],["HIRE"],["HIRE"]]}
    emitted={"farm":[["PASS"]],"market":[["HIRE"]]}
    obs={"step":24,"player":0,"farms":[{"money":6,"hires_today":0,"hands":[]}]}
    rec=observer.observe_runtime_step(
        obs=obs,
        requested_action=requested,
        emitted_action=emitted,
        contract_id="step24-hire-realized-plus1-v0",
        contract_event={"current_feasible_hires":3},
    )
    if requested["market"]!=[["HIRE"],["HIRE"],["HIRE"]] or emitted["market"]!=[["HIRE"]]:
        raise SystemExit("release gate failed: Observer mutated Action")
    if not rec["action_changed"]:
        raise SystemExit("release gate failed: Observer record incorrect")

    proc=subprocess.run(
        [sys.executable,str(ROOT/"run_adaptive_replay_contract_equivalence_fixed10.py")],
        cwd=ROOT,text=True,capture_output=True
    )
    if proc.returncode!=0:
        print(proc.stdout)
        print(proc.stderr,file=sys.stderr)
        raise SystemExit("release gate failed: fixed10 equivalence")

    summary_line=next(
        (line for line in proc.stdout.splitlines() if line.startswith("CONTRACT_EQUIVALENCE_SUMMARY ")),
        None,
    )
    if summary_line is None:
        raise SystemExit("release gate failed: missing equivalence summary")
    eq=json.loads(summary_line.split(" ",1)[1])
    if eq.get("equivalent")!=10 or eq.get("not_equivalent")!=0:
        raise SystemExit("release gate failed: not 10/10 exact")

    out={
        "schema":"adaptive-replay-runtime-v0.1-release-gate-result",
        "release":manifest["release"],
        "active_contract_count":len(active),
        "active_contract_id":active[0]["id"],
        "frozen_behavior_git_blob_sha":hashes,
        "observer_non_intervening":True,
        "fixed10_equivalence":eq,
        "accepted":True,
    }
    Path("adaptive_replay_runtime_v0_1_release_gate_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print("ADAPTIVE_REPLAY_RUNTIME_V0_1_RELEASE_GATE "+json.dumps(out,separators=(",",":")))


if __name__=="__main__":
    main()
