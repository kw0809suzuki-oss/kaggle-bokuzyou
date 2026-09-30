# Adaptive Replay Runtime v0.1

## 目的

**強いReplayを作り直すのではなく、Replayで既に確認された強さをCurrent World上で壊さず成立させるための最小Runtime。**

```
Frozen Replay Skeleton
  -> Active Effect Contract
  -> Minimal Current-World Check
  -> Proven Minimal Transform
  -> Action
```

v0.1は新しいPolicyではない。既存のContract Runtime v0の挙動を凍結したまま、Contract・World Adapter・Evidence・観測境界を成果物として整理したもの。

## 現在のActive Contract

Active Contractは1件だけ。

- `step24-hire-realized-plus1-v0`
- Replay requested: HIRE x3
- Protected realized effect: HIRE +1
- Current World check: HIRE feasibility
- Transform: 必要な場合だけHIRE orderを1件へcap
- Promotion evidence: fixed10 10改善 / 0悪化 / 0同値、mean Δterminal self +37,840.5

Source Effectは自動的にContractへ昇格しない。step85 STRAWBERRY restorationは0/0でClosedのまま。

## Proven Runtime Equivalence

Contract Runtime化によって既存Adaptive Replay v0の挙動を変えていない。

- fixed10: 10/10 exact equivalence
- Self action trace: 719 actions exact ×10
- Opponent action trace: exact ×10
- terminal self/opponent: exact ×10
- Guard event: exact ×10
- Run: 36651253022
- Result: `experiments/adaptive_replay_contract_runtime_v0_equivalence_result.json`

## World Evidence

Boundary Observationでは20条件中16改善 / 4悪化。ただし requested HIRE=3 / feasible HIRE=3 は20/20同一で、Adaptiveな入力分岐自体はまだ未検証。

今回の観測で、ReplayのRequested ActionとRealized Effectを分ける必要性が具体化した。

```
step72  Shop difference
   ↓ official _town_consume()
step73  Market inventory difference
   ↓
step96  same SELL WHEAT 8
   ↓ different per-unit quote sequence
step97  Cash 248 vs 242 (+6)
   ↓
step110 same PLANT STRAWBERRY + BUY_SEED STRAWBERRY 1
   ↓ 100-Cash affordability boundary
step111 Seed inventory 5 vs 4
```

重要なのは、途中で新しいPolicy判断が必要だったわけではないこと。

```
same Requested Action
        !=
same Realized Effect
```

World条件がReplayの実行結果を変える。

## Effect Observer

`adaptive_replay_effect_observer_v0.py` はPolicyではない。RuntimeのActionを変更せず、Requested Action / Contract Event / Current World inputを観測可能なrecordへ正規化する薄いObserver。

Observerの出力はEvidenceであってContractではない。Observerが見つけた差を自動修復しない。

## Evidence Manifest

`adaptive_replay_runtime_v0_1_evidence_manifest.json` が、Active Contract、Promotion Evidence、Closed Evidence、Runtime equivalence、World observations、未確定境界を一箇所に固定する。

## Boundary

v0.1で言わないこと:

- Replayが全Worldで正しい
- Source Effectはすべて守るべき
- Contract RuntimeのAdaptive能力が一般化された
- Shop差やSeed差がterminal差の全原因
- ObserverがRecoveryを選べる
- 新しいGuardが必要

Primary metricはsource fidelityではなく **terminal self**。

## Files

- `adaptive_replay_contract_runtime_v0.py` — frozen Contract Runtime behavior
- `adaptive_replay_effect_contracts_v0.json` — promoted Effect Contracts
- `adaptive_replay_world_adapter_v0.py` — minimal Current-World feasibility adapter
- `adaptive_replay_effect_observer_v0.py` — non-intervening observation surface
- `adaptive_replay_runtime_v0_1_evidence_manifest.json` — evidence / boundary manifest
- `run_adaptive_replay_runtime_v0_1_release_gate.py` — release integrity + fixed10 equivalence gate

## v0.1 Acceptance

Releaseは以下を同時に満たしたときだけ成立する。

1. Active Contractが1件のまま。
2. Runtime / Contract / World AdapterのSHA256がmanifestと一致。
3. Effect ObserverはActionを変更しない。
4. fixed10で旧Adaptive Replay v0と719-action traceを含む10/10 exact equivalence。
5. terminal / Guard eventも10/10 exact。
