# Terminal Cash Keeper v0｜フローちゃん個体

## Objective｜目的

```
terminal で self Cash を残す。
```

「循環を閉じること」自体は目的にしない。
Cashへ戻ることだけを終端とする。

## Body｜身体

Replayなし。Teacherなし。

既存の Official World grounded な ShortPlan Generator / Projector だけを再利用する。

現在Worldから実在する仕事候補を生成し、Cashまでの距離で選ぶ。

## Cash Distance｜現金までの距離

近い順：

1. Shed在庫をSELL
2. CarryをShedへDROP
3. 成熟OutputをHARVEST
4. 既存Plantを生存させる
5. 手持ちSeedをPLANT
6. Seedを持っている面のWEEDを除去
7. BUY_SEED

同じ段階なら、現在価格と回収までの推定stepから value density を使って決める。

## Terminal Gate｜終端Gate

新しい仕事について、現在stepから

```
その仕事
→ Yield
→ Carry
→ Shed
→ SELL
→ Cash
```

までの粗い回収距離を計算する。

残りstepに収まらないCandidateは開始しない。

これは「回収できる」と証明するモデルではない。
回収距離はSelection用のHeuristicに過ぎず、成否はOfficial Worldが返す。

## Preemption｜割り込み

遠い投資仕事を進行中でも、WorldによりCashへ一段近い仕事が発生したら、近い仕事が割り込める。

例：

```
PLANTへ移動中
→ 別WorkerがHARVESTを持つ
→ Deliverが発生
→ Deliverを先に通す
→ Shedに入ったらSELLを先に通す
```

これにより「始めた仕事だから最後まで続ける」よりも、terminal Cashへの距離を優先する。

## Boundary｜境界

v0はCrop経路だけ。

Animal / Land / HIREを新しく発明していない。
現在のGeneratorが観測できる経路だけで一台成立させた。

強いとはまだ言わない。

```
Candidateとして成立
!=
terminal Cash改善確認済み
```

最初のGateは runtime smoke。
その後に必要なら同条件A/Bでterminal Cashを見る。
