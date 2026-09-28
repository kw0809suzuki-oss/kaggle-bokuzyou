# Strong Model 現在地｜2026-09-29

## Objective

提出期限までに、既存Strongのterminal selfを上げる。勝敗とmarginも記録し、selfの改善だけを「強いモデルの完成」とは扱わない。

「資金を次のCash回収まで運ぶ」は先の設計思想として保持する。今は経済台帳や新しい終端評価器を作らず、既存Strongに対する小さな検証を進める。

## Current Question｜Animal Activation

家畜稼働化の候補性は、すでに一度Worldへ介入している。現在問うのは候補の有無ではなく、観測されたself改善がseatとseedを越えて残るか。

ユーザー共有の一例では、Bundle基準のterminal self 12,414に対し、SHEEP早期配置介入後は19,390。差は+6,976。同じ一本でmarginは-101,633から-143,352へ悪化した。

これは一seed・seat 0の介入応答であり、再現性・勝敗改善・採用を確認したものではない。

## Animal Activation計画

### 1. 元実験を特定する

元Replayまたは実行記録から、seed、seat、baseline、介入内容、実行コード・runnerを照合する。再現Battleに必要な条件が揃わなければ、代替条件を推測して走らせず、この介入値は「ユーザー共有・未照合」として保持する。

### 2. 比較条件を固定する

元のbaselineとSHEEP早期配置介入だけを比較する。同一seed集合を両seatで使い、介入以外のコード・runner・条件を揃える。seed数は少数の固定集合とし、結果を見て追加・入替えしない。

### 3. World応答を記録する

各seat・seedのbaselineと介入について、terminal self、self差、margin、勝敗を対で記録する。集計値だけでなく各ケースの符号と値を残す。

### 4. 判断する

self改善がseat・seedを越えて再現しなければ、Animal Activation候補を閉じる。再現した場合だけ、既存Strongへの最小統合候補として次の比較へ進める。margin悪化や敗北が残る場合は併記し、「self改善」と「強さの改善」を同一視しない。

## MELON｜独立したDiscovery線

Strong Public Replayの保存成果物 `sellesta-vs-v21-a0`（seed 92801801）から、v21のMELON回収サイクルを一本逆走した。v21はseat 1。

- day 7植付けのMELONが、step 414 / day 17で座標(5,1)、yield 5として残っている。
- step 419でyield 6となり、step 420の所持品にMELON 6が現れ、同じ座標の株は盤面から消えた。
- step 423に `SELL MELON 4`。その時の表示価格は60。次のstep 424では手持ちMELONがなくなり、ShedにMELON 2が残った。
- step 423→424のcash差は+768だが、同時にWOOL、WHEAT、FERTILIZERの販売指示も出ている。MELONだけのcash寄与とは分離できない。

これは一Replay内で経路が実行されたEvidenceであり、v21がこの対戦に勝った証拠ではない（terminal cashはv21 56,775、Sellesta 78,319）。現行Strongとの同seed状態比較はまだ取れておらず、「最初の実体差」は未確定。Animal Activationの根拠や説明には使わない。

## 次の作業

まずAnimalの元実験を特定する。照合できた場合に限り、固定条件の両seat・少数seed Battleを行う。MELONは現状のDiscoveryに留め、Animal検証へ混ぜない。

## Boundary

- SHEEP介入の+6,976は一seed・seat 0に限る。
- marginは同じ介入で悪化しており、強さや採用は未確認。
- 元実験のseed・実行記録・介入条件は未照合。
- MELONの販売注文は観測できたが、同stepの複数販売があるため、cash差をMELON単独へ帰属できない。
- MELON経路と現行Strongとの差は未確認。二線を一つの原因・設計へ統合しない。
- 以前のHIRE＋作物の単一局面Probeは、上記二線の採否Evidenceに使わない。

## Evidenceの出所

SHEEP介入の数値は本座標を更新したユーザー共有結果。Raw replay、summary、実行runへの参照はまだ対応付けていない。

MELONの観測はActions run 36316237533のartifact内にある `strong_public_92801801_sellesta-vs-v21-a0.json.gz`。run: https://github.com/kw0809suzuki-oss/kaggle-bokuzyou/actions/runs/36316237533
