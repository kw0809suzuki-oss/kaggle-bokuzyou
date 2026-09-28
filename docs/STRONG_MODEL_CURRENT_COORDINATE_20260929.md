# Strong Model 現在地｜2026-09-29

## Objective

提出期限までに、既存Strongのterminal selfを上げる。勝敗とmarginも記録し、selfの改善だけを「強いモデルの完成」とは扱わない。

「資金を次のCash回収まで運ぶ」は先の設計思想として保持する。今は経済台帳や新しい終端評価器を作らず、既存Strongに対する小さな検証を進める。

## Current Position

### Animal Activation｜DiscoveryからVerificationへ

ユーザー共有の介入結果では、Bundle基準のterminal self 12,414に対し、SHEEP早期配置介入後は19,390。self差は+6,976。

同じ一本でmarginは-101,633から-143,352へ悪化した。これは一seed・seat 0の反応であり、採用根拠や一般則ではない。ただし、家畜稼働化候補を再現Battleで確かめる価値は示した。

### MELON｜Discovery

Strong Public Replayの保存成果物 `sellesta-vs-v21-a0`（seed 92801801）から、v21のMELON回収サイクルを一本逆走した。v21はseat 1。

- day 7植付けのMELONが、step 414 / day 17で座標(5,1)、yield 5として残っている。
- step 419でyield 6となり、同stepの2人のHARVEST指示の後、step 420の所持品にMELON 6が現れ、同じ座標の株は盤面から消えた。
- step 423に `SELL MELON 4`。その時の表示価格は60。次のstep 424では手持ちMELONがなくなり、ShedにMELON 2が残った。
- step 423→424のcash差は+768だが、同時にWOOL、WHEAT、FERTILIZERの販売指示も出ている。MELONだけのcash寄与とは分離できない。

これは保存Replay内での一つの実行経路であり、v21がこの対戦に勝った証拠ではない（terminal cashはv21 56,775、Sellesta 78,319）。また、現行Strongとの同seed状態比較はまだ取れていないため、Strongとの「最初の実体差」は未確定。コード変更もしない。

Animal ActivationとMELONは独立したEvidenceとして保持する。似た構造が別々に観測された場合に限り、上位設計へまとめる。

## 次の作業

| 線 | 操作 | 判断 |
|---|---|---|
| Animal Activation | 報告された同じ最小介入を固定し、同seedの両seatと少数の固定seedでBattle比較する。介入以外のコード・条件は固定する。 | terminal selfの改善が再現するか確認。marginと勝敗も併記。再現しなければ閉じる。再現した場合に限り、既存Strongへの最小統合候補へ進める。 |
| MELON | この回収経路を現行Strongの同seed replayと突き合わせ、最初に異なる盤面・行動状態を一件だけ取る。 | まず差分を特定し、原因説明やコード変更はしない。 |

## Boundary

- SHEEP介入の+6,976は一seed・seat 0に限る。
- marginは同じ介入で悪化しており、強さや採用は未確認。
- MELONの販売注文は観測できたが、同stepの複数販売があるため、cash差をMELON単独へ帰属できない。
- MELON経路と現行Strongとの差、およびAnimal Activationとの因果接続は未確認。
- 両線を「OwnedからProductiveへ」という一つの一般則に統合しない。
- 以前のHIRE＋作物の単一局面Probeは、上記二線の採否Evidenceに使わない。

## Evidenceの出所

SHEEP介入の数値は本座標を更新したユーザー共有結果。Raw replay、summary、実行runへの参照はまだ対応付けていない。

MELONの観測はActions run 36316237533のartifact内にある `strong_public_92801801_sellesta-vs-v21-a0.json.gz`。run: https://github.com/kw0809suzuki-oss/kaggle-bokuzyou/actions/runs/36316237533
