# Strong Model 現在地｜2026-09-29

## Objective

提出期限までに、既存Strongのterminal selfを上げる。勝敗とmarginも記録し、selfの改善だけを「強いモデルの完成」とは扱わない。

「資金を次のCash回収まで運ぶ」は先の設計思想として保持する。今は経済台帳や新しい終端評価器を作らず、既存Strongに対する小さな検証を進める。

## Current Position

### Animal Activation｜DiscoveryからVerificationへ

ユーザー共有の介入結果では、Bundle基準のterminal self 12,414に対し、SHEEP早期配置介入後は19,390。self差は+6,976。

同じ一本でmarginは-101,633から-143,352へ悪化した。これは一seed・seat 0の反応であり、採用根拠や一般則ではない。ただし、家畜稼働化候補を再現Battleで確かめる価値は示した。

### MELON｜Discovery

コードは変更しない。強豪側で実際にCashへ戻ったMELONの一本を、販売からRaw replayで逆走する。対象はSELL、HARVEST、Productive MELON、PLANTと、その前の状態遷移。比較側Strongとの最初の実体差を一件記録する。

Animal ActivationとMELONは独立したEvidenceとして保持する。似た構造が別々に観測された場合に限り、上位設計へまとめる。

## 次の作業

| 線 | 操作 | 判断 |
|---|---|---|
| Animal Activation | 報告された同じ最小介入を固定し、同seedの両seatと少数の固定seedでBattle比較する。介入以外のコード・条件は固定する。 | terminal selfの改善が再現するか確認。marginと勝敗も併記。再現しなければ閉じる。再現した場合に限り、既存Strongへの最小統合候補へ進める。 |
| MELON | 成功したCash回収一本をRawから逆走し、Strongとの最初の実体差を特定する。 | コードは触らず、観測を一件で閉じる。 |

## Boundary

- SHEEP介入の+6,976は一seed・seat 0に限る。
- marginは同じ介入で悪化しており、強さや採用は未確認。
- MELONの成功経路とAnimal Activationの因果接続は未確認。
- 両線を「OwnedからProductiveへ」という一つの一般則に統合しない。
- 以前のHIRE＋作物の単一局面Probeは、上記二線の採否Evidenceに使わない。

## Evidenceの出所

SHEEP介入の数値は本座標を更新したユーザー共有結果。Raw replay、summary、実行runへの参照はこの記録へまだ結び付けていない。再現Battleの準備時に対応する成果物を同定し、固定条件と照合する。
