# Strong Model｜Bundle Probe 結果

## 目的と候補

親目的はStrong Modelを強くすること。このProbeでは、労働・資産・実行をまとめた初期運転を既存Strongの選択結果とshadow比較し、World応答を観測した。診断BundleはStrongの `choose()` 候補一覧には追加していない。Independentは候補内容の参照にだけ使用。候補束はstep 0で HIRE×2、COW×1、MELON seed×6、WHEAT seed×6、Pasture建設とし、12作物仕事・COW配置仕事を運転予定に含めた。Worldではこの初手を一度だけ注入し、step 1から既存Strongが毎turn再計画した。

## Selection｜同じstep 0 State

| 候補 | 中央終局Cash見積り | 初手 |
|---|---:|---|
| 現行Strongの選択 | 8,628 | BUILD_PASTURE、BUY_ANIMAL SHEEP×1 |
| 診断Bundle | 13,440 | BUILD_PASTURE、HIRE×2、BUY_ANIMAL COW×1、MELON/WHEAT seed各6 |

Bundleのshadow評価は現行Strongの選択結果より4,812高かった。これは同じ評価器を使った別計算との比較であり、Strong自身の選択器がBundleを選んだ結果ではない。初手後は市場処理され、step 1の観測Cash2,058、雇用Hands2、COW在庫1、MELON/WHEAT seed各6、Pasture1面を確認。

## World｜同seed・席入替

| 席 | 対照self | Bundle self | 差 | 対照margin | Bundle margin | 差 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24,351 | 12,414 | -11,937 | -144,779 | -101,633 | +43,146 |
| 1 | 22,387 | 21,155 | -1,232 | -139,791 | -133,628 | +6,163 |

両席ともDONE、719行動で終了。Bundleは両席でself Cashが下がり、marginは両席で改善した。改善を勝利とは扱えない。どちらのmarginも負のまま。Selection上位とWorldのself方向は一致しなかった。

step 0〜14の実行記録では、席0の非PASS行動枠は対照27からBundle42、生産系行動は9から13へ変化。診断Bundle後も、step 1に既存StrongはBUY_LANDとSHEEP購入を発行し、step 2時点で2面目が解放されていた。これは初手だけを変えたWorld介入であり、30日間のBundle方策を評価したものではない。

## このEvidenceの境界

確認できたのは、一件の診断束を同一Stateでshadow評価したところ、現行選択結果より中央見積りが高かったこと、初手の購入・雇用・建設が成立したこと、席入替二戦の終端応答。selfとmarginは異なる向きに動いた。

一seed二席なので一般的な強さは未確定。Bundle評価は将来の自動再雇用・再投資を含まない。候補の各要素が結果へ与えた寄与や、途中のどの差が終端差を作ったかはこのProbeでは判定していない。結果を見てモデルや候補を追加修正していない。

今回のProbeで確認できたのは、Bundleの別計算上の評価値と、step 0だけを注入した二つのWorld応答。native選択器への統合効果や、Bundle運転を継続した場合の強さは未検証。

## 成果物

- `selection.json`：step 0 raw State、現行選択、束評価
- `bundle-seat0/1.summary.json`：終端・実行時間・hash
- `bundle-seat0/1.trace.json`：初期15turnと各日の判断
- `bundle-seat0/1.replay.json.gz`：Raw Replay
- `world-comparison.json`：対照との比較・日次State
- `run_strong_model_v0_bundle_world_probe.py`：実行runner
- `summarize_strong_model_v0_bundle_probe.py`：artifact照合と集計
- `test_strong_model_v0_bundle_probe.py`：候補の最小Contract
