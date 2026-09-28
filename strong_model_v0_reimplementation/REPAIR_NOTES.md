# Strong Model v0 — Cash回収を比較する修理候補

2026-09-28。対象は `experiment/strong-model-v0-reimplementation-20260928` の `6dcfcb8`。設計の正本はルートの `STRONG_MODEL_V0_DESIGN.md`（元の設計文書をそのまま保存）。本修理候補は設計全体の完成・強さの実証を意味しない。

## 今回集中した箇所

[Turn 5 Probe](https://docs.google.com/document/d/1ReAymP87jxH_9nzXk07UFTUqecy0BufHMU48zQDa0Qk/edit)では、136 jobs、うち125 establish_plantが存在し、WHEATの植付け投影も成立している。それでもPASSと植付けがStrict 2740 / Central 5164 / Immediate 2740で一致した。Seed側とPlant側がともに4 unitsと評価されるため、種を植える前から将来売上が計上されていた。候補不足やStrictだけを停止原因とは扱わない。他Crop・全状態への一般化もしない。

## 今回の実装

- 観測を正本に毎turn再計画する。購入・移動・除草・植付けを同じ `plant_crop` の目的として保持し、購入で仕事が消えない。完了した投資は先読み内でも除去する。
- 候補を一turnだけ投影して資産を換算する方式を、実行可能な運転を終局まで進めて財布を読む方式へ置き換えた。種・立毛・家畜・未売却在庫をCashに足さない。
- 将来の維持・収穫・搬入・売却を再生成し、公式の移動・作業・市場・町需要・日境界処理を使う。生産待ちの期間も日境界で仕事を再確認する。
- 継続、新規投資、目的の解放、早期回収、一人雇用、売却を一turn待つ案を比較し、選んだ一turnだけを発行する。全Crop/Animal種類を候補に残すが、位置は各種類の近い代表へ削減している。
- 同turnのSeed/PICKUPを人数で重複使用しない。同品目SELLとBUY_PRODUCTを一律排除しない。給餌や配置用に運ぶ物が汎用DROPへ奪われないようにする。

## 仮定と未実装の境界

これは**限定した運転方策での先読み**。最適な将来運転、正確な予測、設計全体の忠実な完成版とは呼ばない。

中央見積りは相手の将来注文なし・既存店舗継続・新しい雑草なしの想定。厳しい側は「公開中の相手生産物による供給増」と「初手作業の一turn遅延」の感度を見る一例で、保証された下限でも確率区間でもない。互換フィールド `strict_cash` に保存するが、中央見積り以下とは限らない。現候補の採択は中央Cash、厳しい側は診断用であり、設計7.2の頑健な選択は未実装。

作業の割当には予想収益と距離によるヒューリスティックを使う。固定の作物順位はないが、割当自体を最適化したわけではない。新規投資は一件追加が中心で、設計にある初期混合束・遠方代表を十分に比較していない。未来の再投資・翌日の再雇用は自動的には仮定しない。このため雇用が作る将来の拡大能力は過小評価され得る。

売却数量の全比較、給餌用WHEATを売る代案、肥料利用、予定誤差に応じた継続優先、未来の相手売却・店舗の幅は未完成。単一の近距離代表や保留ルールが強さの根拠になったとは扱わない。

## 今回現物から確認したこと

`run_strong_model_v0_cash_recovery_regression.py`：未稼働のSeedのみならCash3000。WHEAT一回の購入から売却までを先読みすると3097、同じ行動列を公式環境で実行しても3097。雇用・植付け・水やり・市場購入・日境界を含む49遷移で、self farm/private/market/town/day/hourが一致した（相手PASS、雑草と新店舗を止めた検証条件）。

最終行動step718でWHEAT4を倉庫地点に携行ならCash3097、一歩離れていれば3000、step719から追加売却は数えず3000。Seedの人数間取り合いと同品目売買の回帰も通過。

既存Contractは8群、48turnのRuntimeまで通過。これらは収益比較の実装確認であり勝率確認ではない。結果JSONは `battle-results/repair-probe/`。

30日対戦の実測結果と次に見るEvidenceは、ルートの `STRONG_MODEL_V0_REPAIR_RESULT.md` に記載。中断前の記憶上の数字は今回の実測として掲載しない。

## 再現

公式環境を `d7729da06cc1382eb742d6980dc3180aa85caa28` に固定してインストール。今回ルールファイルSHA256は `bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`。

```bash
python run_strong_model_v0_cash_recovery_regression.py
python run_strong_model_v0_reimplementation_contract.py
python run_strong_model_v0_repair_probe.py --seat 0
python run_strong_model_v0_repair_probe.py --seat 1
python run_strong_model_v0_repair_probe.py --model independent --label independent --seat 0
# 修理前は6dcfcb8の別checkoutを指定する。元checkoutを書き換えない。
python run_strong_model_v0_repair_probe.py --source-root /path/to/6dcfcb8 --label before --seat 0
```

対戦runnerは公式 `env.run` と既存の制限時間を使用し、終端・行動数・所要時間・コードhash・生Replayを保存する。OpponentはRepo内の `astra_flow_vendor/seyamalam_v21.py`。主目的のself Cashと、相手Cash/marginを分けて記録する。

修理コードcommit `0ba1f3f` のGitHub Contractも成功（Run 36425971886）。
