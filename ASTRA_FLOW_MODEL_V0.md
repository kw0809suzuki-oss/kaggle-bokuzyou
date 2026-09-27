# Astra Flow Terminal Model v0

目的は Official Kaggriculture の30日後の terminal self 最大化。
これは設計・実装の成果物であり、強さの実証、採用、Frozen Body更新ではない。

## 設計判断

現時点で最も価値のある材料は、Teacherがすでに持つ並行生産の実行能力。
土地・雇用・作物・家畜・世話・運搬を新しく書き直さず、出典を保持した
Teacherを**生産提案器**としてモデル内に固定する。
その提案をそのまま最終意思決定にはせず、新しい売却計画と終局制約を通す。
元のRelationshipSurfaceBodyは参照基準として無変更。

新モデルは `AstraFlowAgent.act(observation)`。
Kaggle用は `agent(observation, configuration=None)`。

処理順序:

1. 前回の公式市場遷移予測と今回の市場在庫との差を観測する。
2. 隔離したTeacherから、Farmer / Hands / Marketの一括提案を受け取る。
3. 終局までの残り手数が帰路とDROPに必要な手数に達したら、携行品の持帰りに切り替える。
4. コピーしたStateへ公式のActor処理を適用し、Market実行直前の在庫を得る。
   PLANTの同時種不足判定、PICKUP、DROP、容量上限も公式コードに従う。
5. 種・家畜について、最短でも成熟→回収→売却が終局に届かない購入を除く。
   日付境界と、購入品を使えるのが次turn以降であることを考慮する。
6. MILK / STRAWBERRY / WOOL / MELONの現在在庫について、今売る量を計算する。
7. 元の生産発注との資金・倉庫・注文枠制約を合わせてActionを返す。
8. 次turnに現在Stateから再計算する。相手とのCash差は目的関数に入れない。

## 売却計画

各品目について0〜全量の整数売却量を候補とし、残量を将来の共通チェックポイントで
売る二段階計画を比較する。チェックポイントは最大24turn先まで。
採用するのは最初の売却量だけで、次turnに再計画する。

評価は現在売却額＋将来売却額のシナリオ平均−0.25×シナリオ間の幅。
公式価格曲線に沿って一単位ずつ値崩れを計算し、価格1では市場在庫が増えない
公式仕様も反映する。同値なら早くCashへ戻す。

既存店舗の消費は公式の間隔・重複店舗数から計算する。
将来の店舗抽選は予言しない。市場残差の指数移動平均と絶対誤差から外部フローの
上下シナリオを作り、フロー0も残す。平滑化係数は0.15。これらは未調整の設計値。
残差には同時取引が自分の約定へ及ぼす影響も含まれ、相手Actionの同定ではない。

購入・雇用等が残るturnでは、元提案の対象品目の売却量を下限として資金源を保持。
新規売却は空いている注文枠に追加し、生産発注を追い出さない。
倉庫＋携行品が容量を超える場合は、追加の売却下限を品目横断で割り当てる。
WHEATとFERTILIZERは生産にも使うため、終局以外はTeacherの保持判断を維持する。
最後の実行turnでは購入を止め、Actor処理後にある売却可能品を売る。

## 意図的な限界

全30日を最適化したという主張はしない。生産側の規模・配置・作業割当はTeacher由来。
売却評価は現在庫の短期回収見込みであり、未生産品の将来価値や全案件の動的最適化ではない。
購入資金の下限は元提案を保持する保守的制約であり、相手の同時取引後の購入成功を保証しない。
将来の新規生産量・新規店舗は予測に含めない。保管余地の制約は現在の携行品を対象とし、
現在turnですでにDROP時点に発生する容量超過を、後続SELLで取り消すことはできない。
注文枠が満杯なら追加の売却もできない。設定は標準の30日・10×10盤を主対象とする。
Teacherの公開日程を使うため、異なる期間や盤面での性能・適合性は未確認。

## 再現・実行

```bash
python -m pip install -r requirements-astra-flow.txt
python -m unittest -v test_astra_flow_terminal_model_v0
python run_astra_flow_runtime_check_v0.py
python build_astra_flow_submission_v0.py /tmp/astra_flow_submission_v0.py
```

単一ファイル版にはTeacherを埋め込むため、実行時の外部取得は不要。
公式 `kaggle-environments==1.32.7` は実行環境に必要。
使用した公式kaggriculture.pyのSHA-256:
`bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e`
（Kaggle/kaggle-environments commit `d7729da06cc1382eb742d6980dc3180aa85caa28` と一致）。
Teacherの出典・ハッシュ・ライセンスは `astra_flow_vendor/NOTICE.md`。

## この実装で確認したこと

- 5つのユニットチェック: 入力不変・reset再現・Teacherインスタンス分離、売却量と保持、
  資金用売却下限・終局に届かない購入、消費イベント境界、最終DROP→SELL。
- Official Worldのseed7001、相手PASSで719turnを実行し、両者DONE。
  実行時間最大約0.003秒/act（今回の環境・この経路のみ）。スコアは出力・評価していない。
- 単一ファイル版の生成・読込とseat1でのAction生成。
- Frozen Bodyファイルに差分なし。

Benchmark、fresh validation、Promotionは未実施。改善・優位性は次工程で判定する。
