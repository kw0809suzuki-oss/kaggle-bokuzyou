# 回収を意識した候補生成モデル v0

目的は終了時の自分の現金を増やすこと。現在の数量・所在・生産段階・作業者配置・残り時間から運転案を作り、農場全体の既存先読みで比較する。

## 実装

- `model.py`: 既存Strongの最良案に回収候補を追加し、予測終了現金で選ぶ。
- `agent.py`: Kaggle形式の `agent(obs, configuration)`。毎手再観測し、選ばれた仕事を次の手へ引き継ぐ。
- `run.py`: 同じseed・席・Seyamalam v21で既存Strongと比較する実行入口。
- `test_model.py`: 既存動作との一致、搬入・販売、期限、状態非破壊などの確認。

基準は `strong_model_v0_reimplementation`、commit `ceb799d4f9ab0b2c3d09733c3c347a79aecb6342`。既存Strongのファイルは変更しない。公式Worldは `d7729da06cc1382eb742d6980dc3180aa85caa28` を用いる。

## 運転の作り方

1. 既存Strongが継続・投資・早期収穫・雇用・売却保留を比較する。
2. その最良案の仕事を土台に、収穫対象を一つ優先する案、携行品を納屋へ運ぶ案、新規生産・拡張の仕事を外して維持と回収を残す案を作る。
3. 仕事と優先指定を既存 `_service_action()` へ渡す。完成済みActionの差し替えやStrongの関数の一時置換はしない。
4. 各案を同じ残り期間まで既存 `rollout()` で進め、終了時の予測現金が最も多い案を選ぶ。同点は既存Strongを維持する。
5. 最初の一手だけ実行し、次の観測から再計画する。仕事は保持するが、回収優先を永久に固定しない。

納屋の販売と空いている作業者への配分は既存処理を使う。これらを独立した新機構として追加していない。既存の値付け・仕事順位付けは残るが、所在地ごとの新しい固定点数や「早く現金に戻るほど加点」は追加しない。

追加先読みは標準で最大6案。これは計算量の上限であり、農場の規模・雇用数の上限ではない。近い対象から搬入と収穫を交互に候補へ入れる。全候補の網羅性や最適性は保証しない。`Runtime(cfg, max_candidates=...)` で変更でき、0なら既存Strongと同じ判断になる。

## 期限と価値

回収時間で割った点数を作らず、先読みの終了現金を使う。期限までに全量回収できないことだけを理由に案全体を捨てない。一部回収でも、他の案より最終現金が多い場合がある。残存資産は追加の現金として加算しない。

## 大切な境界

既存先読みは**自分の農場の近似予測**。現在の市場・既存店舗の消費は使うが、将来の相手行動、雑草抽選、店の解放、再投資・自動再雇用までは再現しない。農場全体を進めることと、対戦World全体を正確に再現することは別である。`strict_cash` は既存の不利条件シナリオであり、下限保証ではない。今回も診断値として残す。

提供されたSurface系2ケースでは、shop path固定でterminal効果の符号が反転している。その報告を全候補へ一般化せず、予測が高いことを実戦で強いことと同一視しない。実戦比較では公式Worldの通常の乱数・店舗更新を通す。shop pathの固定や隠れた未来情報の利用はしない。

## 実行

リポジトリ直下で実行する。このフォルダ単独ではなく、既存Strongと共に使う。

```bash
python -m pip install "git+https://github.com/Kaggle/kaggle-environments.git@d7729da06cc1382eb742d6980dc3180aa85caa28"
python -m unittest return_aware_model_v0.test_model -v
python -m return_aware_model_v0.run --steps 24 --output return_aware_model_v0/smoke_result.json
```

終局比較を行う場合:

```bash
python -m return_aware_model_v0.run --steps 719 --seed 92804001 --seat 0 --output return_aware_model_v0/terminal_seed92804001_seat0.json
```

短期確認でも先読みの終了日は720手の設定のまま。実戦だけを指定手数で止める。`terminal_self_delta` は両方が正常終局した場合だけ値を入れる。候補発生数・評価数・採用数・最初の行動差・World差も記録する。未発動や差なしを改善結果として扱わない。

Pythonから利用する場合:

```python
from return_aware_model_v0 import agent, reset_agent, debug_state

reset_agent()  # 新しい対戦の開始前
action = agent(observation, configuration)
record = debug_state(observation["player"])
```

現在は実装候補。勝率やterminal selfの改善を確認してから採用を判断する。
