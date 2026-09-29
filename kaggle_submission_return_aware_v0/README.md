# Kaggle提出用｜回収を意識した候補生成モデル v0

このフォルダは `model/return-aware-v0-20260929` の `return_aware_model_v0` を Kaggle 牧場へ登録するための提出面です。

- 提出ファイル: `submission.py`
- Kaggle callable: `agent(obs, configuration=None)`
- モデル本体・Strong・必要な補助コードを1ファイルへ埋め込み済み
- 外部の研究用ファイルやテストファイルは提出に不要
- 実行時には Kaggriculture を含む公式 `kaggle_environments` が必要

注意: 24手Smokeでは正常動作確認済みですが、今回の追加回収候補の実戦強化は未確認です。
