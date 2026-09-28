# Strong Model v0

独立実装用フォルダ。既存Agentは変更しない。

設計の中心:
- 公式Stateへ毎turn戻る
- 複数の生産・維持・回収を同時に走らせる
- 同じCash / seed / worker / targetを二重使用しない
- 収穫後も搬入・SELL・Cashまで回収仕事を続ける
- 投資は固定Dayではなく、終局までに回収可能かで止める
- 市場は売却と内部利用を分ける

このフォルダの実装は Strong_Model_v0.md のv0 CandidateをBattleへ出すためのもの。
既存のIndependentや他のexperiment branchを採用済み部品として書き換えない。
