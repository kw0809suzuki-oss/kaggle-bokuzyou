# Adaptive Circulation Opportunity v0

## 狙い

Sourceとの差を直す義務から自由になったうえで、
Current Worldに具体的な仕事競合が出た時だけ別案を一回試す。

## v0の競合

同じturnで、

- 成熟Cropの上にactorがいる
- そのactorを含む誰もHARVESTしていない
- 別actorが PLANT / BUILD / animal PLACE で新しい生産へ労働を使っている

場合だけ、一度だけ成熟Crop上actorをHARVESTへ置換する。

## 重要

これは standing-on-return の救済ではない。

standing-on-return は「成熟tile上なら無条件HARVEST」を毎回行った。
v0は「成熟Outputが待つ一方で新規生産労働が同turnに存在する」という
具体的な競合に限定し、episode中一回だけ試す。

効けば条件付きCandidateとして残す。
効かなければこの条件/置換だけ捨てる。
