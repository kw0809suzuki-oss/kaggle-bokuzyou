# Adaptive Circulation Intent Detour v0

## 親目的

Official Worldで terminal self を高く残す。

## 比較

Baseline:
- actor11は既存運転を続ける。
- step252 EAST -> step253 EAST -> step254 PLANT STRAWBERRY が観測済み。

Candidate:
- step252だけ actor11 を HARVEST へ寄り道させる。
- その後は命令列を保存しない。
- actor11の小さなIntentだけ保持する:
  - target: [5,7]
  - operation: PLANT
  - crop: STRAWBERRY
- 毎turnの実Worldで位置・target tile・seedを確認し、目的地へ移動/PLANTする。
- Official Stateで [5,7] の STRAWBERRY成立を観測してIntentを閉じる。
- WATER以後は保持しない。
- 他WorkerとmarketはCurrent Adaptive Circulation出力をそのまま使う。

## Gate

Gateは有望性判定ではない。実装比較面の成立確認だけ。

- step252でWHEAT3が本当にHARVESTされる。
- override layerがactor11以外とmarketを変えていない。
- actor11が実Worldでtargetへ戻る。
- [5,7]にSTRAWBERRYが実際に成立する。
- IntentがOfficial State確認後に閉じる。

成立したら条件を足さずterminalまで走る。

## Adoption

採用判断は terminal_self。
途中のWHEAT取得量や見栄えは採用根拠にしない。
