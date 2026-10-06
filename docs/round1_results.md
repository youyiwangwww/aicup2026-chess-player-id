# Real Experiment Round 1 — CLOSED TEST

正式資料：100,000 games、1,582 players。

固定設定 `configs/real_quick.yaml`，seed 42。TRAIN / VAL / TEST players = 30 / 10 / 10。

| 方法 | Top-1 | Top-3 | Top-5 | Competition Score |
|---|---:|---:|---:|---:|
| Random | 0 | 0.1 | 0.4 | 0.019028 |
| Opening | 0.6 | 0.9 | 1.0 | 0.715343 |
| Triplet | 0.2 | 0.3 | 0.7 | 0.230301 |

Best epoch = 1；best validation competition score = 0.317174。

Round 1 TEST 已看過結果，永久標記為 **CLOSED TEST**。後續不得用於模型選擇、超參數調整或反覆比較。上述數值為歷史紀錄，不能與不同候選玩家數的新 DEV VAL 分數直接相減作效果比較。

原始 artifacts 保留在原路徑，另以 SHA-256 驗證備份至 `outputs/round1_archive/`。Phase 2.6 artifacts 使用独立的 `outputs/phase26/`，新的 DEV / FINAL TEST 2 排除 Round 1 TEST 玩家。
