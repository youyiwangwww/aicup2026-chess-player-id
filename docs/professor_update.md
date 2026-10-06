# AI CUP 2026 圍棋玩家辨識：Real Experiment Round 1

報告：2026-10-07（三）；驗證紀錄：2026-10-06。

## 1. Competition task

由同一玩家的多盤匿名圍棋棋譜，在候選棋譜庫辨識身份，輸出 Top-5。指標為 Top-1/3/5 Accuracy 與平均 competition score：正確玩家第 r 名得 `exp(-(r-1))`，未進前五得 0。

## 2. Data

正式實驗使用 A rank group 的 `data/training/train_A.csv`，rank 不作辨識 label。正式資料有 100,000 盤、1,582 位玩家，每人棋譜平均 63.21、中位數 24、範圍 1–924 盤。黑白各半，game_id 與 SGF 字串重複數皆 0；原始 CSV 未修改。

## 3. Random baseline

以固定 seed 對每個 TEST question 隨機排列候選身份，提供最低基準；不重複玩家。

## 4. Opening baseline

保留前 40 total moves 範圍內目標玩家的 19×19 heatmap，多盤平均，cosine retrieval，不需訓練。

## 5. Triplet embedding baseline

17-channel 落子前 history → ResNet → unit embedding；positive 同玩家不同棋局，negative 不同玩家，使用 Triplet Loss/AdamW。先平均盤內 positions，再跨盤平均與正規化。

## 6. Train / Validation / Test design

三組玩家身份互斥；TRAIN 只做梯度，VAL 每 epoch 選 best checkpoint，TEST 不在訓練時載入。訓練完成後固定 best.pt，才允許一次 final test。Random/Opening/Triplet 使用同一 TEST，所有 player/game/SGF 交集共 23 項，非零立即報錯。

42 項 tests 全部通過，包含訓練 I/O 隔離與 final test 時序。固定 seed 42，Train / Validation / Test players 完全不同，分別為 30／10／10 位。TRAIN 300 盤，VAL candidate/query 為 100／50 盤，TEST candidate/query 為 100／50 盤；23 項交集檢查全部為 0。五組 feature coverage 皆為 100%，600 盤皆成功，每盤抽 4 個 positions。

## 7. Real experiment results

| Real TEST 方法 | Top1 | Top3 | Top5 | Score |
| --- | --- | --- | --- | --- |
| Random | 0% | 10% | 40% | 0.019028 |
| Opening | 60% | 90% | 100% | 0.715343 |
| Triplet | 20% | 30% | 70% | 0.230301 |

使用原本 `configs/real_quick.yaml` 完成 3 epochs；best epoch 為 1，僅依 Validation competition score 0.317174 選取。固定 best checkpoint 後只執行一次 TEST。三種方法使用相同的 10 位候選玩家與 10 個 questions，每 question 含 5 盤 query。Triplet 與 Opening 皆高於這次 Random，Triplet 低於 Opening。完整 pipeline 在 CPU 耗時 59.20 秒，正式執行沒有 warning/error。

## 8. Current limitation

本輪只是 smoke experiment，不能宣稱最終模型效果。TRAIN 僅 30 位玩家、每人 10 盤，只訓練 3 epochs、每盤抽 4 個 positions，triplets 隨機；TEST 僅 10 個 questions，Random 單次結果波動大。SGF 去重僅字串相等。未使用 TEST 調參或選 checkpoint；目前無法確定 Triplet 較弱的因果原因。

## 9. 下一輪實驗建議（尚未實作）

1. 預先指定多個 seed，完整回報所有結果與 mean/std，評估小型 split 的波動，不挑最佳 TEST seed。
2. 固定新一輪 split，比較更多 TRAIN 玩家與每人棋譜数，只依 VAL 評估資料量影響。
3. 分開比較 epochs 與每盤 sampled positions，控制變因；只依 VAL 決定設定，再對新的 held-out TEST 評估一次。

本輪未新增模型，也未加入 Strength Estimator、rank auxiliary loss、MiniZero 或 policy fingerprint。
