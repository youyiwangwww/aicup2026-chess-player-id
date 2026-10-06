# AI CUP 2026 圍棋玩家辨識：Phase 2.5 進度

報告：2026-10-07（三）；驗證紀錄：2026-10-06。

## 1. Competition task

由同一玩家的多盤匿名圍棋棋譜，在候選棋譜庫辨識身份，輸出 Top-5。指標為 Top-1/3/5 Accuracy 與平均 competition score：正確玩家第 r 名得 `exp(-(r-1))`，未進前五得 0。

## 2. Data

正式實驗只使用 A rank group 的 `data/training/train_A.csv`，player_id 僅用於 TRAIN triplets，rank 不作辨識 label。**正式 CSV 尚未放入，沒有 real 結果。Mock data 僅用於 pipeline validation，不能當作模型效果。**

## 3. Random baseline

以固定 seed 對每個 TEST question 隨機排列候選身份，提供最低基準；不重複玩家。

## 4. Opening baseline

保留前 40 total moves 範圍內目標玩家的 19×19 heatmap，多盤平均，cosine retrieval，不需訓練。

## 5. Triplet embedding baseline

17-channel 落子前 history → ResNet → unit embedding；positive 同玩家不同棋局，negative 不同玩家，使用 Triplet Loss/AdamW。先平均盤內 positions，再跨盤平均與正規化。

## 6. Train / Validation / Test design

三組玩家身份互斥；TRAIN 只做梯度，VAL 每 epoch 選 best checkpoint，TEST 不在訓練時載入。訓練完成後固定 best.pt，才允許一次 final test。Random/Opening/Triplet 使用同一 TEST，所有 player/game/SGF 交集共 23 項，非零立即報錯。

42 項 tests 通過，包含實際攔截訓練 I/O、確認 TEST ground truth/cache 未讀取、Dataset 不使用 validation 身份、checkpoint 只依 VAL，以及 final test 的執行時序。Mock 為 4 TRAIN／4 VAL／6 TEST，五組 preprocessing 成功率 100%，每盤抽 4 個 positions；這只表示流程正常。

## 7. Real experiment results

| Real TEST 方法 | Top1 | Top3 | Top5 | Score |
| --- | --- | --- | --- | --- |
| Random | 未執行 | 未執行 | 未執行 | 未執行 |
| Opening | 未執行 | 未執行 | 未執行 | 未執行 |
| Triplet | 未執行 | 未執行 | 未執行 | 未執行 |

已備妥 real quick（30 TRAIN／10 VAL／10 TEST）、coverage、dataset SHA-256、環境版本與 summary。待資料放入後執行，再手動跑 seeds 42/123/2026，報告全部結果及 mean/std，不挑最好 test seed。

## 8. Current limitation

尚無真實資料證據；特徵取樣少、triplets 隨機、SGF 去重僅字串相等。VAL 是模型選擇資料；不能依 TEST 分數調參，否則最終 TEST 的獨立性失效。

## 9. Next Phase：Strength-aware Player Identification

先取得可信的 real baseline，再研究棋力資訊是否有助於風格辨識，以控制變因與消融驗證。此輪未加入 Strength Estimator、MiniZero 或新模型。
