# AI CUP 2026 圍棋玩家辨識：Phase 2 進度

報告日期：2026-10-07（三）；實驗紀錄：2026-10-05。

## 1. Competition task

由同一玩家的多盤匿名圍棋棋譜，在已知候選棋譜庫中找出最可能身份，輸出 Top-5。指標是 Top-1/3/5 Accuracy 與平均 competition score；正確玩家第 r 名得 `exp(-(r-1))`，未進前五名得 0。

## 2. Baseline 0：Opening Fingerprint

取整盤前 40 手中的目標玩家落子，建立 19×19 heatmap，candidate/query 各自多盤平均，以 cosine 排名。不需訓練，作為開局位置偏好的對照。

## 3. Baseline 1：Triplet Player Embedding

取目標玩家落子前 17-channel board history，經 ResNet 產生 unit embedding。Anchor/positive 同玩家不同棋局，negative 不同玩家，使用 TripletMarginLoss/AdamW。先在每盤平均 positions，再跨盤平均、正規化，以 cosine retrieval。已完成壓縮特徵、訓練、checkpoint、推論與比較。

## 4. Validation 設計

Training 與 held-out evaluation 玩家完全不同，模擬新候選身份。Held-out 再切 candidate/query，三組資料間檢查 game_id、SGF 字串交集，保存 audit。Query 不含身份、rank 不進模型，ground truth 僅稽核及評估。Best 依 held-out score 選擇；獨立最終 test players 是後續工作。

## 5. 目前結果

26 項 tests、syntax/import/依賴檢查通過，Windows CPU 一鍵 mock 成功。Mock：4 位訓練玩家、6 位 unseen evaluation players；16 盤 training／18 盤 candidate／12 盤 query，8 項 overlap count=0。

| Mock 方法 | Top-1 | Top-3 | Top-5 | Score |
| --- | --- | --- | --- | --- |
| Opening | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Triplet | 0.1667 | 0.8333 | 1.0000 | 0.3427 |

兩個 epoch 的 train loss 由 0.2144 降到 0.0973，但 retrieval score 未改善，best 留 epoch 1。這只驗證流程，不能證明真實競賽效果；正式 `train_A.csv` 尚未放入，沒有真實結果。

## 6. 下一步與 Phase 3

先跑正式 `train_A.csv` quick 實驗，確認錯誤率、coverage 與 unseen-player 成績，再比較多 seed、positions 與模型大小。Phase 3 預計研究 Strength Estimator 的棋力資訊是否有幫助，以獨立對照/消融驗證，不預設能提升。目前未整合 Strength Estimator 或 MiniZero。
