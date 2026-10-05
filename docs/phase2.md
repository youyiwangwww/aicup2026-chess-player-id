# Phase 2：Go Player Identification Metric Learning

## 問題與 Baseline 0

玩家辨識的輸入是一位玩家的多盤棋譜，輸出是在 candidate 資料庫中最可能的玩家。它不是預測棋力或下一步落子。

Opening Fingerprint 把整盤前 40 手中目標玩家的落子，累積成 19×19 heatmap。Candidate/query 各自多盤平均，用 cosine similarity 比較 361 維向量。它容易理解、不用訓練，但只描述開局位置偏好，忽略當時盤面、提子及後續決策。

## 為什麼使用 metric learning

一般分類模型輸出固定訓練玩家名單，新候選身份無法直接套用分類頭。Metric learning 改成學共用函式 f：把玩家的決策盤面轉成 embedding，讓同一玩家不同棋局接近、不同玩家較遠。辨識時為新候選建立 embedding 資料庫，不必新增分類類別。

這是研究假設，不代表所有盤面都能反映玩家風格。對手、執色、開局與棋力仍會影響 representation，需要實驗驗證。

## 17-channel 落子前特徵

參考 [官方 SGFParsePlayerIdentification](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/main/utils.py)：8 個目標玩家 stones planes、8 個對手 planes、1 個執色 plane（黑=1、白=0）。這是 stones occupancy，不是落子次數 heatmap。

重播 SGF 主變化，在目標玩家落子**之前**取樣，避免把此手答案放進當前盤面。History 由舊到新，最後是當前盤面；不足八層用空盤左側補齊。處理 AB/AW/AE、提子與 pass。與參考程式的取樣細節不同，本實作保留早期補齊的回合，也把 pass 視為可取樣回合；這些約定有 tests。

每盤固定抽 4 或 8 個 target turns，只有選中的回合才建完整 tensor。保存成 uint8 壓縮 NPZ shards，不每手產生 .npy；CSV hash、特徵設定及 seed 綁定 manifest，防止舊特徵被誤用。

## Triplet Loss

Anchor 是某玩家某盤的位置；positive 是同玩家**另一盤**的位置；negative 是另一玩家的位置。不可將同盤兩個盤面當正例，否則模型可能只辨識共同棋局。

令 embedding 經 L2 normalize，d 為 Euclidean distance，概念公式：

```text
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
margin = 0.2
```

當 negative 沒有比 positive 遠至少 margin 時產生 loss。實作使用 `torch.nn.TripletMarginLoss(margin=0.2, p=2)`、AdamW，沒有 player classification head，也不用 rank label。

ResNet 的通道數、residual blocks、embedding 維度由 YAML 決定。卷積後以 3×3 spatial pooling 保留粗略區域資訊，再投影並 L2 normalize。Quick 是 32 channels／4 blocks／64 維，正式設定為 64／8／128。

## 為什麼 training 與 validation 玩家必須不同

同一身份若同時出現在 training/validation，模型可能只記住訓練玩家，無法泛化到新候選。我們把正式測試候選視為 unseen players，因此採用完全不同的玩家做 held-out retrieval：

```text
training player IDs ∩ evaluation player IDs = 空集合
training game IDs/SGFs ∩ candidate game IDs/SGFs = 空集合
training game IDs/SGFs ∩ query game IDs/SGFs = 空集合
candidate game IDs/SGFs ∩ query game IDs/SGFs = 空集合
```

交集非零直接報錯，保存 audit。Query 不含身份/rank，ground truth 只供稽核及 validation 評分，從未傳入 training dataset、loss 或梯度。Validation 選 checkpoint 是模型選擇，因此不能把它當獨立最終 test。

## Candidate / Query 怎麼比較

每個 position 產生 unit embedding，先在每盤平均，再在每位 candidate 或每題 query 的多盤棋之間平均，最後 L2 normalize。每盤等權，避免棋局有效 positions 較多而主導結果。

Query 與所有 candidates 做 cosine similarity 排序、取五名，同分依 player_id 字串排序。Opening 使用相同切分與 ground truth。推論函式不讀 ground truth。

## 評分與 checkpoint

Top-1/3/5 Accuracy 以全部題目為分母。正確玩家第 r=1..5 名得 `exp(-(r-1))`，未進前五或缺失題目得 0，最後平均全部題目。

每個 epoch 梯度更新後，在 held-out players 做 retrieval。`last.pt` 保存最後 epoch；`best.pt` 保存最高 validation competition score，同分留較早 epoch。Training loss 只供診斷，不能單憑 loss 下降就聲稱辨識改善。

## 目前結果與下一步

2026-10-05：26 項 tests、syntax/import/依賴檢查通過，Windows CPU 一鍵 mock 成功。10 位合成玩家中 4 位訓練、6 位 held-out，16 盤 training、18 盤 candidate、12 盤 query，所有 8 項 overlap count=0。

| Mock 方法 | Top-1 | Top-3 | Top-5 | Score |
| --- | --- | --- | --- | --- |
| Opening | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Triplet | 0.166667 | 0.833333 | 1.000000 | 0.342703 |

兩個 epoch、32 triplets/epoch，loss 由 0.214364 降至 0.097343，但 validation score 不變，選 epoch 1。合成資料有明顯落子區域偏好，Opening 適合此設定；不構成真實資料的模型優劣結論。

正式 CSV 尚未放入 `data/training/train_A.csv`，沒有正式成績。先跑真實 quick 檢查 coverage/錯誤率/Top-k，再做多 seed、位置數及訓練量消融，保留最終獨立玩家池。Strength Estimator 研究留待 Phase 3。
