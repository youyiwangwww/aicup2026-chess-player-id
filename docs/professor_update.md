# Go Player Identification：教授會議更新

## 任務與目前正式結果

由多盤棋譜在 100 位候選玩家中辨識身份。官方資料：100,000 games／1,582 players；TRAIN／VAL／TEST 身份分開。Round 1 TEST、FINAL TEST 2、FINAL TEST 3 全部永久 CLOSED。

已保存的 TEST3：Color-aware Opening Top-1 **0.72**／Score **0.773919**；Frozen Fusion Top-1 **0.74**／Score **0.790240**。Fusion 提升 **+0.016321**，paired bootstrap 95% CI **[−0.022885, 0.052726]** 包含 0，尚不能確認改善穩定。

## Baseline evolution 與主要發現

Opening 用目標玩家實際開局落點 heatmap 識別風格；DEV2 分黑／白後 score 提升 **+0.103179**。Triplet 提供部分互補，但表示高度集中。Stability 也見到正向 fusion point estimate，CI 同樣包含 0；所有正式方法保持 frozen。

## Phase 2.11：collapse 發生在哪裡？

這輪只用 DEV2 TRAIN fitting／VAL exploratory evaluation；只載入原 Triplet-Hard-100 epoch 18 checkpoint，SHA256 不變，沒有 training、optimizer 或 backward。沒有讀 CLOSED TEST 棋局／truth／score matrix，也沒有用 Stability truth 選模型。

**共同方向在 pooled backbone 已出現，projection 後更集中；L2 normalize 不是主要原因。** Raw embedding 在 normalize 前 cosine 已接近 1。改用 raw mean 或改 normalize 時機，DEV score 都維持 **0.417385**。

Raw 的 PC1 吃掉約 **86%** variance，與均值方向對齊；normalized effective rank 約 **93**，表示仍保留微小的高維角度訊號，並非完全恆定輸出。絕對 near-zero variance 門檻受尺度影響，不能單靠它定位 collapse。

去中心、移除 PCs、whitening 都沒有超過原 Triplet；單純 remove dominant direction 不是足夠的修復。Color-aware Triplet＋global centering 的 DEV score 升到 **0.480703**，但固定 alpha=0.9 fusion 反而由 **0.758266 降至 0.743930**。Triplet 更高的獨立 score 不保證互補性更好；color-specific centering 也沒有超過 global centering。

Within／between ratio 仍約 **0.35**：玩家內棋局差異大於玩家間差異。本輪結論是 encoder/projection 角度集中、color-domain shift 與身份訊號分離不足共同存在；不支持 normalization／aggregation 是單一主因。

## 限制與下一個研究方向

DEV2 已反覆探索，本轮最高後處理結果不是新 final model；沒有新 selected YAML／frozen model。只量測 pooled backbone，尚不能定位到某 residual block；沒有訓練 intervention，不能證明特定 loss 是原因。

最值得討論：**先建立新的 DEV protocol，研究能監測並控制共同均值方向／角度集中與玩家分離度的身份辨識目標，並控制黑／白色彩域因素。** 再規劃 sampling／loss／projection 的受控研究，不以 CLOSED TEST 結果調參。

**Phase 2.11：118 tests／imports／syntax 全通過；checkpoint／forward／historical split 全保留。** 未訓練新模型、不加入 Strength Estimator／MiniZero、不建立新 FINAL TEST。完整數據見 [phase211_results.md](phase211_results.md)。

## Phase 2.12：DEV3 資料門檻

排除歷史 715 位身份後，未使用玩家中只有 **175 位至少 20 盤、54 位至少 40 盤**。指定 150 TRAIN＋50 VAL 必須有 200 位互斥身份；保留 50 VAL 後 TRAIN 最多 **125 位**，少 25 位。

因此沒有建立 DEV3 或訓練 A0–A3，沒有降低規模／重用舊身份；anti-collapse regularization **尚未評估，不能判定成功或失敗**。需先補足未使用玩家資料或另行明確制定新的規模，才進行受控實驗。現有完整 **123 tests／imports／syntax 通過**；新增的是資料門檻與歷史隔離測試。詳見 [phase212_results.md](phase212_results.md)。
