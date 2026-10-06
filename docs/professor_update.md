# Go Player Identification：教授會議更新

## Problem／Dataset

任務是由多盤 query 棋譜，辨識 100 位 candidate players 中的同一玩家。官方 train_A.csv：100,000 games／1,582 players。指標為 Top-1／3／5 與 competition score（正確名次 r≤5：exp(-(r-1))，未入 Top-5 為 0）。

TRAIN／VAL／TEST 分開玩家身份，避免識別已見玩家。Round 1 TEST、FINAL TEST 2 永久 CLOSED；本輪沒有重新讀其棋局、truth 或做 inference。

## Baseline evolution／關鍵發現

- Opening Fingerprint：用目標玩家實際落子建立 heatmap，平均多盤後 cosine retrieval。開局偏好已提供強訊號。
- Phase 2.8 color-aware-player-5：黑／白分開 fingerprint，DEV2 score **0.739568**，比非 color-aware player-5 **+0.103179**。
- Triplet-Hard-100：原 64-channel／8-block encoder，128 embedding；DEV2 best epoch 18、score **0.417385**。Opening 錯的 31 題中，Triplet 單獨補對 9 題。
- Cross-color gap 很大；配對 CrossColor positive 沒有改善。Stability 同／異玩家 cosine 都接近 1，separation 4.63e-9，全部 128 維近零 variance：**embedding collapse 尚未解決**。
- Frozen Fusion：每題 candidate scores 分別 z-score，alpha=0.9；DEV2 比 Opening **+0.018698**。模型、epoch、window、alpha 全部 frozen。

## Stability／FINAL TEST 3

Stability 使用 100 位新玩家，每人 candidate 30／query 10；score 差 **+0.028506**，paired bootstrap 95% CI **[−0.010277, 0.068760]**，未排除零改善。

FINAL TEST 3 排除歷史全部 **615 位身份**，剩餘 154 位至少有 40 盤；抽取 100 位完全新玩家，同樣 30／10 games。所有 identity／game／SGF overlap=0。Preregistration SHA256 綁定資料／split／config／checkpoint，13 項 preflight 通過，只有一次 inference／evaluation。

| method | dev2 | stability | final_test3 |
| --- | --- | --- | --- |
| opening | 0.739568 | 0.839382 | 0.773919 |
| triplet | 0.417385 | 0.408832 | 0.388429 |
| fusion | 0.758266 | 0.867889 | 0.790240 |

FINAL TEST 3 四方法正式結果：

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| random | 0.030000 | 0.060000 | 0.080000 | 0.039392 |
| opening_color_player5 | 0.720000 | 0.890000 | 0.910000 | 0.773919 |
| triplet_hard100 | 0.300000 | 0.590000 | 0.650000 | 0.388429 |
| fusion_alpha09 | 0.740000 | 0.900000 | 0.920000 | 0.790240 |

Fusion−Opening **+0.016321**，paired bootstrap 95% CI **[-0.022885, 0.052726]**。差值 95% CI 跨 0，不能確認 Fusion 提升穩定，且不能依結果重選 alpha。 三批方法排名一致：True。Triplet 單獨補對 Opening 錯誤 4 題；Fusion 補對 4 題。

## 限制與下一個研究問題

各 partition 玩家難度可能不同；100 questions 的 CI 仍寬，DEV2 已反覆選模型／alpha，不能把 point estimate 改善當作最終定論。Triplet collapse 尚未解決，跨色效果仍弱。

下一步只提出研究建議：在新的 DEV protocol 控制同色／跨色難度；先診斷 representation collapse 與 aggregation；事前固定 Fusion 假設，以更多獨立身份驗證其改善。

**FINAL TEST 3 已永久 CLOSED，evaluation_count=1。103 tests／imports／syntax 全通過。**不使用 TEST3 調參，不重跑，不進 Phase 3；不加入 Strength Estimator、MiniZero 或新模型。
