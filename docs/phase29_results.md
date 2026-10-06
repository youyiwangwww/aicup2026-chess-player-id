# Phase 2.9：Stability Validation

結論：**READY FOR FINAL TEST 3**。這是全新身份的獨立 validation，並非 FINAL TEST 3；本輪未建立或執行任何新的 TEST，沒有訓練、調參或重新選擇 checkpoint。

## 1. Stability split 與安全檢查

官方來源 SHA256 `0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca`。排除 Round 1 TEST 10、Phase 2.6 VAL 50、FINAL TEST 2 50、DEV2 TRAIN 200、DEV2 VAL 100，共 **410 位**。只從 Phase 2.8 已保存的身份 audit 讀取排除名單；Stability pipeline 禁止開啟任何 Phase 2.6／CLOSED TEST artifact，也禁止讀取 DEV2 原始 splits/truth 或其他 checkpoint。

排除後最多 **317 位**符合至少 40 games，本輪固定 seed 42 抽取 **100 位**，每人 candidate 30/query 10，共 3,000／1,000 games、100 questions。沒有降低玩家數。排除身份 overlap、candidate/query game ID overlap、exact SGF overlap 均為 **0**。官方來源的 game ID/SGF 全域唯一，排除整位玩家也排除其所有來源棋局；沒有為確認 overlap 而重新讀舊 TEST。Query CSV 無 player_id、rank。

| partition | total_games | successful_games | failed_games | success_rate | mean_positions | median_positions | min_positions | max_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_candidate | 3000 | 3000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_query | 1000 | 1000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |

Phase 2.8 selected config SHA256：`185576244d9385cfe10cdb461eb4cdebfe0b29484031571e6e69bb873939af1a`。
固定 checkpoint SHA256：`9048144a9bc7d422c8c321d607f73fd1dc378054988741e975b56fbb7dab2d4f`。載入前後 SHA256 相同；epoch 18、model/features 設定與 TRAIN IDs 都已驗證。既有 encoder/pooling/inference code SHA256 同時綁定 Phase 2.8 manifest，沒有改模型。

## 2–4. 固定 Opening／Triplet／Fusion

Opening：Color-aware **player-5**，pass 消耗窗口但不加入 heatmap；B/W 各自建立 fingerprint，query 僅比較同色，再依 query 棋局數加權。

Triplet：只載入 `outputs/phase28/experiments/Triplet-Hard-100/best.pt`，epoch 18；64 channels、8 blocks、128 embedding、history 8、16 positions/game，equal-position/equal-game aggregation。Inference mode，無 optimizer、backprop 或 checkpoint 更新。

Fusion：每題全部 100 個候選 scores 做 **z-score**，固定 `0.9*Opening+0.1*Triplet`，沒有搜尋 alpha、窗口或其他模型。三方法使用完全相同 candidate/question 集合與分母。

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| opening | 0.780000 | 0.950000 | 0.980000 | 0.839382 |
| triplet | 0.350000 | 0.540000 | 0.670000 | 0.408832 |
| fusion | 0.810000 | 0.980000 | 0.980000 | 0.867889 |

## 5. DEV2 vs Stability

DEV2 metrics 僅從既有 summary 取出作**描述性對照**，不讀 DEV2 ground truth、不再 inference，也不作選擇。Difference = Stability − DEV2。

| method | dev2_score | stability_score | difference |
| --- | --- | --- | --- |
| opening | 0.739568 | 0.839382 | 0.099815 |
| triplet | 0.417385 | 0.408832 | -0.008552 |
| fusion | 0.758266 | 0.867889 | 0.109623 |

| method | dev2_top1 | stability_top1 | dev2_top3 | stability_top3 | dev2_top5 | stability_top5 |
| --- | --- | --- | --- | --- | --- | --- |
| opening | 0.690000 | 0.780000 | 0.840000 | 0.950000 | 0.880000 | 0.980000 |
| triplet | 0.380000 | 0.350000 | 0.500000 | 0.540000 | 0.570000 | 0.670000 |
| fusion | 0.710000 | 0.810000 | 0.850000 | 0.980000 | 0.910000 | 0.980000 |

Stability Fusion−Opening = **+0.028506**，相較 DEV2 的 +0.018698。本輪無論好壞都保留 alpha=0.9，不改 frozen 方法。

## 6. Paired bootstrap：1,000 resamples，seed 42

每次對 100 個 question indices 有放回抽 100 次，Opening/Fusion 使用**同一組 indices**，因此差值保留配對相關性。95% CI 採 bootstrap 分布第 2.5/97.5 percentiles；不是把兩個獨立 CI 相減。

`mean` 為原始 questions 的平均估計；`bootstrap_mean` 為 1,000 個重抽樣平均值的平均。所有 bootstrap samples 已存 CSV。

| method | mean | bootstrap_mean | ci_lower | ci_upper | resamples | seed |
| --- | --- | --- | --- | --- | --- | --- |
| opening | 0.839382 | 0.839409 | 0.775829 | 0.899676 | 1000 | 42 |
| fusion | 0.867889 | 0.867638 | 0.812287 | 0.920793 | 1000 | 42 |
| fusion_minus_opening | 0.028506 | 0.028229 | -0.010277 | 0.068760 | 1000 | 42 |

Fusion−Opening CI 包含 0，尚無足夠證據宣稱穩定正向提升。 此 CI 反映固定模型在本次抽取 identities/questions 上的抽樣不確定性，不涵蓋所有未來資料分布或模型選擇不確定性。

## 7. Embedding collapse

沿用同一 checkpoint，candidate/query game embeddings 平均 positions 後 L2 normalize，計算 same/different game-pair cosine；population std 與 128 維 candidate-game population variance。未用診斷重選任何模型。

| Pair | Mean | Median | Std | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| same | 0.999999875977 | 0.999999877597 | 2.4754451095e-08 | 0.9999998604 | 0.999999893355 |
| different | 0.999999871345 | 0.999999873136 | 2.41308526356e-08 | 0.99999985629 | 0.999999888349 |

- Separation：**4.63241567328e-09**。
- Dimension variance mean／median：**1.00132004727e-09／9.79705095744e-10**。
- Dimension variance min／max：**8.06032147736e-10／1.46183414287e-09**。
- Variance < 1e-8 的維度比例：**100.00%**；≥80% 的 collapse warning：**True**。

完整逐維 variance 存於 `dimension_variance.csv`。Same/different 近 1 且高度重疊時，不能因 retrieval 高於隨機就宣稱 collapse 已解決。

## 8. FINAL TEST 3 readiness

事前固定三個條件：

1. Opening 明顯有效：其 bootstrap 95% CI 下界高於均勻隨機排名的解析期望 score。100 候選的期望為 `sum(exp(-(r-1)),r=1..5)/100` = **0.015713**，這不是額外執行 Random 方法。結果：**True**。
2. 固定 Fusion score 高於固定 Opening。結果：**True**。
3. Paired bootstrap 的 Fusion−Opening 平均差為正。結果：**True**。

因此：**READY FOR FINAL TEST 3**。依照使用者指定規則，差值 CI 是否跨 0 另外報告，沒有偷偷加入「CI 下界必須 >0」作第四個門檻。READY 只代表值得規劃新的 one-shot TEST；本輪**沒有建立 FINAL TEST 3**，也沒有加入 Strength Estimator、MiniZero 或新模型。

## 測試、執行與輸出

完整 **92 tests passed**，failure/error/skipped 均 0；所有 imports/syntax 通過。包含封存路徑不可讀、DEV2 身份排除、固定 alpha/window/checkpoint/epoch、bootstrap deterministic、相同 questions 與 inference 無 optimizer/backward 檢查。

正式 Stability pipeline wall-clock **3.71 分鐘**。輸出隔離於 `outputs/phase29/`：split audit、三方法 predictions/per-question scores、完整 similarities、DEV2 對照、bootstrap CI/samples、embedding diagnostics 與 summary。已完成後 command 拒絕重做 inference，報告可從保存結果重建。

檢查流程曾誤用舊版 verifier，讀取一次 CLOSED receipt 的狀態 metadata；已改用獨立 `check_phase29.py`。未讀舊 TEST 棋局或 ground truth、未重做舊 TEST inference；正式 Stability pipeline 沒有開啟任何 CLOSED artifact。

```powershell
python scripts/check_phase29.py
python -m src.phase29_validation --config configs/phase29.yaml
python scripts/report_phase29.py
```
