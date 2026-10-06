# Phase 2.10：FINAL TEST 3 One-Shot Evaluation

## 1. 身份資格、split 與 preflight

官方資料有 100,000 games／1,582 players。排除所有歷史 TRAIN／VAL／TEST／Stability 身份，共 **615 位不同玩家**；剩餘 **967 位 unseen players**，其中 **154 位至少有 40 games**。沒有降低玩家數。

FINAL TEST 3：100 players，每人 candidate 30／query 10，共 3,000 candidate games／1,000 query games／100 questions。所有歷史 player intersection、candidate/query game_id／exact SGF intersection 均為 0；來源 game_id／exact SGF 全域唯一。Query CSV 沒有 player_id／rank。

除了指定的六組排除名單，也排除 Round 1 TRAIN／VAL、Phase 2.6 TRAIN 及其各實驗 TRAIN。旧 TEST 身份只從先前複製的 audit 名單取得；歷史 TRAIN CSV 只讀 player_id 並 hash，沒有開啟舊 TEST CSV／ground truth，也沒有重新 inference。

在 inference 前已完成 immutable preregistration，status=PREREGISTERED／evaluation_count=0。SHA256 綁定 dataset、split audit、三份 CSV、兩份 configs、checkpoint、模型／inference code 與歷史身份來源。**13 項 preflight 全通過**。

- Dataset SHA256：`0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca`
- Split SHA256：`89797cc162233f073cc5589d19bc194c2452221358b2154979b99dc5844c6997`
- Protocol SHA256：`174aa05316b63fadc9e1f932732e74b0403ba0641bf3cd3cd05985c1e8d38a60`
- Frozen config SHA256：`185576244d9385cfe10cdb461eb4cdebfe0b29484031571e6e69bb873939af1a`
- Run config SHA256：`26a622c624e3587f2a15dfec3ee666e7951541612d75a6be59546e83f3560e92`
- Checkpoint SHA256：`9048144a9bc7d422c8c321d607f73fd1dc378054988741e975b56fbb7dab2d4f`

## 2. 固定方法與 feature coverage

Opening 固定 Color-aware-player-5；Triplet 僅載入 Phase 2.8 Triplet-Hard-100 epoch 18 checkpoint（64 channels／8 blocks／128 embedding），history 8／16 positions/game。Fusion 固定逐 question、100 candidates 的 z-score：`0.9*Opening+0.1*Triplet`。Random seed=42，不參與選擇。

沒有 training、optimizer、backward、checkpoint／epoch／window／color-aware／alpha／normalization search。`model.eval()`／`torch.no_grad()`；checkpoint 和 frozen config 前後 SHA256 相同。

| partition | total_games | successful_games | failed_games | success_rate | mean_positions | median_positions | min_positions | max_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val_candidate | 3000 | 3000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_query | 1000 | 1000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |

Opening 解析失敗：0；候選缺少某顏色 fingerprint 的組數：1，沿用已 frozen 的既有處理規則，未修改 split。

## 3. FINAL TEST 3 — ONE-SHOT RESULT

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| random | 0.030000 | 0.060000 | 0.080000 | 0.039392 |
| opening_color_player5 | 0.720000 | 0.890000 | 0.910000 | 0.773919 |
| triplet_hard100 | 0.300000 | 0.590000 | 0.650000 | 0.388429 |
| fusion_alpha09 | 0.740000 | 0.900000 | 0.920000 | 0.790240 |

四方法使用同一組 100 questions／100 candidate players。Opening／Triplet／Fusion 各保存完整 100×100 score matrix；所有 prediction 保存後才解析 ground_truth.csv。四方法在同一次評估 invocation 中各評分一次。

## 4. Paired bootstrap

固定 1,000 resamples／seed 42；每次同一 question index 同時抽取 Opening 與 Fusion，保留差值 covariance。95% CI 使用 percentile 2.5／97.5；mean 是原始 question 平均，bootstrap_mean 是重抽樣平均。

| method | mean | bootstrap_mean | ci_lower | ci_upper | resamples | seed |
| --- | --- | --- | --- | --- | --- | --- |
| opening | 0.773919 | 0.774043 | 0.699318 | 0.843495 | 1000 | 42 |
| fusion | 0.790240 | 0.789429 | 0.718018 | 0.856314 | 1000 | 42 |
| fusion_minus_opening | 0.016321 | 0.015386 | -0.022885 | 0.052726 | 1000 | 42 |

差值 95% CI 跨 0，不能確認 Fusion 提升穩定，且不能依結果重選 alpha。

## 5. Generalization comparison

只使用保存的歷史 summary，沒有讀取歷史原始 validation／TEST 做 inference。

| method | dev2 | stability | final_test3 |
| --- | --- | --- | --- |
| opening | 0.739568 | 0.839382 | 0.773919 |
| triplet | 0.417385 | 0.408832 | 0.388429 |
| fusion | 0.758266 | 0.867889 | 0.790240 |

| partition | fusion_minus_opening |
| --- | --- |
| DEV2 | 0.018698 |
| Stability | 0.028506 |
| FINAL TEST 3 | 0.016321 |

三批 frozen 方法排名一致：**True**（Fusion > Opening > Triplet）。不同玩家批次的 difficulty 可能不同，不把不同 partition 的分數差直接當成模型進步。

## 6. Error overlap（Top-1）

| category | questions |
| --- | --- |
| opening_only_correct | 46 |
| triplet_only_correct | 4 |
| both_correct | 26 |
| both_wrong | 24 |
| fusion_correct_when_opening_wrong | 4 |

Triplet 單獨補對 Opening 錯誤：4 題；融合後補對 Opening 錯誤：4 題。兩者不等同；融合也可能改錯原本 Opening 正確的題目。

## 7. 結果解讀

1. Color-aware Opening Top-1=72.00%／score=0.773919，相對 Random score=0.039392 仍有高辨識能力。
2. Triplet score 高於 Random：True；但先前 Stability 的 representation collapse 警示仍未解決，本輪沒有依 TEST 結果修改模型。
3. Fusion score 高於 Opening：True，差值 **+0.016321**。差值 95% CI 跨 0，不能確認 Fusion 提升穩定，且不能依結果重選 alpha。
4. DEV2／Stability／TEST3 排名一致：True；正向 point estimate 不等同於提升已獲統計確認。
5. 互補性存在於部分問題，詳見 error overlap；未重選 alpha 或重跑。

## 8. CLOSED TEST、測試與操作

永久 **CLOSED TEST／evaluation_count=1**；保留原始 PREREGISTERED manifest，不修改其 count。Receipt 是完成狀態的權威紀錄。

Atomic reservation 在 preprocessing 前建立。中途失敗也標記 FAILED_CLOSED，禁止自動重跑。完成後所有 split CSV（包括 ground truth）再開啟及 inference command 再 invocation 都被 shared utils 的 process-wide audit hook 拒絕。報告只讀 saved receipt／summary，不重新評分。

完整 **103 tests passed**，failure/error/skipped 均為 0；所有 Python imports／syntax check 通過。涵蓋 identity 排除、固定方法／checkpoint、no optimizer/backward、truth 延後、atomic registration、成功／失敗均不可二次 invocation、paired bootstrap 與相同 question/candidate axes。

正式 one-shot wall-clock：**56.20 秒（0.94 分鐘）**。

準備階段曾發現鎖阻擋首次 audit JSON 原子寫入；在 preregistration 前修正並保留相同 split SHA，新增 regression test。當時未保留 inference slot、未做推論／評分。正式 one-shot 沒有重跑。

```powershell
python scripts/check_phase210.py
# 已 CLOSED；下列 prepare / evaluation commands 現在都必須拒絕
python -m src.phase210_final_test3 --config configs/phase210.yaml --prepare
python -m src.phase210_final_test3 --config configs/phase210.yaml
# 只重建報告，不打開 CLOSED CSV，也不重新評分
python scripts/report_phase210.py
```

## 9. 下一個研究問題（本輪不實作）

- 在新的 DEV protocol 中找出 color-aware opening 的有效訊號是否依玩家顏色習慣／開局偏好而異，控制同色與跨色難度。
- 在 TRAIN／新的 DEV 上診斷 Triplet collapse、極小 cosine 差距與 aggregation，先建立 representation 判準，再提出改進。
- 事前固定 Fusion 假設與主要 endpoint，規劃更多獨立身份的驗證以縮小 paired difference CI；不使用 TEST3 調參。

本輪到此停止，不進 Phase 3，不加入 Strength Estimator／MiniZero／新模型，也不自動建立下一個 TEST。
