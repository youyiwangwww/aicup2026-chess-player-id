# Phase 2.6：Triplet Baseline Diagnosis（DEV ONLY）

Round 1 TEST 為 CLOSED TEST；FINAL TEST 2 為 LOCKED，評估次數 0。所有選擇只依 DEV VALIDATION。

初次 split 建立時曾重開 FINAL TEST 2 truth 檔案計算 SHA-256；未解析身份或評分。已改為在記憶體計算 hash，所有 tuning 步驟均未開啟該 truth 檔。

驗證：52 tests 通過，failures 0、errors 0；syntax passed True。

## Split audit

DEV TRAIN / DEV VAL / FINAL TEST 2 players = 200 / 50 / 50。

TRAIN 4,000 games；DEV VAL candidate/query 1,000/500；FINAL TEST 2 candidate/query 2,000/500。23 項交集為 0，排除 10 位 Round 1 CLOSED TEST 玩家。

## Opening windows

| experiment | val_top1 | val_top3 | val_top5 | val_score | failed_games |
| --- | --- | --- | --- | --- | --- |
| Opening-10 | 0.620000 | 0.740000 | 0.840000 | 0.663844 | 0 |
| Opening-20 | 0.540000 | 0.780000 | 0.820000 | 0.607028 | 0 |
| Opening-40 | 0.460000 | 0.620000 | 0.660000 | 0.510921 | 0 |
| Opening-60 | 0.420000 | 0.520000 | 0.600000 | 0.445559 | 0 |
| Opening-100 | 0.280000 | 0.480000 | 0.580000 | 0.351386 | 0 |
| Full-game heatmap | 0.120000 | 0.200000 | 0.300000 | 0.143219 | 0 |

## Training data size

| experiment | train_players | games_per_player | best_epoch | val_top1 | val_top3 | val_top5 | val_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 30 | 10 | 6 | 0.100000 | 0.240000 | 0.300000 | 0.135257 |
| A2 | 100 | 20 | 2 | 0.100000 | 0.200000 | 0.280000 | 0.130840 |
| A3 | 200 | 20 | 4 | 0.120000 | 0.160000 | 0.280000 | 0.128241 |

## Positions

| experiment | positions_per_game | best_epoch | val_top1 | val_top3 | val_top5 | val_score |
| --- | --- | --- | --- | --- | --- | --- |
| B4 | 4 | 5 | 0.100000 | 0.220000 | 0.260000 | 0.140227 |
| B8 | 8 | 6 | 0.100000 | 0.240000 | 0.300000 | 0.135257 |
| B16 | 16 | 1 | 0.140000 | 0.300000 | 0.360000 | 0.186007 |

## Random epochs curve

| epoch | train_loss | val_top1 | val_top3 | val_top5 | val_score | is_best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.200750 | 0.140000 | 0.300000 | 0.360000 | 0.186007 | True |
| 2 | 0.195750 | 0.060000 | 0.140000 | 0.240000 | 0.089129 | False |
| 3 | 0.209617 | 0.040000 | 0.120000 | 0.240000 | 0.063585 | False |
| 4 | 0.198382 | 0.060000 | 0.220000 | 0.220000 | 0.104908 | False |
| 5 | 0.193151 | 0.040000 | 0.160000 | 0.200000 | 0.080227 | False |
| 6 | 0.184829 | 0.080000 | 0.180000 | 0.240000 | 0.114495 | False |
| 7 | 0.199238 | 0.060000 | 0.120000 | 0.200000 | 0.076125 | False |
| 8 | 0.200983 | 0.100000 | 0.140000 | 0.180000 | 0.112056 | False |
| 9 | 0.197642 | 0.040000 | 0.120000 | 0.200000 | 0.057572 | False |
| 10 | 0.199154 | 0.020000 | 0.120000 | 0.220000 | 0.046555 | False |
| 11 | 0.196554 | 0.020000 | 0.060000 | 0.140000 | 0.028138 | False |
| 12 | 0.193907 | 0.020000 | 0.080000 | 0.140000 | 0.045060 | False |
| 13 | 0.198878 | 0.040000 | 0.140000 | 0.260000 | 0.072202 | False |
| 14 | 0.194407 | 0.040000 | 0.080000 | 0.140000 | 0.056444 | False |
| 15 | 0.191712 | 0.060000 | 0.080000 | 0.120000 | 0.064069 | False |
| 16 | 0.193013 | 0.020000 | 0.100000 | 0.180000 | 0.042853 | False |
| 17 | 0.195596 | 0.060000 | 0.120000 | 0.180000 | 0.079780 | False |
| 18 | 0.191114 | 0.040000 | 0.160000 | 0.160000 | 0.070193 | False |
| 19 | 0.192319 | 0.040000 | 0.100000 | 0.160000 | 0.049848 | False |
| 20 | 0.190082 | 0.020000 | 0.120000 | 0.180000 | 0.049215 | False |

Loss 下降，但 DEV score 未同步改善；可能過擬合或優化目標與檢索不一致，單次曲線無法區分原因。

## Hard epochs curve

| epoch | train_loss | val_top1 | val_top3 | val_top5 | val_score | is_best |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.236957 | 0.160000 | 0.360000 | 0.460000 | 0.204740 | True |
| 2 | 0.204429 | 0.140000 | 0.380000 | 0.400000 | 0.196101 | False |
| 3 | 0.202749 | 0.240000 | 0.360000 | 0.420000 | 0.276572 | True |
| 4 | 0.201744 | 0.220000 | 0.300000 | 0.360000 | 0.241857 | False |
| 5 | 0.201479 | 0.120000 | 0.280000 | 0.400000 | 0.163714 | False |
| 6 | 0.201214 | 0.200000 | 0.320000 | 0.420000 | 0.229262 | False |
| 7 | 0.200910 | 0.240000 | 0.460000 | 0.540000 | 0.295123 | True |
| 8 | 0.200930 | 0.200000 | 0.360000 | 0.480000 | 0.244973 | False |
| 9 | 0.200807 | 0.220000 | 0.360000 | 0.500000 | 0.258611 | False |
| 10 | 0.200689 | 0.300000 | 0.460000 | 0.540000 | 0.352912 | True |
| 11 | 0.200604 | 0.220000 | 0.340000 | 0.400000 | 0.247270 | False |
| 12 | 0.200509 | 0.280000 | 0.480000 | 0.540000 | 0.337960 | False |
| 13 | 0.200498 | 0.300000 | 0.380000 | 0.520000 | 0.325210 | False |
| 14 | 0.200250 | 0.300000 | 0.420000 | 0.480000 | 0.346503 | False |
| 15 | 0.200272 | 0.240000 | 0.460000 | 0.540000 | 0.295752 | False |
| 16 | 0.200206 | 0.300000 | 0.420000 | 0.500000 | 0.338197 | False |
| 17 | 0.200129 | 0.320000 | 0.520000 | 0.580000 | 0.376701 | True |
| 18 | 0.200084 | 0.280000 | 0.440000 | 0.460000 | 0.330555 | False |
| 19 | 0.199942 | 0.340000 | 0.420000 | 0.520000 | 0.369758 | False |
| 20 | 0.199972 | 0.240000 | 0.400000 | 0.500000 | 0.297300 | False |

## Random vs batch hard

| experiment | best_epoch | val_top1 | val_top3 | val_top5 | val_score |
| --- | --- | --- | --- | --- | --- |
| C20-Random | 1 | 0.140000 | 0.300000 | 0.360000 | 0.186007 |
| H20-Hard | 17 | 0.320000 | 0.520000 | 0.580000 | 0.376701 |

## Cross-color

| direction | method | players | top_1_accuracy | top_3_accuracy | top_5_accuracy | competition_score |
| --- | --- | --- | --- | --- | --- | --- |
| B_to_W | Opening-40 | 34 | 0.029412 | 0.088235 | 0.088235 | 0.044212 |
| B_to_W | Triplet-selected | 34 | 0.029412 | 0.147059 | 0.235294 | 0.062480 |
| W_to_B | Opening-40 | 34 | 0.058824 | 0.147059 | 0.176471 | 0.085908 |
| W_to_B | Triplet-selected | 34 | 0.088235 | 0.147059 | 0.176471 | 0.097661 |

Cross-color eligibility/audit: `{"status": "available", "eligible_players": 34, "dev_val_players": 50, "candidate_games_per_player": 20, "query_games_per_player": 10, "game_id_overlap": 0, "sgf_overlap": 0, "note": "DEV-only eligible subset; directions reuse games across experiments."}`

## Embedding diagnostics

| pair_type | pair_count | mean | median | std | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- |
| same_player | 10000 | 0.999992 | 0.999993 | 0.000002 | 0.999991 | 0.999994 |
| different_player | 490000 | 0.999992 | 0.999992 | 0.000002 | 0.999991 | 0.999993 |

Mean separation = 0.000000；IQR overlap = True；standardized separation = 0.124264。

Same/different IQRs overlap and standardized separation is small; embedding has not learned clear player separability.

## DEV-selected configuration

選定 H20-Hard，best epoch 17；DEV Score 0.376701。

同 DEV Phase 2.5 reference Score 0.133546；絕對改善 0.243154。

設定存於 `configs/phase26_selected.yaml`。不能與 Round 1 CLOSED TEST 分數直接比較。

## 限制與下一步

A1→A2 同時改玩家數與每人棋譜數；A2→A3 才能較直接檢查身份數。每 epoch 固定 512 triplets，資料量增大會降低每位玩家的平均曝光。

多次 DEV 模型選擇可能對 DEV 過擬合；跨棋局 similarity pairs 並非獨立樣本。Cross-color 只用符合黑白棋譜數門檻的 DEV 玩家，不可與不同候選數的原 DEV 分數直接比較。

固定 seed、相同 sampler 的不同 K 會產生不同取樣位置，位置集合並非巢狀；positions ablation 仍可能有取樣波動。

不建議立即進 Phase 3：先固定設定完成獨立 FINAL TEST 2 的一次評估，並確認重現性、資料曝光與 representation。FINAL TEST 2 評估需另行使用者授權。

本輪 runner elapsed = 13704.10 seconds。

## 結果解讀

Opening 最佳為 Opening-10（0.663844），短開局優於全局 heatmap，支持優先研究開局取樣；不能據此證明身份訊號只來自開局。

Hard negative 比其他設定相同的 Random negative 改善 0.190694。它仍低於最佳 Opening；高於 reference 的幅度混合了容量、positions、epochs 與 mining 的變化，不能全歸因於 mining。

資料量 ablation 未顯示較大資料帶來穩定提升，但固定 512 triplets/epoch 導致較大資料的每玩家曝光較少，因此無法排除資料不足。

跨顏色辨識偏弱。兩方向使用同一組 34 位玩家，仍缺少這 34 位玩家的同色與混色配對控制，不能只靠與原 50 位 DEV 的分數差斷言顏色是唯一原因。

Embedding cosine 高度集中且同／不同玩家分布重疊，尚未學出清楚的玩家可分性。暫不建議進 Phase 3。

## 下一輪建議（未實作）

1. 固定 hard negative 與每玩家訓練曝光，再比較 30／100／200 training identities。

2. 固定 nested sampled positions，比較前 10／20 手與全局 positions，維持其他設定不變。

3. 在相同 34 位玩家建立 B→B、W→W、混色及跨色控制組，隔離顏色與候選人組成的效果。

FINAL TEST 2 是否執行一次，等待使用者明確確認；本輪不會自動執行。

Runtime warnings and the separate CUDA limitation: [phase26_runtime.md](phase26_runtime.md). All reported experiments use CPU.
