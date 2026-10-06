# Phase 2.12：Anti-Collapse Metric Learning

## 教授 1 分鐘摘要

原 **Phase 2.12-A** 在任何訓練前因 150 TRAIN／50 VAL 身份資格不足被阻擋。使用者事前明確授權 **Phase 2.12-R：100 TRAIN／50 VAL**，不是看模型分數後降規模。原 eligibility／blocked summary 保留原 SHA256。

新 DEV3：100 TRAIN players／2,000 games；50 VAL players／candidate 1,500／query 500。所有历史身份与 TRAIN／VAL 身份、game ID／exact SGF overlap=0。

四組各 20 epochs、同 seed／初始化／cache／取樣預算，只改 mean／variance regularizer。最佳 epoch 只依 normal DEV3 competition score，secondary color-aware／fusion 不參與選擇。

1. A0 best checkpoint 是否仍達本輪 collapse 警示（mean direction>0.95 且 near-zero維≥80%）：**True**。
2. A1 mean-direction regularizer 是否降低 best VAL 共同方向：**False**。
3. A2 variance regularizer 是否增加 best VAL dimension variance：**True**。
4. Combined A3 是否 normal retrieval 最高：**True**；最高為 **A3-Triplet-MeanVariance**。
5. Geometry＋retrieval 同時改善：**A2-Triplet-Variance**。
6. Color-aware secondary 提升的實驗：**A1-Triplet-Mean, A3-Triplet-MeanVariance**。
7. 固定 α=0.9 fusion 高於 Opening 的實驗：**A0-Triplet, A2-Triplet-Variance, A3-Triplet-MeanVariance**。
8. 結論：**PROMISING REPRESENTATION INTERVENTION**。至少一個固定 intervention 在 DEV3 同時提高 retrieval，且四個主要 geometry 指標有至少三項改善。仍只有單 seed／單 DEV3，不能宣稱泛化已確認。 四組 mean direction norm 都仍接近 1，near-zero dimension fraction 都為 100%；A2 cosine separation 雖上升，仍只有約 1e-8，between/within 反而下降。方向性 promising 不等於 collapse 已解決，也不代表統計顯著改善。

下一輪 SupCon：目前已有值得延伸的 representation intervention，建議先做新的 DEV／多 seed 確認，不急於加入 SupCon。 不建立 FINAL TEST 4。

## Phase 2.12-A：Original protocol blocked（保留完整原始紀錄）

以下為原報告，描述的是修訂前的狀態，不代表本輪未訓練：

<details>
<summary>150 TRAIN／50 VAL 原 protocol 與 blocked record</summary>

# Phase 2.12：Anti-Collapse Metric Learning

## 教授 1 分鐘摘要

**目前停在 DEV3 資料資格門檻，尚未訓練 A0／A1／A2／A3。**

排除所有歷史 TRAIN／VAL／TEST／Stability 共 715 位身份後，剩餘 867 位玩家；其中只有 175 位至少有 20 盤，54 位至少有 40 盤。

指定 DEV3 要求 150 位 TRAIN＋50 位 VAL，身份不能重疊，因此至少需要 200 位達 20 盤的未使用玩家。保留 50 位 VAL 後，TRAIN 最大只有 **125 位**，比要求少 **25 位**。沒有降低規模、建立 split、重用舊身份或產生任何模型分數。

因此原 Triplet 在新 DEV3 是否 collapse、mean／variance regularization 是否有效、combined 是否最好、color-aware／fusion 是否互補，都**尚未評估**。不能把本輪標成 regularization 成功或失敗，也不能得出 `Simple anti-collapse regularization insufficient.`。

下一步先解決資料規模：若保持 150／50，至少再需要 25 位未使用且達 20 盤的官方玩家資料。尚無實驗證據支持跳過本輪直接改 SupCon。

## 1. 資料資格與歷史排除

| 項目 | 數量 |
|---|---:|
| 官方全部 players | 1,582 |
| 所有歷史使用過的不同身份 | 715 |
| 尚未使用的 players | 867 |
| 至少 20 games 的未使用 players | 175 |
| 至少 40 games 的未使用 players | 54 |
| 保留指定 50 VAL 後，maximum eligible TRAIN players | **125** |
| 指定 TRAIN players | 150 |
| 指定 VAL players | 50 |
| 所需至少 20 games 的不同 players | 200 |
| 缺少至少 20 games 的不同 players | **25** |

資料 SHA256：`0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca`，與既有官方來源紀錄相同。

歷史排除名單來自 `outputs/phase210/eligibility.json` 的已複製身份 metadata（615 位），加上 `outputs/phase210/inference_complete.json` 的 FINAL TEST 3 candidate identity metadata（100 位）。兩組身份互斥；也包含 Phase 2.11 使用的既有 DEV2 身份。

沒有開啟舊 TEST 原始 CSV、ground truth、score matrix 或 checkpoint；沒有讀取 DEV2／Stability 的 ground truth 進行本輪選擇。官方 CSV 只讀 player_id 進行資格統計，另核對 SHA256。

資格門檻先保留 50 位達 40 盤的 VAL；這些玩家同時包含在 175 位達 20 盤的 pool 中，不能再重複算成 TRAIN。故 `175−50=125`，不是有 175 位就能滿足 150 位 TRAIN＋50 位 VAL。

## 2. 執行狀態

**BLOCKED BY DATA ELIGIBILITY — NOT EVALUATED**

| 工作 | 狀態 |
|---|---|
| 官方資料與 unseen eligibility | 已完成 |
| 歷史身份排除／metadata SHA256 記錄 | 已完成 |
| DEV3 splits | 未建立 |
| player／game_id／exact SGF split overlap audit | 不適用：沒有 split |
| feature preprocessing | 未執行 |
| A0／A1／A2／A3 training | 未執行 |
| best epoch／retrieval／best geometry | 未產生 |
| color-aware retrieval／Opening／fixed fusion | 未執行 |
| regularization hyperparameter／loss implementation | 未新增：先停止於資料門檻 |
| `configs/phase212_best_dev.yaml` | 未建立 |
| 新模型／正式 selected model／FINAL TEST 4 | 未建立 |

既有 production forward、checkpoint、實驗分數与 CLOSED 狀態全部保留。沒有偷偷降低成 125 TRAIN／50 VAL 或 150 TRAIN／25 VAL。

## 3. 測試與輸出

完整 **123 tests passed**，failure／error／skipped 均為 0；所有 Python imports／syntax check 通過。

本輪新增 5 項測試：VAL 不能重複計入 TRAIN、精確滿足 150／50 的門檻、TRAIN 足夠仍不能取代不足的 VAL、歷史 TEST／DEV／Stability／checkpoint paths 拒絕，以及 copied identity metadata 只能讀不可改。

這些測試驗證的是 eligibility／隔離門檻；A0 loss 等價、regularizer gradients 與 checkpoint selection 等訓練測試尚未實作，不能宣稱已通過。

輸出：

- `outputs/phase212/eligibility.json`：完整資格統計、歷史身份組與 metadata SHA256。
- `outputs/phase212/verification.json`／`tests.log`：完整測試與 import／syntax 結果。
- `outputs/phase212/blocked_summary.json`：本輪未訓練與未建 split 的明確狀態。

```powershell
python scripts/phase212_eligibility.py
# 目前 exit code=1 是預期的資料門檻拒絕，不是 training failure。
python scripts/check_phase212.py
```

## 4. 結论與下一步

本輪是 **資料不足，尚未評估 anti-collapse intervention**；PROMISING／INSUFFICIENT 的模型成功條件不可判定。

保持指定規模時，至少需要補足 25 位達 20 盤的未使用玩家，並維持至少 50 位達 40 盤。如果未來另行指定新的實驗規模，應明確改寫 DEV3 protocol，再開始四組共同 split 的受控實驗。

現在不加入 SupCon、ArcFace、classification head、額外 loss、Strength Estimator 或 MiniZero；不建立 FINAL TEST 4，也不使用 CLOSED TEST 來解決資料不足或選模型。


</details>

## Phase 2.12-R：Revised 100／50 protocol

### 1. 修訂、資料隔離與 freeze

`revised_protocol.json` 記錄 original=150／50、status=BLOCKED BEFORE TRAINING、revised=100／50、reason=insufficient unseen eligible identities、user-authorized=true、revision before any A0–A3 training=true。

歷史 715 位身份僅從已保存 copied audit/metadata 排除，未讀 CLOSED TEST 原始 CSV／truth／score matrix。資料 source SHA256：`0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca`。

TRAIN=100×20=2,000；VAL=50×candidate30/query10=1,500／500，共50 questions。Query 不含 player_id／rank。所有 partition 的 game_id／exact SGF overlap=0，TRAIN／VAL／historical identity overlap=0。

Split CSV／audit、修訂與固定 TRAIN subset 建立後 freeze；每次入口 hash-verify，不允許重新抽 identity。原 `eligibility.json` 與 `blocked_summary.json` 未刪除或改寫。

### 2. 共同設定與 loss 定義

17 channels／history8／16 positions per game；64 channels／8 residual blocks／Adaptive Pool3×3／Linear576→128／L2 output。沿用等價 deterministic adaptive-average operator，不改模型結構／state keys。

AdamW lr=0.001／weight_decay=0.0001，batch16，margin0.2、batch-hard negatives，20 triplets/player/epoch，2,000 anchors/epoch，每玩家精確20次，共20epochs，seed42。

四組從相同 random initialization 開始，initial model SHA256：`ec709ebeee5b992e0257165d257f39fdd3450b9507f5be81ff1b56be5b234212`。沒有載入 Phase 2.8 checkpoint；新權重僅寫入 `outputs/phase212/experiments/`。

- A0：Triplet 原 loss，mean／variance components=0。
- A1：Triplet＋0.1×sum(batch mean²)，variance component=0。
- A2：Triplet＋mean(ReLU(1/sqrt128−sqrt(population variance＋1e−4)))，mean component=0。
- A3：Triplet＋0.1×mean loss＋1.0×variance loss。

每 batch anchor16＋positive16＋negative source16=**48 source-forward rows**；每 row 是原一次 forward 的結果。Regularizer 對此完整 pool 算一次；被 hard-negative selection 重用的 rows 不另外附加、不重新計權。採 population variance（correction=0），gamma=0.08838834764831843，沒有 covariance loss／SupCon／classification head。

取樣偶爾可以重複同一來源 position；本輪保持原 source-forward pool 語義，不額外改 sampling／dedup 策略。所有實驗使用完全相同的原始取樣，唯一 objective 差異是固定 regularizer。

Loss log 中 mean／variance 是未乘係數的項，total≈triplet＋0.1×mean＋variance；逐 batch assert，inactive項精確0。Batch std log 不加epsilon，regularizer std 加固定epsilon。

| partition | total_games | successful_games | failed_games | success_rate | mean_positions | median_positions | min_positions | max_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2000 | 2000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_candidate | 1500 | 1500 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_query | 500 | 500 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |

### 3. DEV3 normal retrieval／best epochs

| experiment | best_epoch | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | 16 | 0.560000 | 0.700000 | 0.760000 | 0.585956 |
| A1-Triplet-Mean | 19 | 0.540000 | 0.720000 | 0.780000 | 0.590602 |
| A2-Triplet-Variance | 10 | 0.600000 | 0.680000 | 0.840000 | 0.634879 |
| A3-Triplet-MeanVariance | 19 | 0.600000 | 0.740000 | 0.820000 | 0.645555 |

20 epochs 全部完成。每組 best 只取 normal DEV3 score 最大值，相等時保留最早 epoch；没有用 geometry／color-aware／fusion 改 epoch。

### 4. Best checkpoint 完整 geometry

| experiment | mean_direction_norm | raw_pc1_explained | normalized_effective_rank | cosine_separation | between_within_ratio |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.999999922473 | 0.886261 | 91.447099 | 6.3287219e-09 | 0.398375 |
| A1-Triplet-Mean | 0.999999935276 | 0.747456 | 88.269708 | 6.1774347e-09 | 0.414580 |
| A2-Triplet-Variance | 0.999999808790 | 0.790375 | 97.647899 | 1.1640872e-08 | 0.356794 |
| A3-Triplet-MeanVariance | 0.999999933257 | 0.830781 | 88.369505 | 6.2502277e-09 | 0.403373 |

| experiment | variance_mean | variance_median | variance_min | variance_max | near_zero_fraction | within_mean | between_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 1.2113661e-09 | 1.1946675e-09 | 9.569483e-10 | 1.6011595e-09 | 1.000000 | 0.000377 | 0.000150 |
| A1-Triplet-Mean | 1.0113174e-09 | 1.0035664e-09 | 7.1709052e-10 | 1.4061585e-09 | 1.000000 | 0.000344 | 0.000143 |
| A2-Triplet-Variance | 2.9876498e-09 | 2.9535596e-09 | 2.3222581e-09 | 3.8580741e-09 | 1.000000 | 0.000597 | 0.000213 |
| A3-Triplet-MeanVariance | 1.0428564e-09 | 1.0168952e-09 | 7.6383403e-10 | 1.4434573e-09 | 1.000000 | 0.000350 | 0.000141 |

| experiment | same_cosine_mean | same_cosine_median | same_cosine_std | different_cosine_mean | different_cosine_median | different_cosine_std | same_euclidean_mean | different_euclidean_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.999999851509 | 0.999999853188 | 2.8923343e-08 | 0.999999845180 | 0.999999847289 | 2.8822599e-08 | 0.000542 | 0.000554 |
| A1-Triplet-Mean | 0.999999877040 | 0.999999879284 | 2.5020813e-08 | 0.999999870862 | 0.999999873056 | 2.5280851e-08 | 0.000493 | 0.000506 |
| A2-Triplet-Variance | 0.999999629572 | 0.999999634505 | 7.1014857e-08 | 0.999999617931 | 0.999999623054 | 7.0647414e-08 | 0.000857 | 0.000870 |
| A3-Triplet-MeanVariance | 0.999999872904 | 0.999999875057 | 2.6023903e-08 | 0.999999866653 | 0.999999869023 | 2.647523e-08 | 0.000502 | 0.000514 |

Best VAL geometry 在 **unit-normalized game means** 上計算 normalized mean direction、variance、effective rank／within-between；raw PC1 在 raw game means上計算。Same／different pairs 使用全部 candidate-query games（same15,000／different735,000），cosine separation=same−different。Euclidean 是未平方距離，within/between也用未平方距離。

VAL covariance 只是 best checkpoint 的描述性 diagnostic，不 fitting後處理／不參與checkpoint選擇。

### 5. 固定 TRAIN geometry monitoring

固定20 TRAIN players各5 games，共100盤／1,600 sampled positions，subset IDs／game IDs與SHA在訓練前保存；每 epoch eval mode／no gradients。Mean、variance、raw PC1與 normalized effective rank 使用該 subset 的 **position-level** outputs。與 best VAL game-level geometry 的統計粒度不同，不能直接把兩者數值變化當作 improvement。

下表顯示 epoch1／best／epoch20；完整20epochs保存在各 `geometry_log.csv`：

| experiment | epoch | mean_direction_norm | variance_mean | near_zero_fraction | raw_mean_vector_norm | raw_centered_rms | raw_mean_scatter_ratio | raw_pc1_explained | normalized_effective_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 1 | 0.999978 | 3.5090099e-07 | 0.000000 | 133.011857 | 1.303806 | 102.018144 | 0.542038 | 72.337573 |
| A0-Triplet | 16 | 0.999999851766 | 2.3394052e-09 | 1.000000 | 242.676962 | 0.331812 | 731.369540 | 0.842270 | 81.333450 |
| A0-Triplet | 20 | 0.999999880144 | 1.8772394e-09 | 1.000000 | 254.857242 | 0.264915 | 962.034338 | 0.780123 | 74.766211 |
| A1-Triplet-Mean | 1 | 0.999975 | 3.8280171e-07 | 0.000000 | 130.905430 | 1.567225 | 83.526917 | 0.670438 | 71.341151 |
| A1-Triplet-Mean | 19 | 0.999999874024 | 1.9647931e-09 | 1.000000 | 247.391648 | 0.224761 | 1100.685751 | 0.697998 | 78.145388 |
| A1-Triplet-Mean | 20 | 0.999999876679 | 1.9253722e-09 | 1.000000 | 249.802446 | 0.258999 | 964.490346 | 0.775149 | 74.214004 |
| A2-Triplet-Variance | 1 | 0.999975 | 3.8825857e-07 | 0.000000 | 132.865477 | 1.462440 | 90.851945 | 0.600902 | 75.470710 |
| A2-Triplet-Variance | 10 | 0.999999614124 | 6.0273673e-09 | 1.000000 | 219.836693 | 0.414941 | 529.802001 | 0.785909 | 85.127798 |
| A2-Triplet-Variance | 20 | 0.999999873924 | 1.995921e-09 | 1.000000 | 256.122976 | 0.292155 | 876.667328 | 0.805909 | 77.014284 |
| A3-Triplet-MeanVariance | 1 | 0.999973 | 4.2745896e-07 | 0.000000 | 129.493076 | 1.358850 | 95.296087 | 0.514327 | 69.546559 |
| A3-Triplet-MeanVariance | 19 | 0.999999873286 | 1.9803033e-09 | 1.000000 | 252.609352 | 0.272557 | 926.813272 | 0.785373 | 77.192640 |
| A3-Triplet-MeanVariance | 20 | 0.999999875437 | 1.9396522e-09 | 1.000000 | 254.493496 | 0.347753 | 731.822486 | 0.869913 | 75.492973 |

### 6. Color-aware secondary retrieval

| experiment | retrieval | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | Normal | 0.560000 | 0.700000 | 0.760000 | 0.585956 |
| A0-Triplet | Color-aware | 0.520000 | 0.740000 | 0.900000 | 0.579106 |
| A1-Triplet-Mean | Normal | 0.540000 | 0.720000 | 0.780000 | 0.590602 |
| A1-Triplet-Mean | Color-aware | 0.540000 | 0.780000 | 0.940000 | 0.624438 |
| A2-Triplet-Variance | Normal | 0.600000 | 0.680000 | 0.840000 | 0.634879 |
| A2-Triplet-Variance | Color-aware | 0.520000 | 0.760000 | 0.800000 | 0.605002 |
| A3-Triplet-MeanVariance | Normal | 0.600000 | 0.740000 | 0.820000 | 0.645555 |
| A3-Triplet-MeanVariance | Color-aware | 0.640000 | 0.820000 | 0.860000 | 0.688977 |

B query只比B candidate bank、W只比W；按query各顏色game數加權。沿用 Phase2.11 定義，missing color bank similarity=0，沒有跨色 fallback。僅在既選定 best checkpoint上診斷，沒有用結果重新訓練或改regularizer。

### 7. Opening reference與fixed fusion

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Color-aware-player-5 | 0.880000 | 0.940000 | 0.980000 | 0.903435 |

| experiment | alpha | normalization | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.900000 | z-score | 0.900000 | 0.960000 | 0.980000 | 0.918418 |
| A1-Triplet-Mean | 0.900000 | z-score | 0.860000 | 0.960000 | 0.980000 | 0.893133 |
| A2-Triplet-Variance | 0.900000 | z-score | 0.880000 | 0.940000 | 0.980000 | 0.904064 |
| A3-Triplet-MeanVariance | 0.900000 | z-score | 0.900000 | 0.940000 | 0.980000 | 0.916707 |

Opening只跑固定player5/color-aware一次，沒有window/color搜尋。Fusion固定逐question跨candidate z-score，alpha=0.9，使用normal Triplet scores；不挑color-aware版本、不搜尋alpha、不參與Triplet selection。

### 8. Geometry × Retrieval

| experiment | competition_score | retrieval_delta_vs_A0 | mean_direction_norm | raw_pc1_explained | cosine_separation | between_within_ratio | geometry_notes | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.585956 | 0.000000 | 0.999999922473 | 0.886261 | 6.3287219e-09 | 0.398375 | 0/4 primary geometry directions improved | A0 REFERENCE |
| A1-Triplet-Mean | 0.590602 | 0.004646 | 0.999999935276 | 0.747456 | 6.1774347e-09 | 0.414580 | 2/4 primary geometry directions improved | RETRIEVAL IMPROVED WITHOUT CLEAR GEOMETRY FIX |
| A2-Triplet-Variance | 0.634879 | 0.048923 | 0.999999808790 | 0.790375 | 1.1640872e-08 | 0.356794 | 3/4 primary geometry directions improved | PROMISING REPRESENTATION INTERVENTION |
| A3-Triplet-MeanVariance | 0.645555 | 0.059599 | 0.999999933257 | 0.830781 | 6.2502277e-09 | 0.403373 | 2/4 primary geometry directions improved | RETRIEVAL IMPROVED WITHOUT CLEAR GEOMETRY FIX |

Geometry主要四軸：mean direction↓、raw PC1↓、cosine separation↑、between/within↑。事前規定至少3/4改善才稱majority geometry improvement；effective rank僅輔助、不單獨判成功。這是單seed方向性判準，沒有統計顯著性保證。

最高normal retrieval候選為 **A3-Triplet-MeanVariance**；`configs/phase212_best_dev.yaml`明確標記 DEV3 RESEARCH CANDIDATE ONLY／not_final_model=true／no_test_evaluation=true。Geometry與retrieval分開報告，沒有新frozen competition model。

主要geometry方向改善最多為A2（3/4）；A2 mean direction／cosine separation較佳，A1 raw PC1／between-within較佳，沒有一個intervention在全部主要軸勝出。四組 mean direction norm 都仍接近 1，near-zero dimension fraction 都為 100%；A2 cosine separation 雖上升，仍只有約 1e-8，between/within 反而下降。方向性 promising 不等於 collapse 已解決，也不代表統計顯著改善。

**PROMISING REPRESENTATION INTERVENTION**。至少一個固定 intervention 在 DEV3 同時提高 retrieval，且四個主要 geometry 指標有至少三項改善。仍只有單 seed／單 DEV3，不能宣稱泛化已確認。

### 9. 每epoch loss decomposition與validation log

### A0-Triplet

| epoch | train_total_loss | train_triplet_loss | train_mean_loss | train_variance_loss | val_top1 | val_top3 | val_top5 | val_competition_score | best_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.211649 | 0.211649 | 0.000000 | 0.000000 | 0.300000 | 0.500000 | 0.540000 | 0.351684 | False |
| 2 | 0.201302 | 0.201302 | 0.000000 | 0.000000 | 0.240000 | 0.500000 | 0.580000 | 0.301795 | False |
| 3 | 0.200807 | 0.200807 | 0.000000 | 0.000000 | 0.260000 | 0.460000 | 0.580000 | 0.319688 | False |
| 4 | 0.200578 | 0.200578 | 0.000000 | 0.000000 | 0.280000 | 0.580000 | 0.720000 | 0.351700 | False |
| 5 | 0.200442 | 0.200442 | 0.000000 | 0.000000 | 0.280000 | 0.580000 | 0.700000 | 0.357872 | False |
| 6 | 0.200333 | 0.200333 | 0.000000 | 0.000000 | 0.320000 | 0.560000 | 0.720000 | 0.394506 | False |
| 7 | 0.200274 | 0.200274 | 0.000000 | 0.000000 | 0.380000 | 0.560000 | 0.740000 | 0.443360 | False |
| 8 | 0.200224 | 0.200224 | 0.000000 | 0.000000 | 0.340000 | 0.560000 | 0.700000 | 0.398110 | False |
| 9 | 0.200185 | 0.200185 | 0.000000 | 0.000000 | 0.320000 | 0.620000 | 0.680000 | 0.396144 | False |
| 10 | 0.200149 | 0.200149 | 0.000000 | 0.000000 | 0.340000 | 0.640000 | 0.760000 | 0.422523 | False |
| 11 | 0.200125 | 0.200125 | 0.000000 | 0.000000 | 0.460000 | 0.660000 | 0.840000 | 0.511485 | False |
| 12 | 0.200102 | 0.200102 | 0.000000 | 0.000000 | 0.380000 | 0.720000 | 0.820000 | 0.471592 | False |
| 13 | 0.200088 | 0.200088 | 0.000000 | 0.000000 | 0.520000 | 0.680000 | 0.860000 | 0.580024 | False |
| 14 | 0.200080 | 0.200080 | 0.000000 | 0.000000 | 0.460000 | 0.720000 | 0.780000 | 0.534752 | False |
| 15 | 0.200068 | 0.200068 | 0.000000 | 0.000000 | 0.500000 | 0.740000 | 0.820000 | 0.577063 | False |
| 16 | 0.200059 | 0.200059 | 0.000000 | 0.000000 | 0.560000 | 0.700000 | 0.760000 | 0.585956 | True |
| 17 | 0.200056 | 0.200056 | 0.000000 | 0.000000 | 0.400000 | 0.740000 | 0.820000 | 0.491225 | False |
| 18 | 0.200051 | 0.200051 | 0.000000 | 0.000000 | 0.440000 | 0.740000 | 0.800000 | 0.519536 | False |
| 19 | 0.200046 | 0.200046 | 0.000000 | 0.000000 | 0.420000 | 0.740000 | 0.820000 | 0.517821 | False |
| 20 | 0.200038 | 0.200038 | 0.000000 | 0.000000 | 0.500000 | 0.680000 | 0.780000 | 0.560636 | False |

### A1-Triplet-Mean

| epoch | train_total_loss | train_triplet_loss | train_mean_loss | train_variance_loss | val_top1 | val_top3 | val_top5 | val_competition_score | best_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.311483 | 0.211733 | 0.997498 | 0.000000 | 0.340000 | 0.480000 | 0.580000 | 0.385921 | False |
| 2 | 0.301319 | 0.201322 | 0.999970 | 0.000000 | 0.320000 | 0.500000 | 0.700000 | 0.375684 | False |
| 3 | 0.300822 | 0.200823 | 0.999987 | 0.000000 | 0.340000 | 0.540000 | 0.740000 | 0.405804 | False |
| 4 | 0.300581 | 0.200582 | 0.999993107796 | 0.000000 | 0.360000 | 0.580000 | 0.680000 | 0.430701 | False |
| 5 | 0.300434 | 0.200434 | 0.999996088982 | 0.000000 | 0.340000 | 0.680000 | 0.820000 | 0.418372 | False |
| 6 | 0.300329 | 0.200330 | 0.999997546673 | 0.000000 | 0.380000 | 0.620000 | 0.700000 | 0.447761 | False |
| 7 | 0.300253 | 0.200253 | 0.999998355865 | 0.000000 | 0.440000 | 0.580000 | 0.680000 | 0.477249 | False |
| 8 | 0.300212 | 0.200212 | 0.999998815536 | 0.000000 | 0.400000 | 0.640000 | 0.740000 | 0.462847 | False |
| 9 | 0.300177 | 0.200177 | 0.999999142647 | 0.000000 | 0.500000 | 0.640000 | 0.720000 | 0.544925 | False |
| 10 | 0.300144 | 0.200144 | 0.999999341011 | 0.000000 | 0.460000 | 0.620000 | 0.820000 | 0.506438 | False |
| 11 | 0.300122 | 0.200122 | 0.999999454498 | 0.000000 | 0.420000 | 0.640000 | 0.780000 | 0.501364 | False |
| 12 | 0.300100 | 0.200100 | 0.999999541283 | 0.000000 | 0.480000 | 0.700000 | 0.780000 | 0.548446 | False |
| 13 | 0.300087 | 0.200087 | 0.999999610901 | 0.000000 | 0.440000 | 0.680000 | 0.780000 | 0.513407 | False |
| 14 | 0.300078 | 0.200078 | 0.999999656677 | 0.000000 | 0.460000 | 0.620000 | 0.700000 | 0.520955 | False |
| 15 | 0.300068 | 0.200069 | 0.999999698162 | 0.000000 | 0.480000 | 0.640000 | 0.700000 | 0.527266 | False |
| 16 | 0.300060 | 0.200060 | 0.999999740124 | 0.000000 | 0.360000 | 0.560000 | 0.660000 | 0.414041 | False |
| 17 | 0.300057 | 0.200057 | 0.999999765396 | 0.000000 | 0.460000 | 0.640000 | 0.720000 | 0.511598 | False |
| 18 | 0.300048 | 0.200048 | 0.999999783993 | 0.000000 | 0.520000 | 0.680000 | 0.720000 | 0.556968 | False |
| 19 | 0.300047 | 0.200047 | 0.999999788284 | 0.000000 | 0.540000 | 0.720000 | 0.780000 | 0.590602 | True |
| 20 | 0.300037 | 0.200037 | 0.999999810219 | 0.000000 | 0.480000 | 0.680000 | 0.860000 | 0.536136 | False |

### A2-Triplet-Variance

| epoch | train_total_loss | train_triplet_loss | train_mean_loss | train_variance_loss | val_top1 | val_top3 | val_top5 | val_competition_score | best_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.290267 | 0.212454 | 0.000000 | 0.077813 | 0.400000 | 0.540000 | 0.680000 | 0.442003 | False |
| 2 | 0.279733 | 0.201357 | 0.000000 | 0.078376 | 0.440000 | 0.560000 | 0.640000 | 0.478197 | False |
| 3 | 0.279237 | 0.200855 | 0.000000 | 0.078383 | 0.520000 | 0.680000 | 0.740000 | 0.566636 | False |
| 4 | 0.278980 | 0.200594 | 0.000000 | 0.078385 | 0.420000 | 0.560000 | 0.640000 | 0.455624 | False |
| 5 | 0.278839 | 0.200453 | 0.000000 | 0.078387 | 0.440000 | 0.640000 | 0.720000 | 0.501718 | False |
| 6 | 0.278736 | 0.200349 | 0.000000 | 0.078387 | 0.380000 | 0.700000 | 0.760000 | 0.472803 | False |
| 7 | 0.278672 | 0.200285 | 0.000000 | 0.078388 | 0.500000 | 0.700000 | 0.740000 | 0.560985 | False |
| 8 | 0.278619 | 0.200231 | 0.000000 | 0.078388 | 0.500000 | 0.660000 | 0.780000 | 0.558925 | False |
| 9 | 0.278582 | 0.200194 | 0.000000 | 0.078388 | 0.380000 | 0.660000 | 0.760000 | 0.462842 | False |
| 10 | 0.278544 | 0.200156 | 0.000000 | 0.078388 | 0.600000 | 0.680000 | 0.840000 | 0.634879 | True |
| 11 | 0.278521 | 0.200133 | 0.000000 | 0.078388 | 0.400000 | 0.560000 | 0.800000 | 0.445037 | False |
| 12 | 0.278501 | 0.200113 | 0.000000 | 0.078388 | 0.300000 | 0.620000 | 0.800000 | 0.406821 | False |
| 13 | 0.278486 | 0.200098 | 0.000000 | 0.078388 | 0.400000 | 0.700000 | 0.800000 | 0.482157 | False |
| 14 | 0.278471 | 0.200083 | 0.000000 | 0.078388 | 0.420000 | 0.680000 | 0.760000 | 0.509700 | False |
| 15 | 0.278464 | 0.200076 | 0.000000 | 0.078388 | 0.460000 | 0.700000 | 0.820000 | 0.525101 | False |
| 16 | 0.278455 | 0.200066 | 0.000000 | 0.078388 | 0.440000 | 0.640000 | 0.780000 | 0.499425 | False |
| 17 | 0.278450 | 0.200062 | 0.000000 | 0.078388 | 0.420000 | 0.800000 | 0.840000 | 0.533251 | False |
| 18 | 0.278441 | 0.200053 | 0.000000 | 0.078388 | 0.440000 | 0.720000 | 0.860000 | 0.520183 | False |
| 19 | 0.278440 | 0.200052 | 0.000000 | 0.078388 | 0.400000 | 0.760000 | 0.860000 | 0.493669 | False |
| 20 | 0.278429 | 0.200041 | 0.000000 | 0.078388 | 0.400000 | 0.740000 | 0.860000 | 0.500630 | False |

### A3-Triplet-MeanVariance

| epoch | train_total_loss | train_triplet_loss | train_mean_loss | train_variance_loss | val_top1 | val_top3 | val_top5 | val_competition_score | best_checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.390009 | 0.212482 | 0.997346 | 0.077793 | 0.400000 | 0.540000 | 0.640000 | 0.447180 | False |
| 2 | 0.379753 | 0.201381 | 0.999967 | 0.078375 | 0.240000 | 0.560000 | 0.680000 | 0.339183 | False |
| 3 | 0.379250 | 0.200869 | 0.999986 | 0.078383 | 0.380000 | 0.480000 | 0.620000 | 0.416590 | False |
| 4 | 0.378992 | 0.200608 | 0.999992237568 | 0.078385 | 0.440000 | 0.600000 | 0.680000 | 0.478330 | False |
| 5 | 0.378846 | 0.200460 | 0.999995384216 | 0.078387 | 0.320000 | 0.580000 | 0.680000 | 0.400765 | False |
| 6 | 0.378745 | 0.200358 | 0.999997116089 | 0.078387 | 0.320000 | 0.520000 | 0.660000 | 0.380684 | False |
| 7 | 0.378669 | 0.200282 | 0.999998053551 | 0.078388 | 0.380000 | 0.660000 | 0.780000 | 0.454536 | False |
| 8 | 0.378621 | 0.200233 | 0.999998619080 | 0.078388 | 0.460000 | 0.580000 | 0.680000 | 0.499193 | False |
| 9 | 0.378580 | 0.200193 | 0.999998990536 | 0.078388 | 0.440000 | 0.580000 | 0.720000 | 0.493193 | False |
| 10 | 0.378541 | 0.200153 | 0.999999238491 | 0.078388 | 0.400000 | 0.620000 | 0.740000 | 0.457114 | False |
| 11 | 0.378518 | 0.200130 | 0.999999377728 | 0.078388 | 0.440000 | 0.620000 | 0.740000 | 0.496352 | False |
| 12 | 0.378503 | 0.200115 | 0.999999492645 | 0.078388 | 0.520000 | 0.660000 | 0.740000 | 0.570206 | False |
| 13 | 0.378486 | 0.200098 | 0.999999576569 | 0.078388 | 0.460000 | 0.680000 | 0.780000 | 0.510838 | False |
| 14 | 0.378470 | 0.200082 | 0.999999634266 | 0.078388 | 0.520000 | 0.740000 | 0.800000 | 0.565455 | False |
| 15 | 0.378464 | 0.200076 | 0.999999680042 | 0.078388 | 0.540000 | 0.780000 | 0.840000 | 0.616696 | False |
| 16 | 0.378450 | 0.200062 | 0.999999720097 | 0.078388 | 0.520000 | 0.760000 | 0.860000 | 0.588127 | False |
| 17 | 0.378451 | 0.200063 | 0.999999749184 | 0.078388 | 0.540000 | 0.780000 | 0.840000 | 0.602744 | False |
| 18 | 0.378440 | 0.200052 | 0.999999780655 | 0.078388 | 0.540000 | 0.820000 | 0.860000 | 0.616463 | False |
| 19 | 0.378438 | 0.200049 | 0.999999784946 | 0.078388 | 0.600000 | 0.740000 | 0.820000 | 0.645555 | True |
| 20 | 0.378430 | 0.200042 | 0.999999796391 | 0.078388 | 0.440000 | 0.800000 | 0.860000 | 0.560842 | False |

### 10. 驗證、限制與下一步

完整 **143 tests passed**，failure/error/skipped=0，imports/syntax全部通過。涵蓋四loss等價／分解、常數、collapsed/spread vectors、full sourcepool weighting、finite gradients、TRAIN-only regularizer、歷史排除、相同split／初始化／cache、只依normal score選epoch與CLOSED paths拒絕。

本輪總wall-clock（preprocessing＋訓練＋diagnostics）**21.19 分鐘**。只有DEV3；不讀TEST分數選模型、不用DEV2或Stability selection。未建立FINAL TEST4，未修改Phase2.8／Phase2.10 artifacts。正式結果與各best checkpoints都有保存provenance。

本輪每種lambda只試固定值，只有一個seed、一組DEV3；50 questions的不確定性未以新TEST驗證。不能把本輪highest DEV候選稱為final model。

Feature preprocessing／Opening parsing皆無失敗。Opening reference有 1 個missing color bank，沿用固定 similarity=0、無跨色fallback；沒有因結果更換資料或設定。Protocol audit另驗證80 epochs、loss decomposition、共享初始化／split與6個歷史path拒絕。

下一步：先用新的DEV或多seed確認joint improvement，分開檢查色彩域與身份訊號；SupCon可作後續獨立受控比較，但本輪不實作。
