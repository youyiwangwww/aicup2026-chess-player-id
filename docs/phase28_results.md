# Phase 2.8：Opening-aware / Color-aware DEV2 結果

本輪最佳方法：**Opening+Triplet**，DEV2 competition score **0.758266**。只使用 DEV2 選方法、epoch 與 alpha；Round 1 TEST、FINAL TEST 2 保持 CLOSED，未讀取其 ground truth、未執行其 inference，也未建立 FINAL TEST 3。這是 validation 結果，不能宣稱新的 held-out TEST 效果。

## 會議重點

| 問題 | 本輪答案 |
| --- | --- |
| Total vs player window | total-10 與 player-5 相同；player-10／20 較弱 |
| Color-aware 是否改善 | player-5 分色 score 提高 0.103179 |
| 同色 vs 跨色 | 固定 54 人的 Opening 同色−跨色 score gap 為 0.754040 |
| Exposure 是否改善 | 200 人相較 30 人差 +0.055560；尚無同 DEV fixed-512 對照 |
| CrossColor robustness | 主 VAL score 差 -0.060931；配對色控 gap 從 0.535386 到 0.543701 |
| Collapse 是否解決 | 尚未解決；5 個實驗中 4 個觸發近零維度 warning，未觸發的 30-player cosine 也接近 1 |
| Fusion 是否超過 Opening | 最佳 alpha=0.9，score 差 +0.018698 |
| Error 是否互補 | Opening 錯的 31 題中，Triplet 單獨補對 9 題 |

## 資料與可重現性

來源：`data/training/train_A.csv`；SHA256 `0915826dfeadb32fa6fc5f5a08b1ef90a5ed0e3cc7868452e8158c887e3ccbca`。Seed 42。排除 Round 1 TEST 10 位、Phase 2.6 VAL 50 位與 FINAL TEST 2 的 50 位，共 110 位。最後一組身份只由封存 predictions 取得；receipt 只驗證封存狀態與玩家數，不拿舊 TEST metrics 做選擇。

| Partition | Players | Games/player | Total games |
| --- | --- | --- | --- |
| DEV2 TRAIN | 200 | 20 | 4000 |
| DEV2 VAL candidate | 100 | 30 | 3000 |
| DEV2 VAL query | 100 | 10 | 1000 |

TRAIN/VAL identity overlap = 0；與排除集合 identity overlap = 0；所有 TRAIN/candidate/query pair 的 game ID 與 SGF overlap = 0。原始 CSV 的 game ID 與 exact SGF 字串全域唯一，排除整位玩家也排除了其所有來源棋局。Queries 無 player_id、rank。完整 IDs、CSV SHA256 與 audit 保存在 `outputs/phase28/splits/`。

| partition | total_games | successful_games | failed_games | success_rate | mean_positions | median_positions | min_positions | max_positions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4000 | 4000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_candidate | 3000 | 3000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |
| val_query | 1000 | 1000 | 0 | 1.000000 | 16.000000 | 16.000000 | 16 | 16 |

## 1. Opening 窗口與黑白分開比對

total-N 計算整盤前 N 個 B/W move nodes；player-N 只計算目標玩家的前 N 次輪到落子。兩者都讓 pass 消耗窗口額度，但 pass 不加入 heatmap；setup 不消耗額度；只讀主變化。每盤 heatmap 等權平均，每位玩家/每題比較完整候選集合。沒有合法落點的棋譜記錄錯誤，不讓單一 SGF 停止流程。

Color-aware 將 candidate B/W 分開平均，query 的 B 僅比 B、W 僅比 W，再依該題原始 query B/W 棋局數加權；缺少某色 fingerprint 時該色貢獻為 0，保留全部候選人。未加入 rank、姓名等身份特徵。

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Opening-total-10 | 0.600000 | 0.720000 | 0.770000 | 0.636389 |
| Opening-player-5 | 0.600000 | 0.720000 | 0.770000 | 0.636389 |
| Opening-player-10 | 0.520000 | 0.670000 | 0.740000 | 0.558805 |
| Opening-player-20 | 0.420000 | 0.590000 | 0.650000 | 0.468619 |
| Color-aware-player-5 | 0.690000 | 0.840000 | 0.880000 | 0.739568 |
| Color-aware-player-10 | 0.690000 | 0.800000 | 0.860000 | 0.728174 |

total-10 與 player-5 差 **+0.000000**；一般輪流下棋時它們包含相同目標手數，本次結果相同。player-10／20 的結果見表，窗口變長不保證風格更易辨識。Color-aware-player-5 比非分色 player-5 改善 **+0.103179**；player-10 分色改善 **+0.169369**。

## 2. 相同 cohort 的同色／跨色控制

主 VAL 100 位中，只有 **54 位**同時有至少 40 盤 B 與 40 盤 W；因此明確縮小到這一固定 cohort，五組皆使用相同 IDs、每人 candidate 30/query 10。Mixed 固定 candidate 15B+15W、query 5B+5W。每組內 candidate/query 無 game/SGF overlap；不同控制條件可重用棋局，這是配對控制設計。使用主 VAL 事先選定的最佳**非分色 Opening**，避免跨色組本來就沒有同色 candidate bank 而得到機械式 0 分。

| condition | Opening-total-10 | Triplet-Hard-100 | Triplet-Hard-CrossColor | Triplet-Hard-Matched |
| --- | --- | --- | --- | --- |
| B_to_B | 0.876640 | 0.732449 | 0.797768 | 0.734253 |
| W_to_W | 0.768445 | 0.433631 | 0.468392 | 0.481532 |
| B_to_W | 0.047661 | 0.075450 | 0.038094 | 0.060903 |
| W_to_B | 0.089343 | 0.088178 | 0.140664 | 0.084111 |
| Mixed_to_Mixed | 0.747468 | 0.546362 | 0.527505 | 0.391485 |

| method | same_color_mean | cross_color_mean | gap |
| --- | --- | --- | --- |
| Opening-total-10 | 0.822542 | 0.068502 | 0.754040 |
| Triplet-Hard-Matched | 0.607893 | 0.072507 | 0.535386 |
| Triplet-Hard-CrossColor | 0.633080 | 0.089379 | 0.543701 |
| Triplet-Hard-100 | 0.583040 | 0.081814 | 0.501226 |

gap = 同色 B→B/W→W 平均 score − 跨色 B→W/W→B 平均 score。候選人數與身份組成已固定，因此這些差異支持棋色／棋局分布有影響；棋局內容仍有差異，不能直接宣稱純棋色的因果效果。這個小 cohort 的 score 也不能直接跟主 VAL 100 人混合比較。

## 3. Exposure-controlled Triplet

固定 T=20 **anchor triplets/player/epoch**，每個 epoch 驗證實際每人 exposure 的 min=max=20；30/100/200 人為 nested TRAIN 子集，20 games/player、16 positions/game、64 channels、8 blocks、128 embedding、batch-hard negative、最多 20 epochs，其餘參數一致。正樣本從同玩家不同棋局取位置，negative 在 batch pool 中選最近的其他 TRAIN 玩家；採樣與 mining 都不使用 VAL identity。

| experiment | train_players | samples_per_epoch | best_epoch | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Triplet-Hard-30 | 30 | 600 | 19 | 0.260000 | 0.450000 | 0.520000 | 0.316475 |
| Triplet-Hard-100 | 100 | 2000 | 18 | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| Triplet-Hard-200 | 200 | 4000 | 15 | 0.300000 | 0.530000 | 0.670000 | 0.372035 |

只依 DEV2 competition score 取最高 epoch，平手取較早 epoch。這修正了每人的 anchor exposure，但總 optimizer steps 隨人數上升，所以不能將差異完全歸因於玩家多樣性。沒有同一 DEV2、相同 GPU 環境下的 fixed-512 對照；不拿 Phase 2.6 不同 VAL 的 score 推論 exposure 改善。

## 4. Cross-color positive 的配對對照

父實驗 `Triplet-Hard-100` 共 100 位；至少各 2 盤 B/W 的實際 eligible TRAIN 玩家 **99 位**。Hard 與 CrossColor 使用完全相同 eligible IDs、棋局、每人 T=20 與其他參數；若父實驗有不合格身份，另訓練 matched Hard 對照。CrossColor 每位玩家每 epoch 固定 10 個 B-anchor/W-positive、10 個 W-anchor/B-positive，絕不退回同色 positive。

| experiment | train_players | samples_per_epoch | best_epoch | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Triplet-Hard-Matched | 99 | 1980 | 16 | 0.350000 | 0.530000 | 0.650000 | 0.409307 |
| Triplet-Hard-CrossColor | 99 | 1980 | 17 | 0.280000 | 0.500000 | 0.600000 | 0.348375 |

主 VAL CrossColor − matched Hard score = **-0.060931**。固定色控 cohort 的同色−跨色 gap：Hard **0.535386**、CrossColor **0.543701**；同時看跨色絕對 score 與 gap，不能只因 gap 變小就說模型更好（也可能同色退步）。

CrossColor 的配對 score 變化：B→W **-0.022809**、W→B **+0.056553**、Mixed→Mixed **+0.136020**。本次改善不一致：W→B/Mixed 較好，B→W 較差，平均同色−跨色 gap 也沒有縮小，因此尚無穩定降低棋色依賴的證據。

## 5. Embedding separation 與 collapse

診斷使用選定 epoch 的 DEV2 candidate/query **game embeddings**；先平均 positions，再 L2 normalize。Same/different 以 candidate-game/query-game 配對計算 cosine，std 使用 population std。Dimension-wise variance 在 3000 個單位 candidate-game embeddings 上計算；全部 128 維另存於各 experiment 的 `dimension_variance.csv`。

| experiment | pair_type | mean | median | std | p25 | p75 |
| --- | --- | --- | --- | --- | --- | --- |
| Triplet-Hard-30 | same | 0.999998747741 | 0.999998764516 | 2.41548e-07 | 0.999998601497 | 0.999998915387 |
| Triplet-Hard-30 | different | 0.999998722384 | 0.999998741305 | 2.38271e-07 | 0.999998576378 | 0.999998889656 |
| Triplet-Hard-100 | same | 0.999999875890 | 0.999999877136 | 2.39569e-08 | 0.999999860705 | 0.999999892648 |
| Triplet-Hard-100 | different | 0.999999872358 | 0.999999873997 | 2.36814e-08 | 0.999999857500 | 0.999999888995 |
| Triplet-Hard-200 | same | 0.999999929932 | 0.999999931519 | 1.61372e-08 | 0.999999920138 | 0.999999941181 |
| Triplet-Hard-200 | different | 0.999999926779 | 0.999999928439 | 1.63595e-08 | 0.999999917071 | 0.999999938343 |
| Triplet-Hard-Matched | same | 0.999999870159 | 0.999999871924 | 2.49647e-08 | 0.999999854732 | 0.999999887830 |
| Triplet-Hard-Matched | different | 0.999999866657 | 0.999999868564 | 2.46788e-08 | 0.999999851296 | 0.999999884061 |
| Triplet-Hard-CrossColor | same | 0.999999859629 | 0.999999861747 | 2.68467e-08 | 0.999999842938 | 0.999999878780 |
| Triplet-Hard-CrossColor | different | 0.999999856145 | 0.999999858264 | 2.68808e-08 | 0.999999839474 | 0.999999875134 |

| experiment | separation | variance_mean | variance_median | variance_min | variance_max | near_zero_fraction | collapse_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Triplet-Hard-30 | 2.53576e-08 | 1.00281e-08 | 9.84188e-09 | 7.90787e-09 | 1.30573e-08 | 0.539062 | False |
| Triplet-Hard-100 | 3.53153e-09 | 9.93763e-10 | 9.72628e-10 | 7.50571e-10 | 1.42557e-09 | 1.000000 | True |
| Triplet-Hard-200 | 3.15244e-09 | 5.72055e-10 | 5.61303e-10 | 3.79073e-10 | 9.19144e-10 | 1.000000 | True |
| Triplet-Hard-Matched | 3.5022e-09 | 1.04189e-09 | 1.03155e-09 | 8.0171e-10 | 1.37224e-09 | 1.000000 | True |
| Triplet-Hard-CrossColor | 3.48434e-09 | 1.12215e-09 | 1.1123e-09 | 8.93298e-10 | 1.38459e-09 | 1.000000 | True |

工程診斷門檻：variance < 1e-8 為近零，近零維度比例 ≥ 80% 發出 collapse warning。門檻已固定於 config，不作模型選擇依據；warning 是數值退化提示，不等同可證明所有玩家完全不可分。Same/different 很接近且 variance 小時，不能只看 Top-1 便宣稱學到穩定風格。

## 6. Fusion 與 error overlap

使用最佳 Opening `Color-aware-player-5` 與最佳 Triplet `Triplet-Hard-100`，對**每題全部 100 個 candidate scores**做 z-score，然後 `alpha*Opening+(1-alpha)*Triplet`。支援 min-max，但本輪固定 z-score，不另外搜尋 normalization。alpha=0/1 排名已驗證完全等於純 Triplet/Opening；Top-5 沒有重複 IDs。

| alpha | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| 0.100000 | 0.420000 | 0.610000 | 0.700000 | 0.479166 |
| 0.200000 | 0.550000 | 0.690000 | 0.740000 | 0.589410 |
| 0.300000 | 0.610000 | 0.770000 | 0.800000 | 0.653762 |
| 0.400000 | 0.660000 | 0.800000 | 0.830000 | 0.701370 |
| 0.500000 | 0.680000 | 0.810000 | 0.850000 | 0.722525 |
| 0.600000 | 0.690000 | 0.820000 | 0.850000 | 0.736363 |
| 0.700000 | 0.710000 | 0.830000 | 0.880000 | 0.753365 |
| 0.800000 | 0.710000 | 0.860000 | 0.900000 | 0.754602 |
| 0.900000 | 0.710000 | 0.850000 | 0.910000 | 0.758266 |
| 1.000000 | 0.690000 | 0.840000 | 0.880000 | 0.739568 |

最佳 alpha **0.9**，score **0.758266**；比純最佳 Opening 差 **+0.018698**。這是同一個 DEV2 上的模型與 alpha 搜尋，需新 held-out evaluation 才能確認泛化，不能把 validation 小幅提高視為已證實。

| category | count | proportion |
| --- | --- | --- |
| opening_only_correct | 40 | 0.400000 |
| triplet_only_correct | 9 | 0.090000 |
| both_correct | 29 | 0.290000 |
| both_wrong | 22 | 0.220000 |

Opening 的 31 個 Top-1 錯誤中，Triplet 單獨補對 9 個；兩者都錯 22 個。因此是否互補應以這些逐題錯誤與融合實際分數一起判讀，互補不保證加權分數一定改善。

## 選定設定、測試與下一步

`configs/phase28_selected.yaml` 已 frozen，依據只有 DEV2 competition score。選定 **Opening+Triplet**，score **0.758266**。若融合端點與純 Opening 平手，保留純 Opening；其他平手按預先固定順序／較小 alpha。記錄來源、split、Triplet checkpoint SHA256 與各窗口定義；純 Opening 選中時不需要載入 Triplet。

完整 tests **81 passed**，failure/error/skipped 均 0；所有 src Python imports 與 syntax check 通過。GPU `NVIDIA GeForce RTX 5060 Laptop GPU`，Torch `2.14.1+cu130`，deterministic=True。既有 encoder 結構與 675648 參數不變；CUDA deterministic adaptive pooling backward 的限制以數學等價的 floor/ceil window mean 實作處理，CPU/CUDA forward 與 gradient 已驗證，未修改歷史模型程式或封存 checkpoint。

正式 pipeline wall-clock 時間 **73.98 分鐘**，包含 preprocessing、訓練、DEV 評估與觀察到的時間波動，不能等同純 GPU compute time。完整逐 epoch logs 與 provenance 保存在 `outputs/phase28/experiments/`。測試時的 mock 只位於 `.test-tmp/`，不是正式 DEV2 結果。完整原始 metrics 在本報告同名 CSV。

| experiment | epoch_median_seconds | epoch_min_seconds | epoch_max_seconds | epochs_total_seconds |
| --- | --- | --- | --- | --- |
| Triplet-Hard-30 | 30.333901 | 12.874157 | 51.046161 | 635.168535 |
| Triplet-Hard-100 | 14.859993 | 14.630652 | 15.218803 | 297.709068 |
| Triplet-Hard-200 | 18.078773 | 17.220520 | 82.689381 | 776.240734 |
| Triplet-Hard-Matched | 19.250537 | 18.144916 | 64.063538 | 612.823028 |
| Triplet-Hard-CrossColor | 20.221532 | 18.219301 | 969.185960 | 1390.119982 |

時間警示：部分 epochs 明顯較慢，最長 **969.2 秒**；log 未顯示對應訓練例外，原因未經確認，因此只記錄 wall-clock 波動，沒有因此重跑、改 seed、換 split 或改參數。

下一步先確認 DEV2 選擇是否穩定，再決定一次性 FINAL TEST 3；本輪沒有建立其 split 或執行其 inference。建議：

1. 在獨立 DEV identities／預先固定多個 seeds 上確認 color-aware Opening 與選定 alpha 的穩定性，避免同一 VAL 的多次選擇偏差。
2. 另做相同 DEV、matched optimizer steps 的 fixed-512 vs T=20 對照，釐清總更新數與每人 exposure 的作用。
3. 針對 collapse、同色／跨色差距及互補錯誤進行資料與 embedding 分析，再考慮後續模型研究；尚未加入 Strength Estimator、MiniZero 或其他 Phase 3 元件。

重跑與報告指令（僅 DEV2；已完成的相同實驗會驗證 provenance 後重用 checkpoint，不重訓）：

```powershell
python scripts/check_phase28.py
python scripts/check_phase28_cuda.py
python -m src.run_phase28 --config configs/phase28.yaml
python scripts/report_phase28.py
```
