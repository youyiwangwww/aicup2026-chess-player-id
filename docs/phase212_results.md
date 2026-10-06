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
