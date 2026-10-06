# Phase 2.6：Triplet Baseline Diagnosis

此輪只用 DEV VALIDATION 做控制變因實驗，不讀取或評估 FINAL TEST 2，不使用 CLOSED Round 1 TEST 調參。Round 1 原始 artifacts 保留，另存 SHA-256 備份於 `outputs/round1_archive/`，歷史結果見 `docs/round1_results.md`。

## 執行

```powershell
python scripts/check_phase26.py
python -m src.run_phase26 --config configs/phase26.yaml
```

Runner 建立固定 split、DEV caches、Opening windows、Triplet reference、A/B/C/hard ablations、cross-color 與 embedding diagnostics。各階段與 checkpoint 寫入 `outputs/phase26/`；已完成的實驗可以沿用，不會重新執行 TEST。訓練中断的實驗不會被誤判為完成，重啟會從該實驗的初始 seed 訓練。FINAL TEST 2 不提供自動執行選項。

## 固定資料與控制變因

- 排除 Round 1 TEST 的 10 位 CLOSED 玩家，重新抽取 200 DEV TRAIN／50 DEV VAL／50 FINAL TEST 2 玩家，身份完全互斥。
- DEV TRAIN 每人 20 盤；DEV VAL 每人 candidate 20／query 10；FINAL TEST 2 每人 candidate 40／query 10。人數不足明確報錯。
- split audit 包含 3 項身份、10 項 game_id、10 項 SGF 字串交集。Query CSV 不含玩家身份。
- A1/A2/A3 使用同一固定 TRAIN pool 的巢狀子集：30×10／100×20／200×20。A1→A2 同時改變玩家數和每人棋譜數，不能把結果完全歸因於身份數；A2→A3 固定每人棋譜數。
- A：10 epochs、8 positions、64 channels／8 blocks／128 dim。B：固定 A 的 DEV 最佳資料量，只比較 4／8／16 positions。8-position 組沿用 A，不重新抽資料或重跑。
- C：固定 B 的 DEV 最佳設定，訓練 20 epochs，依最高 DEV score 選 checkpoint，分數相同保留較早 epoch。
- Hard：同 C 的所有設定與 sampling；在同一 batch 的 anchor/positive/random-negative embedding pool 中挑選與 anchor 距離最近且 TRAIN 玩家不同的 negative，positive 保持同玩家不同棋局。
- 所有 triplet 組固定 batch 16、512 triplets/epoch、AdamW lr 0.001、margin 0.2、seed 42。資料量變大不增加更新步數，因此每位玩家平均曝光次數會減少；這也是解讀限制。
- 另將 Phase 2.5 的 30×10、3 epochs、4 positions、32 channels／4 blocks／64 dim 設定，重跑在相同新 DEV VAL。改善幅度只與此 reference 比較，不與 CLOSED TEST 分數直接相減。

## 隔離與計算效率

`dev_only()` 使用 Python audit hook，涵蓋 builtins/io/pathlib/pandas/os 的实际檔案 open，阻擋 FINAL TEST 2 與 CLOSED TEST paths。三個 DEV feature partitions 的 preprocessing 使用獨立函式，不呼叫會讀 TEST truth 的 Phase 2.5 `preprocess_all()`。FINAL TEST 2 preprocessing 暫不執行。

初次 split 建立階段曾重開新產生的 FINAL TEST 2 truth 檔案來計算 provenance SHA-256；未解析玩家對應、未評分，也未作模型選擇。已修正為在記憶體中對輸出 CSV bytes 計算 hash；tuning 各步驟皆由讀取防護隔離，沒有開啟該 truth 檔。

DEV features 以 uint8 載入記憶體，推論跨棋局 batch，但保持盤內平均→跨盤等權平均→正規化的原本 aggregation。測試確認與既有 aggregation 一致，不修改既有 Train / Validation / Test 架構或模型類別。

## 輸出

- `outputs/phase26/dev_results.csv`：所有 Opening 與 Triplet 組的設定、DEV Top-1/3/5、Score。
- `outputs/phase26/opening_signal.csv`：10／20／40／60／100 total moves 與 full-game heatmap。
- `outputs/phase26/experiments/*/training_log.csv`：逐 epoch loss、DEV metrics、best 標記。
- `outputs/phase26/cross_color_results.csv`：Opening-40 與選定 Triplet 的 B→W／W→B。
- `outputs/phase26/cross_color_audit.json`：DEV VAL 中有足夠黑白棋譜的玩家子集與固定數量；不夠時明確回報不可用，不降低每人棋譜門檻。
- `outputs/phase26/embedding_similarity.csv`：所有 DEV candidate/query 棋局 pairs 的同／不同玩家 cosine 統計。
- `outputs/phase26/embedding_diagnostics.json`：mean separation、IQR overlap 與描述性標準化差值；棋局 pairs 不獨立，不作顯著性宣稱。
- `configs/phase26_selected.yaml`：只依 DEV score 選出的設定與 checkpoint、選擇原因、相對同 DEV reference 改善。
- `outputs/phase26/selected/best.pt`：依完成狀態與 SHA-256 驗證後複製的選定 weights，獨立於各實驗的 working checkpoint。

Opening 的不同窗口與 full-game 比較是診斷證據，不足以證明風格訊號的因果來源。Cross-color 使用新 DEV-only 棋局，雙向沿用相同符合條件的玩家；與原 DEV VAL 候選數可能不同，不能直接比較其絕對分數。

選定設定後停止，FINAL TEST 2 須由使用者另行明確授權一次評估。
