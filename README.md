# AI CUP 2026 Go Player Identification

**目前建議使用 Phase 2.5 的 Train / Validation / Test 實驗流程。** Phase 2 的 held-out evaluation 是用來選 checkpoint 的 validation，不能當成最終 test。下方保留 Phase 1/2 操作說明作為歷史對照；新實驗方式詳見 [docs/phase25.md](docs/phase25.md)。

已安裝依賴後，正式資料放在 `data/training/train_A.csv`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_real_quick.ps1
```

目前機器無一般可用的 Python 別名，可以先用現有暫存 Python 跑獨立 mock：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_real_quick.ps1 -Mock -Python "C:\Users\abbywang\.codex\tmp\aicup-python\python.exe"
```

這個 mock 僅驗證流程，不是真實模型效果。正式資料缺少時會印出 `Please place the official train_A.csv at data/training/train_A.csv` 並停止，不下載資料、不產生 real result。

Phase 2.5 新增 `configs/real_quick.yaml`、`configs/phase25_mock.yaml`、final-test evaluation、Random baseline、五組 feature coverage、實驗 metadata 和手動 multi-seed 工具。三個 baseline 使用完全相同的 TEST CSV；TRAIN/VAL/TEST 身份互斥，23 項 player/game/SGF overlap 都須為 0。42 項 tests 已通過（包含原有 26 項）；訓練時的 TEST 存取、final test 執行時序與一次評估規則都有測試。

手動執行三個種子（需已安裝一般 Python/venv 並具備正式資料）：

```powershell
python scripts/run_multi_seed.py --seeds 42 123 2026
```

輸出隔離到 `outputs/seed_42/`、`seed_123/`、`seed_2026/`，彙整至 `outputs/results/multi_seed_summary.csv`；`seed=mean/std` 是彙總列，std 採 sample std（ddof=1）。Quick 預設僅 seed 42，不會自動觸發 multi-seed。

新實驗輸出在 `outputs/splits/`、`outputs/results/`；mock 全部放在 `outputs/phase25_mock/`。正式資料目前不存在，因此沒有 real metrics 或 real multi-seed 結果。教授報告已更新為三層架構：[docs/professor_update.md](docs/professor_update.md)。

本專案研究「根據多盤圍棋棋譜，找出下棋的玩家」。每題提供同一玩家的多盤 query，與已知 player_id 的 candidate 棋譜庫比較，輸出 Top-5 玩家。內容是 **Go 圍棋**，即使遠端 repository 名稱含 chess，也不代表西洋棋。

| 方法 | 特徵與比較方式 | 是否訓練 |
| --- | --- | --- |
| Baseline 0：Opening Fingerprint | 整盤前 40 手中目標玩家的 19×19 落子 heatmap，多盤平均後以 cosine similarity 比較 | 否 |
| Baseline 1：Triplet Player Embedding | 目標玩家落子前 17-channel 盤面 history，經 ResNet、跨棋局 Triplet Loss 訓練，再做 cosine retrieval | 是 |

Opening 原始程式、設定與 Phase 1 mock 保留。Phase 2 不包含 Strength Estimator、MiniZero、rank auxiliary loss 或 policy fingerprint。

## 結構與每個 script 的用途

```text
aicup2026-go-player-id/
├── README.md
├── requirements.txt                 # 基本依賴
├── requirements-triplet.txt         # 基本依賴 + PyTorch
├── configs/
│   ├── baseline.yaml / mock.yaml    # 原始 Opening 與 mock
│   ├── triplet_quick.yaml           # 真實資料 quick
│   ├── triplet_baseline.yaml        # 較正式的訓練設定
│   └── triplet_mock.yaml            # CPU 流程驗證
├── data/README.md
├── docs/phase2.md
├── docs/professor_update.md
├── scripts/run_phase2_quick.ps1
├── src/
├── tests/
└── outputs/
```

| 模組 | 用途 |
| --- | --- |
| `analyze_dataset.py` | 棋譜數、玩家數、每人盤數、B/W 比例與分布 |
| `build_validation.py` | Phase 1 的同玩家 candidate/query 切分 |
| `baseline.py` | 原始 Opening Fingerprint 推論 |
| `evaluate.py` | 共用 Top-1/3/5 與 competition score |
| `create_mock_data.py` | 原始 48 盤 Phase 1 mock |
| `build_metric_split.py` | 互斥 training/held-out players、game/SGF audit |
| `player_features.py` | 落子前 history、固定數量抽樣、壓縮 shards |
| `train_triplet.py` | 訓練、每 epoch retrieval validation、checkpoint |
| `embed_players.py` | 最佳模型、game/group 平均、Top-5 推論 |
| `compare_baselines.py` | 同一 held-out split 比較兩個 baseline |
| `create_metric_mock_data.py` | 60 盤獨立 Phase 2 mock |
| `sgf_parser.py`, `opening_features.py` | 原始 SGF 落子解析與 heatmap，可 import |
| `player_model.py`, `triplet_dataset.py` | ResNet Encoder 與跨 game triplets，可 import |
| `feature_cache.py` | 限量 LRU shard cache，可 import |
| `utils.py`, `metric_utils.py` | YAML/CSV/JSON、路徑、seed/device、雜湊，可 import |

## 安裝：Windows PowerShell

需要 Python 3.10+。若 `python --version` 無法執行，先到 [Python 官方網站](https://www.python.org/downloads/) 安裝一般 Python，勾選 Add Python to PATH，再重新開啟終端機。Windows Store 的 python 別名可能不是可用的 Python。

```powershell
cd "C:\Users\abbywang\比賽\ai cup\aicup2026-go-player-id"
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

若 PowerShell 不允許 Activate.ps1，可用 `.\.venv\Scripts\python.exe` 取代以下的 `python`，不必改動系統政策。

只跑 Opening：`python -m pip install -r requirements.txt`。

Phase 2 先用 CPU 安裝：

```powershell
python -m pip install "torch>=2.6,<3.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-triplet.txt
python -m pip check
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

GPU 版本請依 [PyTorch 官方安裝選擇器](https://pytorch.org/get-started/locally/) 的 Windows／pip／相容 CUDA 指令安裝，再安裝 requirements。`device: auto` 只有在目前 torch 能使用 CUDA 時才選 GPU；CPU build 不會因為有 GPU 就自動變成 CUDA build。

macOS/Linux 使用 `python3 -m venv .venv`、`source .venv/bin/activate`，然後執行相同 Python 模組命令。以下命令從專案根目錄執行，使用 `python -m src.模組` 以支援相對 import。

## 資料格式與放置位置

從 [官方資料集 Releases](https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases) 下載、解壓，把 `training/train_A.csv` 放在 **`data/training/train_A.csv`**。本階段一次只處理 A rank group，不混合其他等級。

| 必要 CSV 欄位 | 意義 |
| --- | --- |
| player_id | 匿名玩家 ID，字串讀入並保留前導零 |
| game_id | 棋譜 ID |
| rank | 原始棋力標籤；不作為 Phase 2 label 或 feature |
| color | 目標玩家執 B 或 W |
| sgf_content | 完整 SGF 字串，使用標準 CSV quoting |

輸入路徑由 YAML 的 `paths.training_csv` 決定。設定內的資料路徑相對於專案根目錄，`--config` 相對路徑則相對於命令執行目錄。

## Validation 設計

Phase 1 從同一批玩家切 candidate/query，供無訓練的 Opening 使用。Phase 2 先切玩家身份：

```text
train_A.csv
├── metric-training players → training games → Triplet gradients
└── held-out players
    ├── candidate games → 玩家 embedding 資料庫
    └── query games → 題目 embedding → Top-5 → ground truth 評估
```

Training 與 evaluation player IDs 必須完全不同，三組資料每一對都檢查 game_id 與 SGF 字串交集，發現 leakage 直接報錯，不會自動略過或減少要求的玩家數。每位訓練玩家內先去除重複棋譜，選出的切分內也禁止重複 game/SGF。

Query 不含 player_id 或 rank。Ground truth 只用於切分稽核與 held-out 評分，不進入模型、Triplet Dataset、loss 或梯度。Validation 用於選 checkpoint，因此不能當作獨立最終 test 成績。

SGF identity 是完整字串相等，不是忽略註解的 canonical identity。Game IDs 在不同官方等級 CSV 可能重複，所以此階段只讀一個檔案。

## 保留的 Phase 1 流程

```powershell
python -m src.analyze_dataset
python -m src.build_validation
python -m src.baseline
python -m src.evaluate
```

預設 `configs/baseline.yaml`：50 位玩家，每人 40 盤 candidate、10 盤 query、seed 42。可調整：

```powershell
python -m src.build_validation --num-players 10 --candidate-games 20 --query-games 5 --seed 42
```

統計輸出 `dataset_statistics.csv`，`section=summary` 是總覽、`player_counts` 是每人盤數、`distribution` 的 key 是盤數、value 是該盤數的玩家數；B/W 比例以所有 CSV 列為分母，另報未知執色比例。

Opening 取整盤前 N 手（雙方手數），只計目標執色落子。Pass 計入手數但不增加 heatmap，設置子不計入手數。每組有效棋譜做算術平均，再 flatten 為 361 維。

原始 Phase 1 mock：

```powershell
python -m src.create_mock_data
python -m src.analyze_dataset --config configs/mock.yaml
python -m src.build_validation --config configs/mock.yaml
python -m src.baseline --config configs/mock.yaml
python -m src.evaluate --config configs/mock.yaml
```

## Phase 2 一鍵執行

有正式資料時：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2_quick.ps1
```

沒有正式資料時：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2_quick.ps1 -Mock
```

腳本優先使用專案 `.venv\Scripts\python.exe`，可用 `-Python "C:\path\to\python.exe"` 指定其他 interpreter。任何步驟失敗就停止並保留錯誤，不會繼續使用舊模型。

Mock 自動產生 `data/mock/metric_train_A.csv`（10 位玩家、60 盤），4 位 training players、6 位 unseen evaluation players，每人 candidate 3 盤、query 2 盤。輸出放在 `outputs/phase2_mock/`，不覆蓋 Phase 1 mock。

本機驗證使用已安裝依賴的暫存 Python，因為一般 python 別名無法執行。目前這台機器可直接跑：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2_quick.ps1 -Mock -Python "C:\Users\abbywang\.codex\tmp\aicup-python\python.exe"
```

正式開發請依安裝章節建立一般 Python 與 `.venv`。

## Phase 2 逐步執行與設定

每一步使用同一份設定：

```powershell
python -m src.build_metric_split --config configs/triplet_quick.yaml
python -m src.player_features --config configs/triplet_quick.yaml
python -m src.train_triplet --config configs/triplet_quick.yaml
python -m src.embed_players --config configs/triplet_quick.yaml
python -m src.evaluate --config configs/triplet_quick.yaml
python -m src.compare_baselines --config configs/triplet_quick.yaml
```

比較腳本自動在相同 held-out split 執行 Opening；不要把另一批 Phase 1 玩家成績直接拿來比。正式配置的完整流程：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2_quick.ps1 -Config configs/triplet_baseline.yaml
```

| 設定 | Quick | 正式 | Mock |
| --- | --- | --- | --- |
| training / eval players | 30 / 10 | 100 / 50 | 4 / 6 |
| training games/player | 10 | 20 | 4 |
| candidate / query games/player | 10 / 5 | 40 / 10 | 3 / 2 |
| history / positions per game | 8 / 4 | 8 / 8 | 8 / 4 |
| channels / blocks / embedding dim | 32 / 4 / 64 | 64 / 8 / 128 | 16 / 2 / 32 |
| batch size / epochs | 16 / 2 | 64 / 20 | 8 / 2 |
| triplets per epoch | 256 | 4096 | 32 |

Learning rate=0.001，TripletMarginLoss margin=0.2、p=2，AdamW。`samples_per_epoch` 是抽樣 triplets 數，不是全部 position 組合。`num_workers=0` 適合 Windows，`device=auto` 自動偵測 CUDA，mock 使用 CPU。

Quick 與正式都寫到 `outputs/`，換設定會重建該輪 split/cache/checkpoint；保留多個實驗時請複製 YAML 並修改全部 output 路徑。

17-channel 特徵是 8 層目標玩家 stones、8 層對手、一層執色（黑=1、白=0）。History 由舊到新，最後是落子前狀態，早期左側補空盤。SGF 使用主變化，處理設置子、提子、pass；每盤最多固定抽 K 個回合。

ResNet 使用 3×3 spatial pooling 保留粗略區域偏好、linear projection 產生 embedding，最後 `F.normalize(..., p=2, dim=1)`，沒有分類頭。Anchor/positive 同玩家不同 game，negative 不同玩家。

每盤先平均 position embeddings，再平均多盤 game embeddings，最後 L2 normalize；各盤等權，以 cosine 排 Top-5，同分用 player_id 字串順序固定處理。

## Output 與錯誤處理

以下是 Phase 2 預設路徑，mock 在 `outputs/phase2_mock/` 有相同內容：

| `outputs/` 檔案 | 內容 |
| --- | --- |
| `metric_train.csv` | training players 棋譜，不含 rank |
| `metric_candidates.csv` | 已知 held-out player_id 的候選棋譜 |
| `metric_queries.csv` | game_id,color,sgf_content,question_id；每題多盤、不含身份 |
| `metric_ground_truth.csv` | question_id,player_id；僅稽核/評估 |
| `split_audit.json` | seed、玩家/盤數、8 項 overlap count、CSV SHA-256 |
| `features/*.npz`, `features/*.json` | uint8 壓縮特徵、game offsets、manifest |
| `feature_errors.csv` | 棋譜解析/重播錯誤與跳過原因 |
| `checkpoints/last.pt`, `best.pt` | 最後 epoch／最高 held-out score 模型 |
| `triplet_training_log.csv` | loss、learning rate、時間、validation metrics |
| `triplet_predictions.csv` | question_id,rank,player_id,similarity |
| `triplet_predictions.metadata.json` | 預測與來源 hash，防止比較過期結果 |
| `player_embeddings.npz` | candidate/query unit embeddings |
| `triplet_metrics.csv`, `triplet_evaluation.csv` | 整體與每題 Triplet 評估 |
| `metric_opening_*.csv` | 同一 split 的 Opening 預測、評估、錯誤 |
| `baseline_comparison.csv` | method,top1,top3,top5,competition_score |

Shards 每檔最多 `games_per_shard` 盤，LRU 最多保留 `max_cached_shards` 檔。CSV、特徵設定或 seed 改變須重新 preprocessing，搬動專案也請重建 cache。舊 shards 不自動刪除。

單一壞棋譜會記錄並跳過。訓練玩家剩不到兩盤有效棋譜時明確列出並排除；有效訓練玩家少於兩人則報錯。任一 partition 完全沒有有效特徵也會報錯。個別題目無有效棋譜、不產生預測時，evaluate 仍算進分母並給 0，不只評估成功題目。

正確玩家排名 r=1..5 得 `exp(-(r-1))`，未進前五為 0，平均全部 ground-truth 題目。`correct_rank=0` 表示未命中。

Best 由 held-out score 決定，同分保留較早 epoch；training loss 不參與最佳模型選擇。Training 重跑從 seed 重新訓練，目前沒有 resume。預測是本地 long-form CSV，**不是官方提交格式**；尚未直接讀官方 wide-form query。資料/cache/checkpoint/output 不納入 Git。

## 檢查與目前結果

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m pip check
```

2026-10-05，Python 3.12.10、PyTorch 2.14.1+cpu：26 項 tests（含原有 9 項）通過，import/syntax/pip check 通過，Windows 一鍵 mock 流程成功。涵蓋黑白回合、落子前狀態、history/padding、setup/提子/pass、所有跨組 leakage、triplet 關係、unit embeddings、等權 game 聚合、checkpoint 選擇與 Top-5 不重複。

同一組 6 位 held-out mock 玩家：

| 方法 | Top-1 | Top-3 | Top-5 | Competition score |
| --- | --- | --- | --- | --- |
| Opening | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Triplet | 0.166667 | 0.833333 | 1.000000 | 0.342703 |

Mock loss 由 0.214364 降到 0.097343，retrieval score 未改善，best 保留 epoch 1。這只驗證流程，不能當真實競賽效果。正式 `data/training/train_A.csv` 目前不存在，沒有真實成績。

紀錄：`outputs/phase2_mock/verification.json`。概念：[docs/phase2.md](docs/phase2.md)；教授報告：[docs/professor_update.md](docs/professor_update.md)。

## 已知限制與下一步

- 隨機 triplets，尚未使用 hard/semi-hard negative mining。
- 少量 position、簡單 ResNet 與平均聚合，不能保證超過 Opening。
- sgfmill 處理提子，但不做完整 ko/superko 裁判驗證。
- SGF 字串去重不能辨識改過註解的相同棋局。
- 單一 validation seed 可能偶然；取得正式資料後應做多 seed、更多 unseen players、位置數與模型大小消融，保留獨立最終 test players。
- CSV 仍一次載入記憶體；特徵已分 shard，尚無分批 CSV 讀取。
- 固定 seed 的重現性限相同硬體/套件環境，不保證跨平台數值完全一致。

Phase 3 預計研究 Strength Estimator 整合，先以對照實驗確認棋力資訊是否有助於風格辨識。

## 參考

- [官方 Tutorial](https://github.com/AILAB-NDHU/AICup-2026-Tutorial)
- [官方 Player Identification features](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/main/utils.py)
- [官方訓練資料](https://github.com/AILAB-NDHU/aicup2026-go-dataset)
- [PyTorch TripletMarginLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [Strength Estimator](https://github.com/rlglab/strength-estimator)：Phase 3 參考，目前未整合。
