# 資料目錄

將官方下載的 `training/train_A.csv` 放在 `data/training/train_A.csv`。
也可修改 `configs/baseline.yaml` 的 `paths.training_csv`，例如改成 `training/train_A.csv`。
一次分析一個等級 CSV，可保留原始檔名；本專案不會自動下載百萬盤資料。

必要欄位：`player_id,game_id,rank,color,sgf_content`。
`color` 是目標玩家在該盤的 B/W 執色，SGF 必須保留完整字串。
請用 pandas 或 CSV 工具讀寫，避免自行用逗號分割 SGF 欄位。
原始資料、mock CSV 與輸出都不納入 Git。

`python -m src.create_mock_data` 會產生 `data/mock/train_A.csv`（純合成資料）。

Phase 2 仍只使用 `data/training/train_A.csv`，先將玩家分為互斥的 metric training 與 held-out evaluation。
`python -m src.create_metric_mock_data` 另產生 `data/mock/metric_train_A.csv`（10 位合成玩家，60 盤），不覆蓋 Phase 1 mock。
目前沒有正式 CSV；請從官方 Releases 下載、解壓後放到上述 training 路徑。

Phase 2.5 一鍵 mock 使用 `data/mock/phase25_train_A.csv`（14 位玩家、84 盤）；與 Phase 1/2 mock 分開。
正式 `real_quick.yaml` 仍只讀 `data/training/train_A.csv`。正式 quick 不會自動下載或以 mock 替代資料。
