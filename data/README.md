# 資料目錄

將官方下載的 `training/train_A.csv` 放在 `data/training/train_A.csv`。
也可修改 `configs/baseline.yaml` 的 `paths.training_csv`，例如改成 `training/train_A.csv`。
一次分析一個等級 CSV，可保留原始檔名；本專案不會自動下載百萬盤資料。

必要欄位：`player_id,game_id,rank,color,sgf_content`。
`color` 是目標玩家在該盤的 B/W 執色，SGF 必須保留完整字串。
請用 pandas 或 CSV 工具讀寫，避免自行用逗號分割 SGF 欄位。
原始資料、mock CSV 與輸出都不納入 Git。

`python -m src.create_mock_data` 會產生 `data/mock/train_A.csv`（純合成資料）。
