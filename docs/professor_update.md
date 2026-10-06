# Go Player Identification：教授會議更新

## Round 1（CLOSED TEST）

正式資料 100,000 games／1,582 players；TRAIN／VAL／TEST players = 30／10／10，identity 不重疊。Random Score 0.019028，Opening 0.715343，Triplet 0.230301。此 TEST 已關閉，不再調參。

## Phase 2.6 diagnosis（DEV ONLY）

新 split：DEV TRAIN／DEV VAL／FINAL TEST 2 players = 200／50／50，三組玩家完全不同；game_id、SGF 亦無交集，排除 Round 1 CLOSED TEST 玩家。

Opening 前 10 手最佳 DEV Score 0.663844，全局 heatmap 0.143219，顯示開局偏好值得優先研究。

相同設定下，Hard Negative DEV Score 0.376701 高於 Random Negative 0.186007，最佳 epoch 17；同 DEV Phase 2.5 reference 0.133546。增加資料未改善，但固定 512 triplets/epoch 有曝光次數混淆。

選定模型實際使用 master TRAIN 中的 30 players × 10 games、16 positions/game、64 channels／8 blocks／128 dimensions；並非使用全部 200 位 TRAIN 玩家。

Embedding 有 collapse 警訊：same／different cosine 都約 0.999992，separation 僅 2.05e-7，分布高度重疊。Cross-color 的 34-player Score 僅 0.044–0.098，需要配對控制。

## FINAL TEST 2（ONE-SHOT，已 CLOSED）

| Method | Top-1 | Top-3 | Top-5 | Score |
|---|---:|---:|---:|---:|
| random | 0.0000 | 0.0800 | 0.1000 | 0.016473 |
| opening10 | 0.7400 | 0.8800 | 0.9200 | 0.783563 |
| triplet_hard | 0.3800 | 0.6400 | 0.8000 | 0.459101 |

固定 H20-Hard epoch 17，沒有訓練或修改 frozen artifacts。Triplet TEST−DEV = +0.082401；DEV／TEST 方法排序一致：True。

Triplet 比 Random 高 0.442628。最佳 TEST 方法為 opening10。Opening − Triplet 差距相較 DEV 擴大 0.037319。

## 限制與下一步研究問題

單 seed、50 queries、DEV 多組比較，且 DEV／TEST candidate games 不同。CPU tests 全通過；獨立 CUDA 訓練的 deterministic pooling 限制未放寬。

暫不進 Phase 3。下一輪在新 DEV 上研究：等每玩家曝光下的 training identities 數量、固定 nested 開局／全局 positions、同一玩家集合的同色／跨色 robustness。兩個 CLOSED TEST 都不得再作調參依據。
