# Phase 3.0：三個來源的靜態稽核

稽核日期：2026-10-07。僅讀原始碼與 GitHub tree/releases metadata，沒有執行外部程式、下載權重、build、training 或正式 inference。來源固定為下列 commit；本地下載副本在 ignored `external_refs/`，可追溯摘要在 `outputs/phase30/source_audit.json`。

| 來源 | branch／commit | tree entries | releases |
|---|---|---:|---:|
| Official | main／359e06fcaf811f9e87179f12f71425dd18dbabd1 | 7 | 0 |
| Strength | iclr2025／d9498964e2ba84b5798c40e3f5b8a91a8791d66c | 241 | 0 |
| Policy | policy／b44b70f53de6cdaa7e2f44a590d541492148a9ad | 157 | 0 |

## Official Player Identification

輸入每個 position 為 `(17,19,19)`：target-relative own/opponent 各八個 history planes，另加 target 為黑時全1、白時全0的 color plane。history planes 分組排列；不能只因 shape 相同就假設與其他模型的交錯排列相容。parser 在目標方落子前抽 history，跳過 pass，要求完整 history；沒有處理 setup stones 的完整保證。不是直接把一整盤 SGF當作單一 feature。[utils.py](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/359e06fcaf811f9e87179f12f71425dd18dbabd1/utils.py)

實際 class defaults 是128 channels、16 residual blocks、256 embedding。global average pooling 1×1 後 MLP 128→64→256，最後 L2 normalize；docstring 的 embedding 敘述不能取代程式 default。loss 是 margin0.5 triplet margin loss；random positive 為同玩家不同盤、negative 為不同玩家，每盤再隨機取 position。AdamW lr1e-4，batch100、5epochs、每epoch15000steps，全部 rank training CSV 合併；best 依 training loss 選擇，notebook沒有本專案 identity-disjoint VAL checkpoint protocol。[network.py](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/359e06fcaf811f9e87179f12f71425dd18dbabd1/network.py)、[notebook](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/359e06fcaf811f9e87179f12f71425dd18dbabd1/player-identification.ipynb)

candidate/query 都把所有 games 的所有 window embeddings 加總再除 window 數，所以長盤權重大。平均後沒有再次 L2 normalize，以 Euclidean distance 排名。zero-window query 有直接取前五 candidate 的 fallback，不能當作可靠識別。本專案則限制每盤 positions、game→player 平均、cosine retrieval；保留明確 coverage audit。

官方 notebook public score **0.2845**是來源自身報告的歷史值，資料／候選數／protocol 與 DEV3不同，不可比較，也不是 Phase3.0 新結果。正式格式為 wide question row：question_id、num_games、sgf_i/color_i；candidate games 為每位100盤、候選400位、query每題5–20盤。自己的 long-form DEV split 不應為了配合 notebook而重構。多 rank data 的 game_id只在rank內唯一，未來應使用 `(rank,game_id)` composite key。[官方規格](https://github.com/AILAB-NDHU/AICup-2026-Tutorial/blob/359e06fcaf811f9e87179f12f71425dd18dbabd1/README.md)

| 面向 | Current frozen Triplet | Official |
|---|---|---|
| feature | own8＋opponent8＋targetcolor，早期zero-padding、pass計history、setup處理 | 同grouped17 planes，但跳pass、滿8history才取window、缺setup處理 |
| model | 64ch／8blocks／128embedding／3×3pool | 128ch／16blocks／256embedding／1×1pool+MLP |
| sampling | max16 positions/game，TRAIN100players×20games | 全 rank、多 window、random positions |
| triplets | batch hard，2000triplets/epoch，20epochs | random，batch100×15000steps×5epochs |
| aggregation | fixed sample→game→player，equal game | 所有windows一起平均，position-weighted |
| selection | identity-disjoint VAL competition score | training loss |
| retrieval | normalized prototype cosine | unnormalized mean Euclidean |

因此目前實作不是 official reproduction；同shape、同triplet名稱不能排除 sampling、aggregation、容量和資料量差異。官方環境範例 Ubuntu22.04/Python3.10/Torch2.6 cu124；multiprocessing fork/NPROC8須調整Windows，不應套用到 frozen runner。

## Strength Estimator

核心問題是從不同棋力族群的局面學習可比較的 strength ranking，不是 player identity supervision。Go features 是18×19×19（八步own/opponent交錯＋黑白turn planes）；`calculateFeatures`直接轉交 environment loader，沒有把 action one-hot自行追加到18planes。loader 的前後手邊界仍需未來smoke逐步驗證。[game_wrapper.cpp](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/misc/game_wrapper.cpp)、[Go features](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/minizero/minizero/environment/go/go.cpp)

Go configs 為256channels、20blocks；SL baseline 是 policy/value AlphaZero。RankNetwork額外輸出rank logits及probabilities；BTNetwork輸出policy logits/probabilities、value、**unbounded scalar score**與sigmoid weight，沒有直接回傳 player embedding。SE可用weight做position aggregation；SE_infinity設定加入non-human/random-move最低層族群、關閉weight，提供額外低端參照。不能把value或未訓練head直接當棋力。[networks](https://github.com/rlglab/strength-estimator/tree/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer)、[configs](https://github.com/rlglab/strength-estimator/tree/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/cfg)

BT batches按rank分組，rank內抽多局面，計算每rank平均score（或加權平均），用ordered rank的 Bradley–Terry型log-softmax ranking loss；trainer另有policy CE與value MSE。Go典型rank為3–5k、1–2k、1d到9d，11bins；SE_infinity多一個non-human bin。預设SE 32×11×7 positions、infinity32×12×7；2/3rankcfg是額外比較方案，不能認為只有單一rank也能恢復完整ordinal calibration。[training](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/strength/trainer/train.py)

SGF loader要求SZ/KM/RE/PB/PW/BR/WR，getRank實際從**BR**解析English k/d，不是CSV target rank或自動依目標色選WR。直接用匿名train_A row不滿足完整training契約；PB/PW若用identity只留metadata，不能送進模型，對手rank缺失須拒絕或另設無supervision inference adapter。train_A的A=1–3k跨原repo兩個kyu bins，不能硬指定某個exact rank。

已讀的 `se_go.cfg` 和 `se_infty_go.cfg` **都設定bt_use_weight=false**；SE前者不加non-human，後者增加。weight head存在不代表此cfg已訓練有效weight；「可用weight」是trainer的可選支路，不是這兩份cfg的實際使用。Official AdamW未指定weight_decay、採PyTorchdefault，本專案固定1e-4；容量之外也有optimizer差異。

**PRETRAINED GO CHECKPOINT NOT VERIFIED**。完整pinned tree沒有.pt/.pth/.pkl/.onnx/.bin/.h5權重；releases為0，已讀README與training/download相关scripts未提供可驗證的預訓練下載連結。cfg `model/se_go.pt`是路徑設定，不是檔案存在證據。範圍僅此commit與稽核來源，並非斷言全網沒有權重。README描述訓練後產生model目錄。GPL-3.0需在未來實際複用程式時保留授權／attribution；checkpoint授權、資料来源／AI CUP污染另行詢問。Linux/C++/LibTorch/Boost/OpenCV/Docker/NVIDIA為主要整合成本。[README](https://github.com/rlglab/strength-estimator/blob/d9498964e2ba84b5798c40e3f5b8a91a8791d66c/README.md)

## MiniZero Policy Detection：policy branch

`policy_play.py`其實是 next-move Top5 prediction，不是五位玩家識別。features `(B,18,19,19)`、trunk residual AlphaZero，policy head conv/BN/ReLU/flatten/FC後輸出 `(B,362)` logits及softmax；361棋盤點＋pass。value head存在，但policy `train2.py`主要CE(actual human move)+small bounded parameter regularizer，不能把value當已訓練strength。cfg example/腳本參數必須與checkpoint一致；trainpolicy.sh提6blocks256channels18planes，不可據此假設所有權重架構相同。[network](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/network/py/alphazero_network.py)、[learner](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/learner/train2.py)

Go18planes=last8 own/opponent alternating16＋black turn＋white turn；本專案17cache不可拿來forward。SGF action mapping `(18-y_top)*19+x`、pass361；`policy_play.py`反轉非pass座標正確，但把361也套同公式會得到非法'a`'，需專門pass分支。原converter按legacy第三CSV欄的comma-separated B[xy]讀入，不是2026 quoted sgf_content，pass處理不足。[policy_play](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/policy_play.py)、[SGF loader](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/utils/sgf_loader.cpp)

`TestDataLoader`讀一行SGF後replay到最後一步，再產生features；空move prefix回空features。計算actual-move probability必須在**該手之前**取board，否則洩漏答案。後續改用binding Env.reset/get_features/act，第一手直接reset state；不能無改動呼叫legacyloader。KM/PB/PW/BR/WR的複製在此loader被comment掉；須確認future棋規/komi/turn契約。binding有legal actions，但v1定義raw362 softmax，不會宣稱已有合法手masked probability。[data_loader](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/learner/data_loader.cpp)

training資料路徑指向legacy TrainingDataset dan*.sgf，validation也由legacyconverter切出；沒有本專案player-disjoint split保證。inference腳本hardcode best1.csv／public_submission_template及offset11340，不能沿用本專案question mapping。腳本即使只推論仍建optimizer/scheduler並載入其state；未來需獨立inference-only wrapper，不直接跑原腳本。

**PRETRAINED POLICY CHECKPOINT NOT VERIFIED**。`dan_training_new/model/weight_iter_400000.pkl`在pinned tree不存在；完整tree沒有上述權重副檔名、releases0，稽核README／scripts未見可驗證GoogleDrive或其他pretrained下載連結。clone完成不等於可以infer。完整實驗需教授提供checkpoint及匹配cfg／provenance／license，或未來自訓policy；這輪均未執行。C++17、LibTorchABI、Boost/ALE/OpenCV、Linux container和CUDA相容性尚未build驗證。通用MiniZero RL/MCTS流程也不是per-move fingerprint必須全跑的流程。

## 可複用程度

| 方法 | 可複用概念／元件 | 必須adapter／驗證 | 現在不適用 |
|---|---|---|---|
| Official | history-relativefeatures、triplet概念 | parser/pass、aggregation與split差異 | 原樣覆蓋frozen runner、用trainloss挑模型 |
| Policy | policy logits、Go Env action mapping、hidden hook設計 | 2026CSV、pre-move replay、18planes、pass、cfg/weights、Linuxbuild | legacyCSVconverter／hardcodedsubmission／直接clone推論 |
| Strength | ordinalrankloss、score/weight、rankheads | 真實BR/WR/RE/KM、rankbins、featureboundary、weights/env | 只用A自訓完整棋力尺度、偽造opponentrank |

Hook可取最後residual block的 `(B,C,19,19)`，global-average得到C維向量，分色分階段game/player平均後cosine。forward未return trunk，需read-onlyhook並檢查輸出、移除hook；不改checkpoint、不執行於Phase3.0。hidden偏board/context，不保證包含actual decision或identity，和顯式policy偏離統計應分別比較。
