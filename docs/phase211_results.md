# Phase 2.11：Embedding Collapse Forensics

## 教授 1 分鐘摘要

已保存的正式 TEST3 結果：Color-aware Opening score **0.773919**；frozen Fusion **0.790240**，提升 +0.016321 的 95% CI **[−0.022885, 0.052726]** 包含 0。這些數字只作研究背景，不作本輪選擇依據；没有開啟任何 CLOSED TEST inputs／scores，也沒有重新 inference。

本輪只用 DEV2。共同方向在 **pooled backbone 就已出現**，projection 後角度集中更嚴重：game cosine separation 從 **8.061e-07** 降至 **3.531e-09**。Normalize 前 raw cosine 已接近 1，因此 **L2 normalize 不是主因**。現有 projection 是單一 `Linear(576,128)`，沒有新增 MLP 或改 forward。

Raw 的 variance 沒有歸零，但其 TRAIN PC1 吃掉 **85.89%** variance，且與共同均值方向高度對齊；normalized effective rank **93.26**，仍有高維微小訊號，不能稱為完全恆定輸出。問題更像 **共同均值方向／projection 的角度集中，加上色彩域差異與身份訊號弱**。

Current／RawMean／RawGameNormalize score 都是 **0.417385**；remove-PC、whitening 均未超過 Current。Color-aware + global centering 的 DEV2 Triplet score **0.480703**，但固定 fusion 從 **0.758266** 降至 **0.743930**。沒有選出新正式模型。

## 1. 隔離與固定 checkpoint

Round 1、FINAL TEST 2、FINAL TEST 3 的 CLOSED／consumed markers 均存在。所有 Phase 2.11 I/O entry points 拒絕其 candidate／query／truth／score matrix，以及 Stability raw inputs。只允許 DEV2 TRAIN／VAL；歷史正式數字取自需求與已整理文件，不載入 CLOSED score arrays。

- Checkpoint：`outputs/phase28/experiments/Triplet-Hard-100/best.pt`，epoch 18。
- SHA256：`9048144a9bc7d422c8c321d607f73fd1dc378054988741e975b56fbb7dab2d4f`，執行前後完全相同。
- 64 channels／8 residual blocks／128 embedding；production model/config 原始 SHA256 通過。
- DEV2 TRAIN：200 players／4,000 games／64,000 positions；VAL：100 players／candidate 3,000／query 1,000／100 questions，共 64,000 positions。
- 每盤固定 16 positions，全部 cache provenance 與既有 split SHA256 相符；没有重新抽樣／改棋譜。
- TRAIN 與 VAL player identities 交集為 0；原 checkpoint 訓練的 100 位玩家是此 TRAIN pool 的子集。

| partition | total_games | successful_games | failed_games | positions |
| --- | --- | --- | --- | --- |
| train | 4000 | 4000 | 0 | 64000 |
| val_candidate | 3000 | 3000 | 0 | 48000 |
| val_query | 1000 | 1000 | 0 | 16000 |

## 2. Representation 擷取與計算定義

`PlayerEncoder.forward()` 完全未修改。Diagnostic hooks 捕捉 projection 的輸入（pooled backbone 576 維）、projection output（raw 128 維），以及原 forward 最終 normalized 128 維；同一次 forward 取得三者。`model.eval()`／no_grad/inference_mode，不建立 optimizer、不 backward、不更新 BatchNorm。

Variance/norm 分別計算 **所有 64,000 VAL positions** 與 **4,000 equal-position game means**。Position 同／異玩家各固定 seed 42 抽 20,000 pairs；game-level 使用全部 30,000 same-player／2,970,000 different-player candidate-query pairs。兩者不混用。Cosine separation=same−different；Euclidean separation=different distance−same distance。

Cosine 在 float64 中重新以每個向量的 norm 計算；Raw/Backbone 的 norm 和 variance 保留原尺度。Euclidean 先減共同原點再算距離，避免共同大均值造成 catastrophic cancellation。Variance near-zero 門檻固定 1e-8，但此指標依尺度改變，不能單獨用它定位 collapse。

### Position geometry

| layer | variance_mean | variance_median | variance_min | variance_max | near_zero_fraction | norm_mean | norm_median | norm_std | norm_min | norm_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backbone_feature | 0.000626253922822 | 0.00059461939635 | 0 | 0.00203272723257 | 0.0104166666667 | 66.4189523768 | 66.4199699345 | 0.0712929363857 | 66.1609788848 | 66.6907087025 |
| raw_embedding | 0.000666375946538 | 0.00066325012639 | 0.000114522633038 | 0.00173992166045 | 0 | 252.488208867 | 252.491260396 | 0.263912920845 | 251.508224995 | 253.498320832 |
| normalized_embedding | 1.91739348072e-09 | 1.88921983621e-09 | 1.50906420047e-09 | 2.71939847581e-09 | 1 | 1.0000000002 | 1.00000000011 | 2.43177856283e-08 | 0.999999907359 | 1.00000008743 |

| layer | same_cosine_mean | same_cosine_median | same_cosine_std | same_cosine_p25 | same_cosine_p75 | different_cosine_mean | different_cosine_median | different_cosine_std | different_cosine_p25 | different_cosine_p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backbone_feature | 0.999920315094 | 0.99992063701 | 1.29024841226e-05 | 0.99991249837 | 0.999928221674 | 0.999919353456 | 0.999919911553 | 1.27233651588e-05 | 0.999911776615 | 0.999927340707 |
| raw_embedding | 0.999999758055 | 0.99999975949 | 4.04367284154e-08 | 0.99999973218 | 0.999999784501 | 0.999999754369 | 0.999999756035 | 3.97949306363e-08 | 0.9999997293 | 0.999999781002 |
| normalized_embedding | 0.999999758055 | 0.999999759491 | 4.04367004153e-08 | 0.99999973218 | 0.999999784503 | 0.999999754369 | 0.999999756037 | 3.97949126828e-08 | 0.9999997293 | 0.999999781002 |

| layer | cosine_separation | same_euclidean_mean | different_euclidean_mean | euclidean_separation |
| --- | --- | --- | --- | --- |
| backbone_feature | 9.61637767083e-07 | 0.841448223457 | 0.846737861715 | 0.00528963825812 |
| raw_embedding | 3.68607289136e-09 | 0.365173365684 | 0.368996120126 | 0.00382275444137 |
| normalized_embedding | 3.68608654711e-09 | 0.000693131553668 | 0.00069854730991 | 5.41575624181e-06 |

### Game geometry

| layer | variance_mean | variance_median | variance_min | variance_max | near_zero_fraction | norm_mean | norm_median | norm_std | norm_min | norm_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backbone_feature | 0.000318714085266 | 0.000302219203821 | 0 | 0.0010898713195 | 0.0104166666667 | 66.4176310847 | 66.4187858928 | 0.0587848449835 | 66.2068013003 | 66.5961738609 |
| raw_embedding | 0.000434071006588 | 0.000427929179915 | 5.78873347038e-05 | 0.00116274204633 | 0 | 252.488193967 | 252.495247361 | 0.217805481857 | 251.702878869 | 253.122759237 |
| normalized_embedding | 9.9524368952e-10 | 9.82448522756e-10 | 7.62353137512e-10 | 1.4321642096e-09 | 1 | 0.999999941187 | 0.999999940946 | 1.15303266233e-08 | 0.999999904383 | 0.999999977185 |

| layer | same_cosine_mean | same_cosine_median | same_cosine_std | same_cosine_p25 | same_cosine_p75 | different_cosine_mean | different_cosine_median | different_cosine_std | different_cosine_p25 | different_cosine_p75 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backbone_feature | 0.999959926831 | 0.99996017697 | 7.59320840473e-06 | 0.999954984469 | 0.999965261946 | 0.999959120753 | 0.999959474568 | 7.25771933988e-06 | 0.999954509297 | 0.999964217622 |
| raw_embedding | 0.999999875879 | 0.999999877128 | 2.39569992484e-08 | 0.999999860691 | 0.999999892637 | 0.999999872348 | 0.999999873987 | 2.36814818178e-08 | 0.99999985749 | 0.999999888985 |
| normalized_embedding | 0.99999987589 | 0.999999877136 | 2.39569070886e-08 | 0.999999860705 | 0.999999892648 | 0.999999872358 | 0.999999873997 | 2.36813897766e-08 | 0.9999998575 | 0.999999888995 |

| layer | cosine_separation | same_euclidean_mean | different_euclidean_mean | euclidean_separation | mean_vector_norm | centered_rms_norm | common_mean_to_centered_rms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| backbone_feature | 8.06078415239e-07 | 0.597356392972 | 0.603728894771 | 0.00637250179929 | 66.4162750781 | 0.428461565503 | 155.011045157 |
| raw_embedding | 3.53113494e-09 | 0.290078244277 | 0.294614875437 | 0.00453663116024 | 252.488177883 | 0.235713997979 | 1071.16327434 |
| normalized_embedding | 3.53152751487e-09 | 0.000495889220244 | 0.000503095414643 | 7.20619439966e-06 | 0.999999877492 | 0.000356919027594 | 2801.7555809 |

最早可觀察的共同方向集中在 pooled backbone，game cosine 約 0.999959927／0.999959121。這不代表 backbone 全部維度完全相同：只有 1.04% 維 near-zero。Projection raw 的 mean/scatter ratio 從 backbone 155.0 升到 1071.2，cosine 更集中。Raw/normalized cosine 幾乎相同，說明 normalize 沒有創造這個角度問題。本輪未擷取 stem／各 residual block，不能進一步斷言發生在哪個卷積層。

## 3. TRAIN-only covariance／PCA

所有 statistics 用 **DEV2 TRAIN 4,000 個 game means** fitting；先平均每盤 positions，使每盤等權。不是在 VAL／Stability／TEST fitting。PCA 用中心化 population covariance，eigenvectors 以最大絕對元素正號固定 sign；effective rank=exp(entropy(eigenvalue proportions))，participation ratio=trace²/sum(eigenvalue²)。

| layer | top1_explained | top5_cumulative | top10_cumulative | top20_cumulative | effective_rank | participation_ratio | total_variance | anisotropy_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_embedding | 0.858904947749 | 0.877500794693 | 0.892718208547 | 0.914685020768 | 2.85666851758 | 1.35503038963 | 0.0561660077497 | True |
| normalized_embedding | 0.0490399268688 | 0.168056460581 | 0.2685523566 | 0.417738790739 | 93.2586002133 | 69.0552223456 | 1.269063901e-07 | False |

| layer | uncentered_top1_energy | mean_norm | centered_rms | mean_direction_first_centered_pc_alignment |
| --- | --- | --- | --- | --- |
| raw_embedding | 0.999999873085 | 252.491111696 | 0.236993687152 | 0.998268335773 |
| normalized_embedding | 0.999999873094 | 0.999999877855 | 0.000356239231556 | 1.86775971037e-05 |

| layer | rank | eigenvalue | explained_variance |
| --- | --- | --- | --- |
| raw_embedding | 1 | 0.0482412619515 | 0.858904947749 |
| raw_embedding | 2 | 0.000348987067105 | 0.0062134924857 |
| raw_embedding | 3 | 0.000272465305913 | 0.00485107125873 |
| raw_embedding | 4 | 0.000212405991618 | 0.00378175341506 |
| raw_embedding | 5 | 0.000210596118954 | 0.00374952978486 |
| raw_embedding | 6 | 0.000192641110735 | 0.00342985229774 |
| raw_embedding | 7 | 0.000172601302857 | 0.00307305628035 |
| raw_embedding | 8 | 0.000164437027734 | 0.002927696561 |
| raw_embedding | 9 | 0.000163477062735 | 0.00291060499553 |
| raw_embedding | 10 | 0.00016154488042 | 0.00287620371987 |
| raw_embedding | 11 | 0.000140618925275 | 0.0025036304147 |
| raw_embedding | 12 | 0.00013989539488 | 0.00249074841679 |
| raw_embedding | 13 | 0.000133915186307 | 0.0023842746115 |
| raw_embedding | 14 | 0.000127384867613 | 0.00226800644584 |
| raw_embedding | 15 | 0.000120528843398 | 0.00214593930078 |
| raw_embedding | 16 | 0.000119500794835 | 0.0021276355508 |
| raw_embedding | 17 | 0.000117207270181 | 0.00208680080492 |
| raw_embedding | 18 | 0.00011428975574 | 0.00203485631825 |
| raw_embedding | 19 | 0.000113035694585 | 0.002012528558 |
| raw_embedding | 20 | 0.000107411412574 | 0.00191239179848 |
| normalized_embedding | 1 | 6.22348008967e-09 | 0.0490399268688 |
| normalized_embedding | 2 | 5.21364102109e-09 | 0.041082572887 |
| normalized_embedding | 3 | 3.49222733585e-09 | 0.0275181362665 |
| normalized_embedding | 4 | 3.31695647573e-09 | 0.0261370327619 |
| normalized_embedding | 5 | 3.08113382296e-09 | 0.0242787917972 |
| normalized_embedding | 6 | 2.74760046955e-09 | 0.0216506077227 |
| normalized_embedding | 7 | 2.59953235568e-09 | 0.0204838570669 |
| normalized_embedding | 8 | 2.56752696348e-09 | 0.020231660214 |
| normalized_embedding | 9 | 2.53772093717e-09 | 0.0199967939769 |
| normalized_embedding | 10 | 2.30119065766e-09 | 0.0181329770381 |
| normalized_embedding | 11 | 2.20167610022e-09 | 0.0173488198545 |
| normalized_embedding | 12 | 2.10049200275e-09 | 0.0165515069896 |
| normalized_embedding | 13 | 2.02869223949e-09 | 0.0159857375023 |
| normalized_embedding | 14 | 1.90068567989e-09 | 0.0149770683603 |
| normalized_embedding | 15 | 1.87849657236e-09 | 0.0148022221016 |
| normalized_embedding | 16 | 1.83883240599e-09 | 0.0144896754572 |
| normalized_embedding | 17 | 1.79567510658e-09 | 0.0141496035399 |
| normalized_embedding | 18 | 1.77474326122e-09 | 0.0139846642854 |
| normalized_embedding | 19 | 1.73953122422e-09 | 0.0137071996363 |
| normalized_embedding | 20 | 1.67388721567e-09 | 0.0131899364118 |

`pca_diagnostics.csv` 同時包含 top-20 eigenvalue columns，逐 rank 版本另存 `pca_top20_eigenvalues.csv`。TRAIN mean／B/W means／components 存 NPZ，來源與 SHA256 存 `fit_statistics_manifest.json`。

**ANISOTROPY WARNING** 門檻固定 top1≥50% 或 top5≥80%；raw 觸發、normalized 不觸發。兩者未中心化 second moment 的 PC1 energy 都約 99.999987%，說明共同**均值方向**主導；不能把它和中心化後的 covariance PC1 混為一談。Raw PC1 與 TRAIN mean 對齊約 0.998268，大部分變化偏向長度／徑向；normalize 消除徑向變化後，剩餘很小但高 effective-rank 的角度訊號。

## 4. Aggregation／normalization

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Current | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| RawMean | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| RawGameNormalize | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| Backbone | 0.310000 | 0.450000 | 0.580000 | 0.356785 |

Current：normalized positions→game mean→player/query mean→最後 normalize/cosine。
RawMean：raw positions→game mean→player/query mean→最後 normalize/cosine。
RawGameNormalize：raw positions→game mean→game normalize→player/query mean→normalize/cosine。
Backbone：pooled position features→game mean→player/query mean→normalize/cosine。

三個 Triplet aggregation 的 Top-k/score 完全相同，且 Current 精確重現原 checkpoint DEV2 score；沒有證據支持「改 normalize 的時機就能解決」。Backbone score 較低；projection 有壓縮角度差異，也保留／重組部分 retrieval 訊號，不能直接稱為無用。

## 5. Mean-centering

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Current | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| Centered-Raw | 0.140000 | 0.210000 | 0.250000 | 0.157812 |
| Centered-Normalized | 0.330000 | 0.540000 | 0.600000 | 0.380011 |

每個 game embedding 減對應 TRAIN game mean後 L2 normalize，再平均同一 player/question 的 games並 L2 normalize。Raw／Normalized fitting 各使用自己的 TRAIN representation，無 VAL label fitting。Centered-Raw 明顯變差，Centered-Normalized 也低於 Current。

## 6. Remove top PCs

| removed_pcs | top1 | top3 | top5 | competition_score | same_diff_separation | effective_rank |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.14 | 0.21 | 0.25 | 0.1578118487 | 0.01977937986 | 8.309591224 |
| 1 | 0.3 | 0.51 | 0.6 | 0.35458209 | 0.02763600482 | 95.99693867 |
| 2 | 0.33 | 0.51 | 0.56 | 0.3738796656 | 0.01915097382 | 97.86989202 |
| 4 | 0.24 | 0.45 | 0.55 | 0.2997308438 | 0.01561974473 | 98.92999127 |
| 8 | 0.22 | 0.39 | 0.45 | 0.2656643393 | 0.01358856478 | 98.79213659 |

固定 k=0/1/2/4/8，只 fit TRAIN raw PCA。Game raw−TRAIN mean→移除 PCs→game normalize→平均 games→player/query normalize。表內 effective rank 是同一 transform 後 TRAIN game covariance 的 effective rank，沒有利用 VAL fitting。k=0 與 Centered-Raw 相同。

移除 PC1／PC2 相對 k=0 回復部分表現，但所有 k 都沒有超過 Current。因此 dominant PC 是幾何問題的一部分，去掉它不足以恢復更強的玩家辨識；也可能丟失有用訊號。

## 7. Whitening

| comparison | method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- |
| Raw | RawMean | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| Centered | Centered-Raw | 0.140000 | 0.210000 | 0.250000 | 0.157812 |
| Whitened | Whitened | 0.350000 | 0.530000 | 0.560000 | 0.393828 |

TRAIN raw mean/PCA fitting；game raw−TRAIN mean→PCA rotation→除 sqrt(max(eigenvalue,**1e-5**))→game normalize→player/query mean normalize。固定 epsilon，沒有搜尋。

0/128 eigenvalues 被 floor；所有輸出 finite，沒有數值不穩定 warning。Whitening 改善相對純 raw centering，但仍低於 Current；未把它選成正式模型。

## 8. Color-aware Triplet／color-specific centering

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Current | 0.380000 | 0.500000 | 0.570000 | 0.417385 |
| Color-aware Triplet | 0.390000 | 0.590000 | 0.680000 | 0.445869 |

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Color-aware Triplet | 0.390000 | 0.590000 | 0.680000 | 0.445869 |
| Color-aware + global centering | 0.420000 | 0.620000 | 0.720000 | 0.480703 |
| Color-aware + color-specific centering | 0.400000 | 0.620000 | 0.710000 | 0.472528 |

Baseline 使用 normalized-position game means，B query 只對 candidate B，W 只對 W，按 query B/W game 數加權。Missing color bank 用零 similarity，禁止跨色 fallback。本批沒有缺色 candidate bank。

Global／color-specific center 使用 TRAIN **normalized game means**；先在 game 層級減全球 TRAIN mean，或相應 TRAIN B/W mean並 normalize，再分色 aggregation。Color-aware Triplet 高於 Current；global centering 最高，color-specific 比未中心化 color-aware 好，但沒有超過 global centering。表示 color-domain shift 有影響，尚不能說每色中心才是主要修復方式。

## 9. Within／Between player separability

| method | within_mean | between_mean | ratio |
| --- | --- | --- | --- |
| Current | 0.000344003292906 | 0.000122880266203 | 0.357206656847 |
| RawMean | 0.204546547975 | 0.0692890547208 | 0.338744678934 |
| Color-aware + global centering | 0.968255547785 | 0.343612438322 | 0.354877840988 |

按需求使用未平方 Euclidean mean distance；這不是統計學的 squared variance。Within=各 candidate game 到其 player centroid 距離平均；between=不同 player centroid 間距離平均。Current／RawMean 保留各自 game 尺度；best diagnostic 使用 transform 後 game vectors。Ratio 無尺度，但 color-aware 的表列 centroid 是 mixed game centroid，無法完整反映分色 score banks。

三者 between/within 都約 0.34–0.36：玩家間差異小於同玩家內的棋局差異。即使後處理提高 DEV retrieval，也沒有讓 pooled within/between ratio 明顯變好。

## 10. 固定 alpha diagnostic fusion

| method | triplet_diagnostic | alpha | normalization | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| old_triplet_fusion | Current | 0.900000 | z-score | 0.710000 | 0.850000 | 0.910000 | 0.758266 |
| new_triplet_fusion | Color-aware + global centering | 0.900000 | z-score | 0.700000 | 0.840000 | 0.900000 | 0.743930 |

只對 DEV2，Opening 固定 Color-aware-player-5、alpha=0.9／z-score。最高 DEV-only post-processing 是 `Color-aware + global centering`，但 new fusion 比 old fusion **-0.014336**。Triplet 單獨 retrieval 提升不保證能提供更有互補性的 fusion 訊號。

這個最高值是已使用多次的 DEV2 上 exploratory comparison，沒有獨立泛化保證；不建立 `phase211_selected.yaml`，不更新任何 frozen competition model。

## 11. 技術結論 A–H

A. 最早在 pooled backbone 已觀察到共同方向集中，projection 後角度差異進一步縮小；固定 threshold 的近零 variance 警示只在 normalized 出現，不能因此把原因歸給 normalize。
B. Raw 絕對 variance 較大／無 near-zero 維，但方向仍接近恆定、centered covariance 強 anisotropy，不能說它較健康；normalized 不是零秩，其 TRAIN effective rank 仍約 93。
C. 共同均值方向強烈主導 uncentered energy；raw covariance PC1 也對齊均值方向。兩種 dominant-direction 指標不同，但皆支持徑向／共同方向成分很大。
D. Remove-PC 只比 centered raw 好，沒有超過 Current。
E. Whitening 數值穩定、沒有超過 Current。
F. Color-aware Triplet 改善 DEV2 retrieval。
G. Color-specific centering 比單純 color-aware 改善，但低於 global centering；未支持其為最優修復。
H. 更像 **多個因素共同存在：encoder/projection 的角度集中、色彩域差異與玩家間／玩家內分離不足**；不支持 normalize／aggregation 是單一主因，也不是完全恆定 representation。

這些是固定 checkpoint 的觀察，沒有因果 intervention training，不能斷言某 loss／某 residual block 造成問題。

## 12. Tests、warnings、執行與下一步

完整 **118 tests passed**，failure/error/skipped=0；imports／syntax check 通過。Production forward 與全部 buffers 不變；raw projection／norm≈1、TRAIN-only center/PCA/whitening、deterministic remove-PC、B/W-only matching、所有 CLOSED paths 拒絕、同一 questions 與無 optimizer/backward 全部涵蓋。

正式 DEV-only inference＋診斷耗時 **33.72 秒**，未訓練任何模型。Warnings：

- ANISOTROPY WARNING: TRAIN covariance exceeds preregistered top1/top5 concentration threshold
- NEAR-ZERO VARIANCE WARNING: absolute threshold is scale-dependent; inspect raw norms and relative scatter

明天最值得討論的模型方向：**在新的 DEV 研究中，以可監測 mean direction／variance／within-between separation 的身份辨識目標，研究防止表示角度集中，並把 B/W 色彩域因素納入控制**。先設計 diagnostics 與驗證 protocol，再考慮 loss／sampling／projection 的受控改動；不要從 TEST3 分數倒推修改。

本輪不實作新模型／loss、不改 checkpoint／epoch、不加入 Strength Estimator／MiniZero、不建立新 FINAL TEST。

```powershell
python scripts/check_phase211.py
# 已完成；runner 拒絕重做，使用保存的 DEV artifacts
python -m src.phase211_forensics --config configs/phase211.yaml
python scripts/report_phase211.py
```
