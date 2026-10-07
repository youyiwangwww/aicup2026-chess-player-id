# Phase 2.13：Multi-Seed Anti-Collapse Stability

所有模型 **DEV RESEARCH ONLY**。沒有final model／FINAL TEST4，沒有使用任何CLOSED TEST、DEV2或Stability作selection。

## 1. Frozen protocol與執行紀錄

DEV3 TRAIN100×20=2,000games；VAL50×candidate30/query10=1,500/500games。split seed永久42。直接沿用Phase2.12的CSV、16positions/game cache與固定TRAIN diagnostic subset，沒有重新抽identity/game、沒有重新preprocess positions。

四training seeds：42/123/2026/31415。42僅reference metadata，沒有copy checkpoint或training；其餘各四objective×20epochs，共240新epochs。training seed控制model initialization、epoch player permutation、cached-position sampling、PyTorch與CUDA RNG；cache sampling seed仍42。

所有hyperparameters與Phase2.12一致：batch16、lr.001、weight_decay.0001、margin.2、每player20triplets/epoch，2,000anchors/epoch、batch-hard。A1 lambda_mean=.1；A2 lambda_var=1；A3兩者；gamma=1/sqrt128、epsilon1e-4、population variance。regularizer在48-row source forward pool計算一次；hard-negative reuse不增加regularizer權重。

每seed四組初始化SHA相同，四seed初始化不同；best epoch只看normal DEV3 score，tie保留最早。Color/fusion/geometry都不參與selection。

Wall-clock：**69.36分鐘**（訓練、cache載入、diagnostics與hash checks）。原Phase2.12所有artifacts/hash不變；完整protocol見`outputs/phase213/protocol.json`。

## 2. 四seeds × 四experiments retrieval

| seed | experiment | best_epoch | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | 16 | 0.56 | 0.7 | 0.76 | 0.585956 |
| 42 | A1-Triplet-Mean | 19 | 0.54 | 0.72 | 0.78 | 0.590602 |
| 42 | A2-Triplet-Variance | 10 | 0.6 | 0.68 | 0.84 | 0.634879 |
| 42 | A3-Triplet-MeanVariance | 19 | 0.6 | 0.74 | 0.82 | 0.645555 |
| 123 | A0-Triplet | 20 | 0.52 | 0.76 | 0.84 | 0.587131 |
| 123 | A1-Triplet-Mean | 19 | 0.48 | 0.76 | 0.82 | 0.560851 |
| 123 | A2-Triplet-Variance | 11 | 0.5 | 0.66 | 0.74 | 0.556934 |
| 123 | A3-Triplet-MeanVariance | 20 | 0.48 | 0.72 | 0.88 | 0.547093 |
| 2026 | A0-Triplet | 18 | 0.58 | 0.74 | 0.9 | 0.639029 |
| 2026 | A1-Triplet-Mean | 18 | 0.54 | 0.76 | 0.86 | 0.618744 |
| 2026 | A2-Triplet-Variance | 17 | 0.48 | 0.72 | 0.76 | 0.547028 |
| 2026 | A3-Triplet-MeanVariance | 19 | 0.56 | 0.78 | 0.8 | 0.632627 |
| 31415 | A0-Triplet | 18 | 0.54 | 0.74 | 0.74 | 0.594972 |
| 31415 | A1-Triplet-Mean | 17 | 0.52 | 0.72 | 0.8 | 0.578326 |
| 31415 | A2-Triplet-Variance | 19 | 0.54 | 0.74 | 0.88 | 0.614636 |
| 31415 | A3-Triplet-MeanVariance | 11 | 0.54 | 0.74 | 0.76 | 0.59999 |

## 3. Aggregate與sign consistency

STD使用跨四seeds的 **sample standard deviation（ddof=1）**，不是standard error或confidence interval。四seeds共用同一DEV3，不能視為四個獨立泛化資料集。

| experiment | top1_mean | top1_std | competition_score_mean | competition_score_std | competition_score_min | competition_score_max |
| --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.55 | 0.0258199 | 0.601772 | 0.0251582 | 0.585956 | 0.639029 |
| A1-Triplet-Mean | 0.52 | 0.0282843 | 0.587131 | 0.0243556 | 0.560851 | 0.618744 |
| A2-Triplet-Variance | 0.53 | 0.052915 | 0.588369 | 0.043013 | 0.547028 | 0.634879 |
| A3-Triplet-MeanVariance | 0.545 | 0.05 | 0.606316 | 0.0438915 | 0.547093 | 0.645555 |

| experiment | retrieval_improved_count | geometry_majority_improved_count | seeds | mean_retrieval_improved | classification |
| --- | --- | --- | --- | --- | --- |
| A1-Triplet-Mean | 1 | 2 | 4 | False | SEED-SENSITIVE RESULT |
| A2-Triplet-Variance | 2 | 1 | 4 | False | SEED-SENSITIVE RESULT |
| A3-Triplet-MeanVariance | 2 | 1 | 4 | True | SEED-SENSITIVE RESULT |

A2穩定promising需要平均retrieval>A0、retrieval至少3/4、geometry majority至少3/4。A3穩定retrieval需要平均>A0且retrieval至少3/4；geometry另判。Geometry主要四軸mean↓、rawPC1↓、cosine separation↑、between/within↑，每seed至少3/4才算majority；不單看effective rank。

## 4. Same-seed paired deltas

| seed | experiment | competition_score_delta | mean_direction_norm_delta | raw_pc1_explained_delta | cosine_separation_delta | between_within_ratio_delta | geometry_improved_directions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | A1-Triplet-Mean | 0.00464637 | 1.28031e-08 | -0.138805 | -1.51287e-10 | 0.0162055 | 2 |
| 42 | A2-Triplet-Variance | 0.048923 | -1.13682e-07 | -0.0958861 | 5.31215e-09 | -0.0415807 | 3 |
| 42 | A3-Triplet-MeanVariance | 0.0595993 | 1.07846e-08 | -0.0554799 | -7.84942e-11 | 0.00499791 | 2 |
| 123 | A1-Triplet-Mean | -0.0262806 | -6.1992e-09 | 0.101586 | 6.60146e-10 | -0.000354498 | 2 |
| 123 | A2-Triplet-Variance | -0.0301974 | -6.82955e-08 | 0.0894651 | 2.25553e-09 | -0.0576048 | 2 |
| 123 | A3-Triplet-MeanVariance | -0.0400385 | 9.01677e-11 | 0.0581871 | 2.78877e-10 | 0.00189794 | 2 |
| 2026 | A1-Triplet-Mean | -0.020285 | -3.67265e-09 | 0.0263519 | 4.76184e-10 | 0.0100574 | 3 |
| 2026 | A2-Triplet-Variance | -0.0920005 | -5.1735e-09 | 0.0608329 | -5.62005e-11 | 0.0027495 | 2 |
| 2026 | A3-Triplet-MeanVariance | -0.00640116 | 2.01613e-09 | 0.0773097 | 5.33639e-11 | 0.00786794 | 2 |
| 31415 | A1-Triplet-Mean | -0.0166465 | -8.23023e-09 | -0.0110514 | 3.51405e-10 | -0.00655907 | 3 |
| 31415 | A2-Triplet-Variance | 0.019664 | 6.03912e-09 | 0.0787349 | -7.08318e-11 | 0.000350796 | 1 |
| 31415 | A3-Triplet-MeanVariance | 0.0050172 | -8.74556e-08 | -0.0889815 | 3.26015e-09 | -0.0472248 | 3 |

每row減去同seed A0，沒有跨seed錯配。全部geometry mean/std保存於aggregate_results.csv。

## 5. Best checkpoint geometry

| seed | experiment | mean_direction_norm | raw_pc1_explained | normalized_effective_rank | cosine_separation | between_within_ratio | variance_mean | near_zero_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | 0.999999922473 | 0.886261 | 91.4471 | 6.32872e-09 | 0.398375 | 1.21137e-09 | 1 |
| 42 | A1-Triplet-Mean | 0.999999935276 | 0.747456 | 88.2697 | 6.17743e-09 | 0.41458 | 1.01132e-09 | 1 |
| 42 | A2-Triplet-Variance | 0.999999808790 | 0.790375 | 97.6479 | 1.16409e-08 | 0.356794 | 2.98765e-09 | 1 |
| 42 | A3-Triplet-MeanVariance | 0.999999933257 | 0.830781 | 88.3695 | 6.25023e-09 | 0.403373 | 1.04286e-09 | 1 |
| 123 | A0-Triplet | 0.999999944610 | 0.732286 | 84.9188 | 5.1938e-09 | 0.414562 | 8.6547e-10 | 1 |
| 123 | A1-Triplet-Mean | 0.999999938411 | 0.833872 | 84.2455 | 5.85395e-09 | 0.414208 | 9.62333e-10 | 1 |
| 123 | A2-Triplet-Variance | 0.999999876314 | 0.821752 | 95.3751 | 7.44933e-09 | 0.356957 | 1.93259e-09 | 1 |
| 123 | A3-Triplet-MeanVariance | 0.999999944700 | 0.790474 | 83.3675 | 5.47268e-09 | 0.41646 | 8.64062e-10 | 1 |
| 2026 | A0-Triplet | 0.999999940258 | 0.771444 | 85.8952 | 5.84561e-09 | 0.404547 | 9.33469e-10 | 1 |
| 2026 | A1-Triplet-Mean | 0.999999936585 | 0.797795 | 82.7197 | 6.32179e-09 | 0.414605 | 9.90854e-10 | 1 |
| 2026 | A2-Triplet-Variance | 0.999999935084 | 0.832276 | 86.9959 | 5.78941e-09 | 0.407297 | 1.01431e-09 | 1 |
| 2026 | A3-Triplet-MeanVariance | 0.999999942274 | 0.848753 | 80.5782 | 5.89897e-09 | 0.412415 | 9.01967e-10 | 1 |
| 31415 | A0-Triplet | 0.999999929757 | 0.862874 | 88.5545 | 6.31381e-09 | 0.405116 | 1.09755e-09 | 1 |
| 31415 | A1-Triplet-Mean | 0.999999921527 | 0.851823 | 88.9842 | 6.66522e-09 | 0.398557 | 1.22615e-09 | 1 |
| 31415 | A2-Triplet-Variance | 0.999999935796 | 0.941609 | 85.7522 | 6.24298e-09 | 0.405467 | 1.00319e-09 | 1 |
| 31415 | A3-Triplet-MeanVariance | 0.999999842301 | 0.773893 | 96.9845 | 9.57397e-09 | 0.357891 | 2.46404e-09 | 1 |

所有best checkpoints near-zero維度fraction均100%：**True**。mean direction與cosine separation需看絕對量級，方向性改善不等於collapse已解決。相同game-level幾何定義沿用Phase2.12。

## 6. TRAIN-only mean gradient diagnostic

固定TRAIN48-row forward batch：原TRAIN cache、seed42/epoch0前16triplets的a/p/n，不隨training seed或模型更換。診斷用eval-mode、無optimizer step、autograd只求embedding梯度，parameter .grad保持None；不寫checkpoint或參與selection。

| seed | best_epoch | mean_loss | mean_gradient_normalized_norm | mean_gradient_raw_norm | mean_gradient_raw_tangential_norm | raw_norm_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 19 | 0.999999761581 | 0.288675 | 5.86037e-07 | 5.86037e-07 | 247.408 |
| 123 | 19 | 0.999999821186 | 0.288675 | 5.55353e-07 | 5.55353e-07 | 250.172 |
| 2026 | 18 | 0.999999761581 | 0.288675 | 5.58348e-07 | 5.58349e-07 | 249.219 |
| 31415 | 17 | 0.999999821186 | 0.288675 | 6.37069e-07 | 6.37069e-07 | 246.56 |

L_mean=||mean(z)||²；對normalized z的梯度2mean(z)/N。對raw x須通過L2的Jacobian：(I-zzᵀ)/||x||。完全同方向時normalized梯度可以非零，但raw tangential梯度為零，正是synthetic test驗證的駐點。實際數值以表為準；不能把非零normalized gradient解讀成有效旋轉raw方向。

## 7. Variance gradient diagnostic

| seed | experiment | variance_loss | variance_gradient_raw_norm | actual_normalized_std_mean | actual_normalized_std_min | actual_normalized_std_max |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | 0.0783882 | 2.53035e-07 | 4.78615e-05 | 3.63145e-05 | 6.25568e-05 |
| 42 | A1-Triplet-Mean | 0.0783882 | 2.28919e-07 | 4.41495e-05 | 3.46991e-05 | 5.56513e-05 |
| 42 | A2-Triplet-Variance | 0.0783881 | 4.46502e-07 | 7.65922e-05 | 5.60264e-05 | 0.000100025 |
| 42 | A3-Triplet-MeanVariance | 0.0783882 | 2.25812e-07 | 4.43872e-05 | 3.24642e-05 | 5.92829e-05 |
| 123 | A0-Triplet | 0.0783883 | 2.06975e-07 | 4.01324e-05 | 2.95321e-05 | 5.54195e-05 |
| 123 | A1-Triplet-Mean | 0.0783883 | 2.16933e-07 | 4.22952e-05 | 3.08694e-05 | 5.40559e-05 |
| 123 | A2-Triplet-Variance | 0.0783882 | 3.51857e-07 | 5.98585e-05 | 3.81611e-05 | 7.49935e-05 |
| 123 | A3-Triplet-MeanVariance | 0.0783883 | 2.05598e-07 | 3.96742e-05 | 2.88795e-05 | 5.41864e-05 |
| 2026 | A0-Triplet | 0.0783883 | 2.1784e-07 | 4.16134e-05 | 2.93181e-05 | 5.69982e-05 |
| 2026 | A1-Triplet-Mean | 0.0783883 | 2.18103e-07 | 4.22827e-05 | 3.20315e-05 | 5.98198e-05 |
| 2026 | A2-Triplet-Variance | 0.0783883 | 2.17265e-07 | 4.21673e-05 | 3.32814e-05 | 5.45507e-05 |
| 2026 | A3-Triplet-MeanVariance | 0.0783883 | 2.11436e-07 | 4.09633e-05 | 2.72409e-05 | 6.16469e-05 |
| 31415 | A0-Triplet | 0.0783882 | 2.32787e-07 | 4.47552e-05 | 3.15915e-05 | 5.52213e-05 |
| 31415 | A1-Triplet-Mean | 0.0783882 | 2.48852e-07 | 4.77351e-05 | 3.10626e-05 | 7.03955e-05 |
| 31415 | A2-Triplet-Variance | 0.0783883 | 2.20422e-07 | 4.33615e-05 | 2.92546e-05 | 5.99368e-05 |
| 31415 | A3-Triplet-MeanVariance | 0.0783881 | 4.01061e-07 | 7.04432e-05 | 4.68332e-05 | 9.36192e-05 |

actual std不加epsilon；loss仍sqrt(population variance+1e-4)，沒有改epsilon。完全collapsed時penalty=gamma-.01≈.07838835；非零raw梯度表示可微訊號存在，但若極小，不能推論強到足以對抗triplet dynamics。所有四objective的best checkpoints同批診斷，A2是本項主比較。

## 8. Color-aware secondary stability

| seed | experiment | retrieval | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | Normal | 0.56 | 0.7 | 0.76 | 0.585956 |
| 42 | A0-Triplet | Color-aware | 0.52 | 0.74 | 0.9 | 0.579106 |
| 42 | A1-Triplet-Mean | Normal | 0.54 | 0.72 | 0.78 | 0.590602 |
| 42 | A1-Triplet-Mean | Color-aware | 0.54 | 0.78 | 0.94 | 0.624438 |
| 42 | A2-Triplet-Variance | Normal | 0.6 | 0.68 | 0.84 | 0.634879 |
| 42 | A2-Triplet-Variance | Color-aware | 0.52 | 0.76 | 0.8 | 0.605002 |
| 42 | A3-Triplet-MeanVariance | Normal | 0.6 | 0.74 | 0.82 | 0.645555 |
| 42 | A3-Triplet-MeanVariance | Color-aware | 0.64 | 0.82 | 0.86 | 0.688977 |
| 123 | A0-Triplet | Normal | 0.52 | 0.76 | 0.84 | 0.587131 |
| 123 | A0-Triplet | Color-aware | 0.56 | 0.76 | 0.84 | 0.627628 |
| 123 | A1-Triplet-Mean | Normal | 0.48 | 0.76 | 0.82 | 0.560851 |
| 123 | A1-Triplet-Mean | Color-aware | 0.54 | 0.78 | 0.82 | 0.611679 |
| 123 | A2-Triplet-Variance | Normal | 0.5 | 0.66 | 0.74 | 0.556934 |
| 123 | A2-Triplet-Variance | Color-aware | 0.46 | 0.68 | 0.7 | 0.523326 |
| 123 | A3-Triplet-MeanVariance | Normal | 0.48 | 0.72 | 0.88 | 0.547093 |
| 123 | A3-Triplet-MeanVariance | Color-aware | 0.54 | 0.76 | 0.88 | 0.602395 |
| 2026 | A0-Triplet | Normal | 0.58 | 0.74 | 0.9 | 0.639029 |
| 2026 | A0-Triplet | Color-aware | 0.62 | 0.82 | 0.92 | 0.67002 |
| 2026 | A1-Triplet-Mean | Normal | 0.54 | 0.76 | 0.86 | 0.618744 |
| 2026 | A1-Triplet-Mean | Color-aware | 0.54 | 0.76 | 0.84 | 0.617748 |
| 2026 | A2-Triplet-Variance | Normal | 0.48 | 0.72 | 0.76 | 0.547028 |
| 2026 | A2-Triplet-Variance | Color-aware | 0.58 | 0.74 | 0.86 | 0.634904 |
| 2026 | A3-Triplet-MeanVariance | Normal | 0.56 | 0.78 | 0.8 | 0.632627 |
| 2026 | A3-Triplet-MeanVariance | Color-aware | 0.54 | 0.74 | 0.86 | 0.609619 |
| 31415 | A0-Triplet | Normal | 0.54 | 0.74 | 0.74 | 0.594972 |
| 31415 | A0-Triplet | Color-aware | 0.56 | 0.62 | 0.76 | 0.586525 |
| 31415 | A1-Triplet-Mean | Normal | 0.52 | 0.72 | 0.8 | 0.578326 |
| 31415 | A1-Triplet-Mean | Color-aware | 0.54 | 0.78 | 0.92 | 0.61942 |
| 31415 | A2-Triplet-Variance | Normal | 0.54 | 0.74 | 0.88 | 0.614636 |
| 31415 | A2-Triplet-Variance | Color-aware | 0.6 | 0.86 | 0.92 | 0.66608 |
| 31415 | A3-Triplet-MeanVariance | Normal | 0.54 | 0.74 | 0.76 | 0.59999 |
| 31415 | A3-Triplet-MeanVariance | Color-aware | 0.54 | 0.82 | 0.88 | 0.612808 |

| experiment | retrieval | competition_score_mean | competition_score_std |
| --- | --- | --- | --- |
| A0-Triplet | Normal | 0.601772 | 0.0251582 |
| A0-Triplet | Color-aware | 0.61582 | 0.0419649 |
| A1-Triplet-Mean | Normal | 0.587131 | 0.0243556 |
| A1-Triplet-Mean | Color-aware | 0.618321 | 0.00526196 |
| A2-Triplet-Variance | Normal | 0.588369 | 0.043013 |
| A2-Triplet-Variance | Color-aware | 0.607328 | 0.0613025 |
| A3-Triplet-MeanVariance | Normal | 0.606316 | 0.0438915 |
| A3-Triplet-MeanVariance | Color-aware | 0.62845 | 0.0405859 |

Bquery只對Bbank，Wquery只對Wbank，按query game數加權；missing bank固定similarity=0，無cross-color fallback。這是secondary，沒有選color-aware checkpoint。

## 9. Fixed Opening／Fusion stability

| method | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- |
| Color-aware-player-5 | 0.88 | 0.94 | 0.98 | 0.903435 |

Opening只reference一次Phase2.12保存的score matrix，沒有重新計算。固定alpha=.9，逐question z-score fusion。

| seed | experiment | alpha | normalization | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | 0.9 | z-score | 0.9 | 0.96 | 0.98 | 0.918418 |
| 42 | A1-Triplet-Mean | 0.9 | z-score | 0.86 | 0.96 | 0.98 | 0.893133 |
| 42 | A2-Triplet-Variance | 0.9 | z-score | 0.88 | 0.94 | 0.98 | 0.904064 |
| 42 | A3-Triplet-MeanVariance | 0.9 | z-score | 0.9 | 0.94 | 0.98 | 0.916707 |
| 123 | A0-Triplet | 0.9 | z-score | 0.9 | 0.94 | 0.98 | 0.916707 |
| 123 | A1-Triplet-Mean | 0.9 | z-score | 0.86 | 0.94 | 0.98 | 0.890792 |
| 123 | A2-Triplet-Variance | 0.9 | z-score | 0.88 | 0.96 | 0.98 | 0.905775 |
| 123 | A3-Triplet-MeanVariance | 0.9 | z-score | 0.88 | 0.94 | 0.98 | 0.904064 |
| 2026 | A0-Triplet | 0.9 | z-score | 0.88 | 0.94 | 0.98 | 0.904064 |
| 2026 | A1-Triplet-Mean | 0.9 | z-score | 0.9 | 0.94 | 0.98 | 0.916077 |
| 2026 | A2-Triplet-Variance | 0.9 | z-score | 0.86 | 0.94 | 0.98 | 0.886142 |
| 2026 | A3-Triplet-MeanVariance | 0.9 | z-score | 0.88 | 0.96 | 0.98 | 0.905146 |
| 31415 | A0-Triplet | 0.9 | z-score | 0.88 | 0.94 | 0.98 | 0.899413 |
| 31415 | A1-Triplet-Mean | 0.9 | z-score | 0.88 | 0.96 | 0.98 | 0.905775 |
| 31415 | A2-Triplet-Variance | 0.9 | z-score | 0.9 | 0.94 | 0.98 | 0.916707 |
| 31415 | A3-Triplet-MeanVariance | 0.9 | z-score | 0.86 | 0.94 | 0.98 | 0.886771 |

| experiment | competition_score_mean | competition_score_std |
| --- | --- | --- |
| A0-Triplet | 0.90965 | 0.00935694 |
| A1-Triplet-Mean | 0.901444 | 0.0117675 |
| A2-Triplet-Variance | 0.903172 | 0.0126596 |
| A3-Triplet-MeanVariance | 0.903172 | 0.0123406 |

## 10. Opening complementarity（Top1）

| seed | experiment | opening_wrong_triplet_correct | opening_correct_triplet_wrong | both_correct | both_wrong | fusion_correct_when_opening_wrong | fusion_wrong_when_opening_correct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | A0-Triplet | 2 | 18 | 26 | 4 | 1 | 0 |
| 42 | A1-Triplet-Mean | 2 | 19 | 25 | 4 | 1 | 2 |
| 42 | A2-Triplet-Variance | 3 | 17 | 27 | 3 | 1 | 1 |
| 42 | A3-Triplet-MeanVariance | 3 | 17 | 27 | 3 | 1 | 0 |
| 123 | A0-Triplet | 3 | 21 | 23 | 3 | 1 | 0 |
| 123 | A1-Triplet-Mean | 1 | 21 | 23 | 5 | 0 | 1 |
| 123 | A2-Triplet-Variance | 2 | 21 | 23 | 4 | 1 | 1 |
| 123 | A3-Triplet-MeanVariance | 0 | 20 | 24 | 6 | 0 | 0 |
| 2026 | A0-Triplet | 1 | 16 | 28 | 5 | 1 | 1 |
| 2026 | A1-Triplet-Mean | 1 | 18 | 26 | 5 | 1 | 0 |
| 2026 | A2-Triplet-Variance | 1 | 21 | 23 | 5 | 1 | 2 |
| 2026 | A3-Triplet-MeanVariance | 2 | 18 | 26 | 4 | 1 | 1 |
| 31415 | A0-Triplet | 1 | 18 | 26 | 5 | 0 | 0 |
| 31415 | A1-Triplet-Mean | 3 | 21 | 23 | 3 | 0 | 0 |
| 31415 | A2-Triplet-Variance | 1 | 18 | 26 | 5 | 1 | 0 |
| 31415 | A3-Triplet-MeanVariance | 3 | 20 | 24 | 3 | 0 | 1 |

跨seeds平均question counts：

| experiment | opening_wrong_triplet_correct | opening_correct_triplet_wrong | both_correct | both_wrong | fusion_correct_when_opening_wrong | fusion_wrong_when_opening_correct |
| --- | --- | --- | --- | --- | --- | --- |
| A0-Triplet | 1.75 | 18.25 | 25.75 | 4.25 | 0.75 | 0.25 |
| A1-Triplet-Mean | 1.75 | 19.75 | 24.25 | 4.25 | 0.5 | 0.75 |
| A2-Triplet-Variance | 1.75 | 19.25 | 24.75 | 4.25 | 1 | 1 |
| A3-Triplet-MeanVariance | 2 | 18.75 | 25.25 | 4 | 0.5 | 0.5 |

Opening共50questions，6題Top1錯。Triplet補對不一定表示fusion補對：Opening90%權重與score scale可能維持原prediction。另記fusion破壞Opening正確題數，避免只報rescue。

## 11. 結論、限制與下一步

| experiment | classification |
| --- | --- |
| A1-Triplet-Mean | SEED-SENSITIVE RESULT |
| A2-Triplet-Variance | SEED-SENSITIVE RESULT |
| A3-Triplet-MeanVariance | SEED-SENSITIVE RESULT |

A2平均score低於A0，只有2/4 retrieval改善、1/4 geometry majority改善；不符合VARIANCE REGULARIZATION STABLE PROMISING。A3平均略高於A0，但只有2/4 retrieval改善、1/4 geometry majority改善；不符合COMBINED RETRIEVAL IMPROVEMENT STABLE。A1 retrieval只有1/4改善，geometry改善的兩seed並未提高retrieval。

A0 fusion平均score最高（.909650），比Opening .903435高.006215；A2/A3 fusion平均皆.903172，略低於Opening。A3純Triplet補對Opening錯題平均2題，A0/A1/A2為1.75題；但A0 fusion補對.75題、破壞.25題，net+.5題；A2補對1題但破壞1題，net0。單獨Triplet retrieval較高不等於fusion complementarity較好。

A2與A3均為SEED-SENSITIVE RESULT，不把seed42的promising當成穩定修復。下一輪值得研究能避免L2共同方向駐點的anti-collapse目標，或事前固定protocol的SupCon受控比較；不是在這輪追加lambda/gamma搜尋。本輪不實作。

僅4training seeds、同一50question DEV3，而且seed42與此DEV已用於研究；不把最高single-seed score稱final model，不作formal統計顯著性宣稱。保留不改善/退步seeds，不重抽split、不延長epochs、不改lambda/gamma/alpha。

## 12. Tests與warnings

**160 tests passed**，failures/errors/skipped=0，imports/syntax通過。Saved-result audit驗證240新增epochs＋80referenceepochs、same/different initialization、normal-only best、paired deltas/sample STD、固定TRAIN batch與Phase2.12 artifacts未改。

原Opening有1個missing color bank，沿用規則；沒有SGF preprocessing error，因本輪沒有preprocessing。未加入SupCon、ArcFace、classification、Strength Estimator、MiniZero或新TEST。

暫存aggregate在僅seed42完成時，ddof=1的sample STD無法估計，曾產生NumPy degrees-of-freedom／invalid-divide warnings並暫存NaN。這不影響training；正式四seed aggregate需全部finite，saved-result verifier會核對。沒有因warning重跑seed42或改STD定義。

訓練前第一輪新增tests有1個error：新hash verifier把舊audit的CSV檔名誤當成project-relative path。修正路徑解析後完整160 tests通過，才開始訓練；沒有更改舊split或降低測試標準。正式training failures=0。
