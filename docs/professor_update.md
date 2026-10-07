# Go Player Identification：教授會議更新

## 目前結論（Phase 2.13 更新；Phase 3.0 僅 static audit）

1. Color-aware Opening 是目前最穩定的主訊號：DEV3 score 0.903435。
2. Triplet 有部分 complementary signal；固定 A0 Fusion mean 0.909650，但不能把單獨 retrieval 較高等同 fusion 更好。
3. Embedding concentration 確實存在，16 個 best checkpoints 的 near-zero fraction 均100%、mean direction norm接近1。
4. Phase 2.12 單seed曾觀察A2/A3改善，但Phase 2.13確認A1/A2/A3均為 **SEED-SENSITIVE RESULT**；不能宣稱Mean/Variance穩定解決collapse。
5. 下一步先分析教授的Policy／Strength方法；Phase 3.0不training、不建立新TEST、不產生新performance score。

Phase 3.0 audit 已完成：Policy 的18-plane／362-action設計可研究局面條件下的move偏離，但必須用落子前context；legacy TestDataLoader與pass轉換不能原樣沿用。Policy／Strength預訓練權重均未驗證。**主實驗：Phase3.1 Policy Smoke Test**（少量既有TRAIN SGF、14gates）；**次實驗：TRAIN-only Raw-space Anti-Collapse Gradient Feasibility Audit**。單A資料下暫緩Strength，SupCon不立即重訓。完整設計見[Phase3.0策略](phase30_strategy.md)；以下Phase2結果仍保留為歷史紀錄。

## 研究問題與已知背景

由多盤棋譜辨識未見玩家。既有Triplet在normalize前已共同方向集中，projection後更嚴重；centering／remove-PC／whitening沒有超過原方法。所有既有TEST永久CLOSED，不作這輪選擇。

## Phase 2.12-A／2.12-R（歷史單seed；結論已由Phase 2.13更新）

原150 TRAIN／50 VAL在任何training前因資格不足阻擋。使用者事前明確改成100／50，保留blocked紀錄；不是看結果後改規模。

新DEV3：100 TRAIN×20games；50 VAL×candidate30/query10。排除715位歷史身份，所有identity／game／exact SGF overlap=0。四組同seed42、初始化、架構、cache、sampling與20epochs，只改固定mean／variance regularizer。

## 核心結果

| experiment | best_epoch | top1 | top3 | top5 | competition_score |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | 16 | 0.560000 | 0.700000 | 0.760000 | 0.585956 |
| A1-Triplet-Mean | 19 | 0.540000 | 0.720000 | 0.780000 | 0.590602 |
| A2-Triplet-Variance | 10 | 0.600000 | 0.680000 | 0.840000 | 0.634879 |
| A3-Triplet-MeanVariance | 19 | 0.600000 | 0.740000 | 0.820000 | 0.645555 |

Best epoch只依normal DEV3 score，color-aware／fusion不參與選擇。

| experiment | mean_direction_norm | raw_pc1_explained | cosine_separation | between_within_ratio | classification |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.999999922473 | 0.886261 | 6.3287219e-09 | 0.398375 | A0 REFERENCE |
| A1-Triplet-Mean | 0.999999935276 | 0.747456 | 6.1774347e-09 | 0.414580 | RETRIEVAL IMPROVED WITHOUT CLEAR GEOMETRY FIX |
| A2-Triplet-Variance | 0.999999808790 | 0.790375 | 1.1640872e-08 | 0.356794 | PROMISING REPRESENTATION INTERVENTION |
| A3-Triplet-MeanVariance | 0.999999933257 | 0.830781 | 6.2502277e-09 | 0.403373 | RETRIEVAL IMPROVED WITHOUT CLEAR GEOMETRY FIX |

原A0是否仍達collapse警示：True；A1共同方向是否降低：False；A2spread是否增加：True；A3是否normal retrieval最高：True。不能只因effective rank高就判成功。

歷史單seed標籤為 **PROMISING REPRESENTATION INTERVENTION**，只代表當時A2的方向性觀察；**不是目前穩定結論**。Phase 2.13已更新為SEED-SENSITIVE RESULT。四組mean direction norm仍接近1、near-zero fraction為100%；不能宣稱collapse已解決。

## Secondary diagnostics

Opening reference score=0.903435。Color-aware提升的實驗：A1-Triplet-Mean, A3-Triplet-MeanVariance。固定alpha0.9 fusion高於Opening的實驗：A0-Triplet, A2-Triplet-Variance, A3-Triplet-MeanVariance；沒有alpha search。

## 限制與下一步

單seed／單DEV3、50questions，候選僅DEV3 research candidate，非final model。Geometry與retrieval分開判斷，沒有建立新FINAL TEST。

以上是Phase 2.12當時的研究建議。多seed驗證已在Phase 2.13完成，未支持穩定anti-collapse改善；目前改為先audit教授Policy／Strength方法，再決定後续受控比較。

**143 tests／imports／syntax通過；原blocked records與歷史artifacts保留。** 不加入Strength Estimator／MiniZero／classification head，不建立FINAL TEST4。完整結果見[phase212_results.md](phase212_results.md)。

## Phase 2.13 multi-seed stability

固定DEV3 split/cache，seed42引用保存結果，新增123/2026/31415；每seed四objective同初始化，20epochs。只改training RNG，沒有新TEST或final model。

| experiment | competition_score_mean | competition_score_std | retrieval_improved_count | geometry_majority_improved_count |
| --- | --- | --- | --- | --- |
| A0-Triplet | 0.601772 | 0.0251582 | reference | reference |
| A2-Triplet-Variance | 0.588369 | 0.043013 | 2 | 1 |
| A3-Triplet-MeanVariance | 0.606316 | 0.0438915 | 2 | 1 |

A2/A3皆 **SEED-SENSITIVE RESULT**，未達各自穩定門檻。A3平均略升、但2/4改善，A2平均下降；四seed/16模型near-zero dimensions皆100%，collapse未解決。

固定alpha=.9 Fusion與Opening complementarity（跨seed平均題數；Opening score=.903435）：

| experiment | competition_score_mean | competition_score_std | opening_wrong_triplet_correct | fusion_correct_when_opening_wrong | fusion_wrong_when_opening_correct |
| --- | --- | --- | --- | --- | --- |
| A0-Triplet | 0.90965 | 0.00935694 | 1.75 | 0.75 | 0.25 |
| A1-Triplet-Mean | 0.901444 | 0.0117675 | 1.75 | 0.5 | 0.75 |
| A2-Triplet-Variance | 0.903172 | 0.0126596 | 1.75 | 1 | 1 |
| A3-Triplet-MeanVariance | 0.903172 | 0.0123406 | 2 | 0.5 | 0.5 |

Mean loss接近1時，L2 Jacobian可能讓raw tangential gradient接近零；這輪TRAIN-only gradient diagnostic與synthetic test分開驗證normalized/raw梯度。Geometry仍須看絕對量級，不能以effective rank或最高seed score宣稱collapse修復。A2與A3均為SEED-SENSITIVE RESULT，不把seed42的promising當成穩定修復。下一輪值得研究能避免L2共同方向駐點的anti-collapse目標，或事前固定protocol的SupCon受控比較；不是在這輪追加lambda/gamma搜尋。本輪不實作。

160 tests通過；完整結果：[phase213_results.md](phase213_results.md)。
