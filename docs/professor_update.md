# Go Player Identification：教授會議更新

## 研究問題與已知背景

由多盤棋譜辨識未見玩家。既有Triplet在normalize前已共同方向集中，projection後更嚴重；centering／remove-PC／whitening沒有超過原方法。所有既有TEST永久CLOSED，不作這輪選擇。

## Phase 2.12-A／2.12-R

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

**PROMISING REPRESENTATION INTERVENTION**。至少一個固定 intervention 在 DEV3 同時提高 retrieval，且四個主要 geometry 指標有至少三項改善。仍只有單 seed／單 DEV3，不能宣稱泛化已確認。 四組 mean direction norm 都仍接近 1，near-zero dimension fraction 都為 100%；A2 cosine separation 雖上升，仍只有約 1e-8，between/within 反而下降。方向性 promising 不等於 collapse 已解決，也不代表統計顯著改善。

## Secondary diagnostics

Opening reference score=0.903435。Color-aware提升的實驗：A1-Triplet-Mean, A3-Triplet-MeanVariance。固定alpha0.9 fusion高於Opening的實驗：A0-Triplet, A2-Triplet-Variance, A3-Triplet-MeanVariance；沒有alpha search。

## 限制與下一步

單seed／單DEV3、50questions，候選僅DEV3 research candidate，非final model。Geometry與retrieval分開判斷，沒有建立新FINAL TEST。

先確認joint improvement的多seed／新DEV穩定性，再討論SupCon受控比較；目前不需要直接跳過此intervention。

**143 tests／imports／syntax通過；原blocked records與歷史artifacts保留。** 不加入Strength Estimator／MiniZero／classification head，不建立FINAL TEST4。完整結果見[phase212_results.md](phase212_results.md)。
