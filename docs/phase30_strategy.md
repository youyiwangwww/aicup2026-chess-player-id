# Phase 3.0：Professor Method Strategy Audit

教授一分鐘摘要：Color-aware Opening仍是目前最穩定主訊號；Triplet有小幅互補，但四seed證據不支持Mean/Variance已解決collapse。下一輪首先驗證教授Policy模型能否正確輸出**落子前**的362-action logits，再用分色分階段的policy偏離統計研究互補性。Policy與Strength都尚無可驗證pretrained權重，不能承諾clone後直接接入。單一A棋力族群下Strength不是優先項；更合理留給後續multi-rank研究。這輪只有audit、CPUadapter/tests與設計，没有新performance。

## 1. Phase 2最終結論

保留Phase212單seed改善為歷史結果；Phase213四seed顯示A1/A2/A3皆SEED-SENSITIVE RESULT，不能宣稱stableanti-collapse。Opening歷史DEV3score0.903435；A0fusionmean0.909650，但其他regularizerfusion沒有穩定收益。全部16best仍near-zero fraction100%、mean direction norm≈1。下一步應分開問「是否提供互補訊號」與「是否改善representation」，不用單一retrievalscore推導fusion或collapse。

## 2. Official baseline

官方17planes／128ch16blocks256emb／randomtriplets／all-rank大量training／position-weightedaggregation／Euclidean，與本專案64ch8blocks128emb／batchhard／limitedpositions／equalgame／cosine有實質差異。保留自己的identity-disjoint TRAIN/VAL與VALcheckpointselection，不能聲稱目前是officialreproduction。[逐項codeaudit](phase30_repo_analysis.md#official-player-identification)

## 3. Strength Estimator

rank-orderedstrengthlearning，輸出BTscalar+weight或RankNetworkprobabilities，不直接識別玩家。Go18planes256ch20blocks；BTloss需多rank與rank內positions。預訓練Go權重未驗證；rankmetadata與匿名CSV契約不匹配。[來源與限制](phase30_repo_analysis.md#strength-estimator)

## 4. MiniZero Policy Detection

humanmoveCEpolicy，18planes→362logits，pass361。legacyconverter/testloader/submission不同於2026，尤其post-last-movefeatures不能拿來算當手probability。weights `dan_training_new/model/weight_iter_400000.pkl`未在repo/release驗證；matchingcfg與build先於任何效果實驗。[稽核](phase30_repo_analysis.md#minizero-policy-detectionpolicy-branch)

## 5. 四種訊號與Player ID

| 方法 | input／訊號 | GPU | 直接identity supervision | 分色／中盤 | 成本／風險 |
|---|---|---|---|---|---|
| Opening | targetmoves位置；opening/joseki偏好 | 無 | 無，prototype識別 | 分色／主要開局 | 低；無context |
| Policy | pre-moveboard＋actualchoice相對policy | 推論需未來環境 | 無 | 分色分phase／有 | 中高；權重缺、contextconfound |
| Triplet | boardhistory；learnedidentityembedding | 已有checkpoint仍需推論 | 有TRAINidentity | 分色可聚合／有 | 中；concentration與seed敏感 |
| Strength | boardcontexts；skillconsistency | 未來推論／訓練 | rank非identity | 應分色分phase／有 | 高；單A辨識力與metadata不足 |

Policy比opening新增conditionaldecisiondeviation，Triplet是identity-trained但不保證可分，Strength是skill差距而非style。沒有一種可僅憑理論被宣稱有效。

## 6. Policy Fingerprint v1

每手prob/logprob/rank/top1prob/action/entropy/topK/gap/normalizedrank，固定1–30／31–150／151+。B/W/combined×三phase/overall的468numericfields，None/masks與coverage，equalgameplayeraggregation。第一版TRAIN-onlystandardizedEuclidean，bank等權；不搜尋weight、threshold、alpha。[完整schema](policy_fingerprint_v1.md)

Hidden方案可hooktrunk→spatialaverage→分色分phasegame/playeravg→cosine，作為另一種獨立特徵；沒有actualchoicecondition、不等於Triplet、不保證抗collapse，優先度低於顯式policy偏離統計。

## 7. Strength integration

只有train_A時不優先自訓ordinalstrength；A=1–3k不能冒稱exactrank，也不能填假對手rank。全D/C/B/A/1D…6D較適合coarsefilter、strengthconsistency、auxiliaryrepresentation與softrerank；先校準、查跨rankidentity/gameID，hardfilter可能排除真玩家。[設計](strength_integration_design.md)

## 8. Candidate architectures与fusion

| System | input／feature | score／normalization／fusion point | 成本與主要風險 |
|---|---|---|---|
| A Opening | targetcolor前5本方手heatmap | 既有color-awarecosine，既有missingcolorprotocol | 低，context不足 |
| B A＋Policy | 同candidate/querygames＋premovestatbank | −TRAINstandardizeddistance；各methodcandidatewisezscore後score-levelfixedfusion | policyinference成本；scalermissingphase／rankconfound |
| C B＋Triplet | 同games＋frozenembeddings | cosine；與B同樣candidateorder，獨立scorezscore，preregister權重 | tripletconcentration；重複訊號 |
| D C＋Strength | 同games＋verifiedstrengthbanks | softconsistencyscore，TRAINcalibration後scorelevelfusion | 最高；weight/metadata/skillshift |

以上都是integrationdesign，沒有實作newmodel/fusionrunner。固定fusion係數必須在下一個DEV-onlyprotocol開始前註冊；本輪不替新的Policy/Strength憑空挑最佳alpha。現有Opening+Tripletα0.9僅保留原historicaldiagnostic，不表示可套所有新訊號。

每種新feature先看standalone，再算samequestions的Opening wrong/newcorrect、Openingcorrect/newwrong、bothcorrect、bothwrong四格；最後看預註冊fixedfusionrescue/damage。不能用standalone排名代替互補性證據；不可因DEV分數差就重抽split，CLOSED TEST永不讀。

## 9. Compute plan

CPUNOW：audit、adapter/mapping、schema/aggregation、synthetictests、docs完成。GPULATER：build/binding、verifiedpolicyforward、largefeatures、training。savedPhase213CUDAstorage提供CUDA證據，但GPU型號/Torch/CUDAversion無法追溯；未來四字段pretraining必記。[compute](phase30_compute_plan.md)、[provenance](phase213_device_audit.md)

## 10. Phase 3.1 Policy Smoke Test

僅≤10既有TRAINplayers或幾盤SGF，verifiedweights/confighash、evalmode、不optimizer/noBackprop。14gates：build、binding、SGF、CSVadapter、targetmoves、forward、logits、actualindex、probability、rank、TopKfeature、entropy、pass、B/W；另查premovestate。未過停止，不跑Phase3.2。[逐gate契約](phase30_compute_plan.md#phase-31-policy-smoke-test下一輪主實驗)

## 11. 研究優先順序

| 順位／方法 | 研究價值／Opening互補 | GPU／工程成本 | 判斷 |
|---|---|---|---|
| 1 Policy Fingerprint | conditionalmove偏離，最貼教授policy方向 | 中高；權重和build先驗證 | 先smoke再DEV研究 |
| 2 Raw-space anti-collapse | 解釋L2projection梯度消失，直接對應已知機制 | feasibility低，正式訓練高 | TRAIN-onlyrawgradient/norm可行性先行 |
| 3 Policy Hidden Fingerprint | contextrepresentation可能互補 | policy環境可用後增量中 | 顯式policy成功後比較 |
| 4 SupCon | 更多positive/negativejointsupervision | 重訓多seed高 | 值得controlledstudy但非現在立即轉 |
| 5 Strength Estimator | skill一致性、narrowA互補未知 | 高，權重／metadata缺 | 暫緩單Aintegration |
| 6 Multi-rank strength filtering | 跨棋力候選較有價值 | 很高，需新資料稽核 | 後續multi-rank專題 |

**NEXT PRIMARY EXPERIMENT（唯一）：Phase3.1 Policy Smoke Test**，通過後才提Phase3.2fingerprint。

**SECONDARY EXPERIMENT（唯一）：TRAIN-only Raw-space Anti-Collapse Gradient Feasibility Audit**。先問normalize前的norm/varianceobjective是否有足夠gradient且不靠增加rawnorm作弊；不在本輪實作或訓練。

不建議現在立刻轉SupCon：loss換名不保證消除projection幾何瓶頸，需要batchidentity/positives、temperature、rawnorm／gradientdiagnostics和固定multi-seed控制；代價高、Opening互補未確證。Policy研究價值是與已強Opening不同的conditionalchoice訊號，符合教授方向但權重可用性仍是阻塞。

## 12. 風險与尚未確認

checkpoint/cfg/source/license/data污染未確認；MiniZeroWindows/LinuxABI未build；policybranchlegacy流程存在postmoveprefix與pass缺陷；setup／illegalSGF需顯式coverage。統計可能主要反映difficulty/skill/context而非identity；B/W與phasecoverage不足、feature尺度、sharedbank規則須預註冊。DEV已多次使用，下一效果實驗需合法DEV-onlyprotocol而非借CLOSED TEST或這輪偷建新split。沒有新performance、modelselection、alpha/lambdasearch或FINAL TEST。
