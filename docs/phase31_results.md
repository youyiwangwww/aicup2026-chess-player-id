# Phase 3.1 CPU Policy Smoke

## 1分鐘摘要

**STRUCTURAL NETWORK PASS / BINDING BLOCKED**。

教授pinned AlphaZeroNetwork已在CPU成功forward，18×19×19 syntheticinput→362logits／softmax／value、每手probability/rank/TopK/entropy及temporaryhiddenhookshape均通過。這是**RANDOM INITIALIZED NETWORK、STRUCTURAL SMOKE ONLY、NOT POLICY MODEL、NOT RESEARCH RESULT**。沒有verifiedcheckpoint，WSL也缺C++dependencies，因此沒有真實MiniZero18-planefeatures、沒有tinybootstrap、沒有fingerprintaggregation／retrieval，**NOT READY FOR PHASE 3.2**。

## Environment

| 欄位 | 本輪保存值 |
|---|---|
| OS | Windows11，10.0.26200；既有UbuntuWSL2可用 |
| Python／Torch | 3.12.10／2.14.1+cpu |
| torch.cuda.is_available | false；此smoke使用CPU版Torch，不表示硬體沒有GPU |
| device | cpu，Intel Core 7 240H |
| CPU cores | physical10／logical16；Torchthreads4 |
| RAM | 16,790,892,544bytes，15.64GiB |
| cuda_version | null（CPU版Torch） |

實際值：[environment.json](../outputs/phase31/environment.json)。沒有安裝WSL、Docker、BuildTools或系統套件。

## Professor source

完整clone在ignored `external_refs/minizero_policydetection/`；既有WSLGit2.53.0執行clone及detachedcheckout。

- commit：`b44b70f53de6cdaa7e2f44a590d541492148a9ad`
- git tree：`b65c9dbb05ef7f459d003b97f3f2fb4e31c2577c`
- 原network_unit.py／alphazero_network.py SHA256与Phase3.0pinned來源一致；沒有重寫architecture或改教授原始碼。

[source_audit.json](../outputs/phase31/source_audit.json)與[source_git_identity.txt](../outputs/phase31/source_git_identity.txt)保存證據。沒有commit整個外部repo。[教授network來源](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/network/py/alphazero_network.py)

## Checkpoint availability

**NOT AVAILABLE**。完整clone（含使用者可能放入的未tracked檔案，排除.gitinternals）搜尋.pkl/.pt/.pth/.ckpt/.onnx為空；hardcoded `dan_training_new/model/weight_iter_400000.pkl`不存在。重新查GitHubreleases為0；repo-localREADME/scripts原始碼downloadlink搜尋無verifiedpretrainedlink。沒有deserialize任何checkpoint。

腳本referencedcfg `dan_training_new/go_19x19_gaz_6bx256_n18-0c403e.cfg`也不存在，故checkpoint/cfg匹配是**不可確認**，而非不匹配的實測結果。`ProfessorCheckpointBackend`／`VerifiedProfessorBackend`僅保留拒絕未驗證載入的futureinterface，不偽造可用性。

## C++ build

WSL可用，故完成Linuxdependencypreflight；不是「nativeWindows沒有WSL」。已有g++15.2.0、pkg-config2.5.1；缺cmake、Torch／LibTorch、Boost、OpenCV、ALE。詳見[wsl_dependency_probe.log](../outputs/phase31/wsl_dependency_probe.log)。

依「dependency齊才build」條件，**沒有執行** `scripts/build.sh go release`；buildexit_code=null，不捏造編譯失敗exitcode。沒有build/go/minizero_py可import，因此bindinggateBLOCKED。預期command、dependencylog、reason保存於[build_audit.json](../outputs/phase31/build_audit.json)。沒有無限修環境或自動裝dependencies。

## Python network structural smoke

直接importclone的`minizero/network/py/alphazero_network.py`，使用privatePythonnamespace處理relativeimports。matchingcfg缺失，因此僅用repo **example.cfg** 的13blocks／256channels／valuehidden256；Go18channels、19×19、362actions是已稽核sourcecontract，scalarvaluehead。**不是hardcoded6-blockcheckpointreproduction**；examplecfg僅structuralrole，其SHA256保存於networkcontract。

RandomStructuralBackend固定seed42、cpu、eval＋inference_mode、無optimizer／backprop。輸出logits/policy1×362、value1×1，全部finite；softmaxrow sum0.9999999404。Temporaryhook最后residual得到1×256×19×19、spatialmean1×256，finally移除hook，不做hiddenretrieval。

[network_contract.json](../outputs/phase31/network_contract.json)、[hidden_contract.json](../outputs/phase31/hidden_contract.json)全部標random_weights=true、meaningful_policy_statistics=false。

## MiniZero feature pipeline

**BLOCKED**，不建立假的exact18-planeextractor。feature_trace只有adapter的move trace，不含plane sums，明確real_features_obtained=false；不能宣稱AI CUP SGF→exactMiniZerofeatures已完成。

TRAINsubset：只讀Phase212DEV3 `splits/train.csv`，seed42抽10players、每人5games，共50games。manifest保存identityhash、gameIDs、TRAINCSV與原officialCSV SHA256；不是新研究split，沒有讀VAL／Stability／任何CLOSED TEST。

Random structural只使用其中5games的前4targetmoves（20筆metadata）；**每個forwardinput是獨立synthetictensor，不是對應SGF局面**。SGF只驗證parser、actualtargetcolor、座標和passmetadata。Replayinterface及mockunit已驗證reset→get_features→forwardcallback→act順序；第一手讀reset前局面。這只是呼叫順序契約，不是C++棋規／historyplanes已驗證。

新發現：Env.act(vector<string>)要GTP `A1..T19`（跳過I）／`PASS`，不是SGF `aa..ss`；已依原BaseBoardAction／SGFLoader轉換接口。aa→A19、ss→T1、pass→PASS；realbinding仍須後續驗證。[原coordinateparser](https://github.com/b08202011/minizero_policydetection/blob/b44b70f53de6cdaa7e2f44a590d541492148a9ad/minizero/utils/sgf_loader.cpp)

## Policy output validation

`policy_move_statistics.py`以float64穩定log-softmax計算actualprob/logprob/rank、top1prob/action、entropy、Top1/3/5flags、gap及normalizedrank。排序logitdescending，tieactionindexascending，含pass361；underflow時保留finiteactual logprob。uniformentropy≈ln362、最高手rank1、第三名top3與equal-logitsties皆通過synthetictests。

每手TopK只是movefeature，不是Player-IDaccuracy。[random_structural_move_diagnostics.json](../outputs/phase31/random_structural_move_diagnostics.json)每筆都有random／meaningless／syntheticinput標籤。Randombackend拒絕fingerprintaggregation；本輪沒有呼叫aggregate_game/player產生randomresearchfeature。

## CPU runtime

20syntheticpositions，batch1、CPU4threads，forward累計0.637324秒，約**31.38 positions/sec**；mainprocessing0.975480秒（不含Pythonimports/startup、clone、sourceaudit及tests）。這不是SGFfeaturepipeline吞吐量，也不是verifiedpolicy實驗benchmark。因補齊actual_log_probability輸出欄位而重跑structuralcontract；固定seed與move set未變，沒有用重跑選模型。最終計時見[runtime.json](../outputs/phase31/runtime.json)。Random權重沒有保存為正式checkpoint。

## Tiny bootstrap（如果有）

**NOT EXECUTED**。前提realMiniZerofeaturepipeline沒有PASS；matching教授cfg也缺失。不建立fakefeatures、不猜bootstraparchitecture、不執行200-steptraining。runtime／loss為N/A；沒有TinyBootstrapcheckpoint或aggregation。若未來取得verifiedprofessorcheckpoint應優先驗證，不bootstrap。

## 14 gates

| gate | 狀態 | 證據範圍 |
|---|---|---|
| 1 build | BLOCKED | dependencypreflight不齊，未build |
| 2 binding import | BLOCKED | 沒有compiledbinding |
| 3 SGF read | PASS | TRAINSGF經sgfmillparse；非C++loader |
| 4 AI CUP adapter | PASS | targetmetadata／B/Worder／pass |
| 5 target moves | PASS | CSVtargetside篩選 |
| 6 network forward | STRUCTURAL_ONLY | random教授原class |
| 7 logits B×362 | STRUCTURAL_ONLY | finite、shape、softmax |
| 8 actual action index | PASS | CPUmapping／cornerround-trip，不表示binding對齊已通過 |
| 9 probability | STRUCTURAL_ONLY | syntheticlookup／log-softmax |
| 10 rank | STRUCTURAL_ONLY | stableties／actualrank |
| 11 Top1/3/5 | STRUCTURAL_ONLY | move-levelsyntheticflags |
| 12 entropy | STRUCTURAL_ONLY | uniformreference |
| 13 pass | STRUCTURAL_ONLY | index361、mockGTPpass，非realenvpass |
| 14 B/W pre-move feature | BLOCKED | 沒有exactfeatures／turn/historyaudit |

機器可讀：[gates.json](../outputs/phase31/gates.json)。不能把上述4PASS／7STRUCTURAL_ONLY／3BLOCKED解讀為14PASS。

## Blocking issues

最少兩項：①在可用Linux/WSL提供既有matchingCMake/LibTorch/Boost/OpenCV/ALE環境並buildbinding；②教授提供verifiedcheckpoint與matchingcfg（來源、SHA256、授權、資料provenance）。接著才可用TRAIN少量SGF做真实pre-move18-plane／turn／history／passaudit。這轮没有要求自動安裝或自訓來繞過阻擋。

## Ready for Phase 3.2?

**NOT READY FOR PHASE 3.2**。

總216tests（原191先執行＋新增25）通過，所有srcimports與src/scripts/testssyntax通過。Phase212／213／30 artifacts前後hash不變，frozenprotocolsource/artifactbindings驗證通過。原有tests有暫存synthetictrainingfixtures，僅為regressiontests，不是本輪bootstrap／正式效果實驗。[verification.json](../outputs/phase31/verification.json)

沒有PlayerTop1/3/5、CompetitionScore、Openingfusion、Policyretrieval、新TEST、VALmodelselection。CLOSED TEST永久保持CLOSED。
