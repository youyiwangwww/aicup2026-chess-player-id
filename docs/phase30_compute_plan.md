# Phase 3 compute plan

## CPU NOW：本輪完成範圍

公開repo pinned source/tree/release audit；Phase213 saved metadata provenance；小型in-memory SGFadapter／19×19action mapping；preexistingmove stats dataclass/schema及game/player聚合；syntheticunit tests、imports/syntax、策略文件。沒有讀歷史CLOSED TEST、沒有抽新split、沒有正式training/inference或新performance。

原有160tests包含暫存synthetic fixtures的小型CPUforward/backprop；只因使用者要求完整原tests而執行，不是Phase3研究訓練，沒有新增正式score或checkpoint。new31tests純CPUutils，不依MiniZero/Torchforward。共191tests通過；59個srcmodules import與src/scripts/tests syntax通過。check script抑制fixtureperformance stdout，保留test結果。Phase212/213 artifacts前後SHA256相同，原protocol綁定的source／artifacts hash也通過。

## GPU LATER：未執行

MiniZero C++/binding build、verifiedweights forward、Policy／Strength training、大量policy/hiddenembeddings、多seeddeep models。build本身主要CPU，但屬未來完整GPU環境驗證工作，不在本輪執行。Windows不可假設Linuxscripts直接運行；先選匹配Docker/Linux/WSL與Torch/LibTorch/CUDAABI再測。記錄device_type/device_name/torch_version/cuda_version及weights/cfg/source/dataSHA。

## Phase 3.1 Policy Smoke Test：下一輪主實驗

前提教授提供matchingpolicycheckpoint/cfg/provenance，或先單獨核准自訓計畫；checkpoint不可得時標記BLOCKED，不用randomweights偽稱policy fingerprint。僅≤10個既有TRAIN players或幾盤TRAIN SGF，不取VAL／TEST、不評playerretrieval、不選模型。少量pre-movepositions，固定seed、eval/inference_mode，沒有optimizer/backprop。

| gate | PASS證據 | FAIL處理 |
|---|---|---|
| 1 build | pinnedC++source成功，保存編譯環境 | 停止，修環境 |
| 2 binding import | Python/C++ABI匹配 | 停止 |
| 3 SGF read | tinySGF各move完整 | 停止／adapterbug |
| 4 AI CUP row adapter | 2026quotedCSV與in-memorySGF對應 | 停止 |
| 5 target moves | CSVcolor正確含pass | 停止 |
| 6 forward | verifiedSHA/cfg，finiteoutputs | 停止，不換checkpoint挑結果 |
| 7 logits | shape B×362，18×19×19input | 停止 |
| 8 actual index | 四角center/pass對齊binding | 停止 |
| 9 probability | sum1，actuallookup與reference一致 | 停止 |
| 10 rank | stable排序/tie rule，1..362 | 停止 |
| 11 Top1/3/5 feature | rankderived與手算fixture一致 | 停止 |
| 12 entropy | natural-log有限0..ln362 | 停止 |
| 13 pass | action361、prefix與下一turn正確 | 停止 |
| 14 B/W | pre-moveown/opponent及turnplanes正確 | 停止 |

所有gates過關才允許提議Phase3.2PolicyFingerprint DEV實驗。第一手用Env.reset/get_features；不使用post-movefeatures。另檢查illegal/setup/shortgamefail分類、samplingdeterminism及hookcleanup。Phase3.2先preregistersplit／features／normalization／missingbank／fixedfusion，再測standalone與Opening四格互補性；不直接開FINAL TEST。

## Phase 3.1B 環境稽核更新（2026-10-07）

保留上述 Phase 3.0／3.1 歷史。本輪最終 **ENVIRONMENT BLOCKED**：Windows 與 WSL 均未找到 Docker／Podman，官方 `kds285/minizero:latest` 未拉取／啟動；WSL 仍缺 CMake、Torch／LibTorch、Boost、OpenCV、ALE。沒有自動安裝系統套件、修改教授 source 或啟動 bootstrap。

教授 pinned commit 正確且 tracked source clean；professor checkpoint 與 referenced matching cfg 仍 NOT AVAILABLE。建立 feature-only cfg、real binding 驗證函式、TRAIN-only guards、18 個 gates 與獨立 checker。原 216 tests 通過；新增 14 通過、9 integration skipped。230 passed／9 skipped 不代表 real feature pipeline PASS。Phase 2.12／2.13／3.0／3.1 artifacts hashes 與 frozen bindings 均保留。

下一步先提供既有可用 container runtime／官方 image CPU 環境，再 build 與 real Env／18-plane semantics／TRAIN pre-move replay。這些 PASS 且 bootstrap architecture 三方核對成功後，才可使用本輪授權的 tiny bootstrap 預算：CPU、seed42、batch8、最多200steps、≤5000 TRAIN positions；連續3steps各超過5分鐘則停止並保存 work。不得以 filename 推導當作 checkpoint matching cfg。**目前不是只剩 checkpoint blocker，NOT READY FOR PHASE 3.2 ENGINEERING**。

完整結果與證據：[phase31b_results.md](phase31b_results.md)。禁止 Player-ID performance、VAL／Stability／CLOSED TEST／FINAL TEST 保持有效。
