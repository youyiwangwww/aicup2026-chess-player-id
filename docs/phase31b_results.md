# Phase 3.1B：MiniZero Binding & Real Feature Pipeline Enablement

## 1 分鐘摘要

**ENVIRONMENT BLOCKED**（2026-10-07）。Windows 與既有 Ubuntu WSL 均未找到 Docker／Podman；標準 Docker Desktop／Podman 安裝位置也不存在。教授 `kds285/minizero:latest` 未拉取、未啟動。WSL 缺 CMake、Torch／LibTorch、Boost、OpenCV、ALE，manual feasibility audit 亦不滿足 build 前提。沒有安裝系統套件、沒有修改教授 source、沒有 fake feature fallback。

原 216 tests 全部通過；新增 23 tests 中 **14 通過、9 skipped**。總共 230 通過、9 skipped，不能宣稱 239 全部通過。8 個 real binding tests 與 1 個 bootstrap checkpoint integration test 因前提未成立而 skipped。src imports／syntax、歷史 artifacts SHA256 與 Phase 2.13 frozen bindings 均通過。

本輪沒有 Player-ID performance、VAL、Stability、CLOSED TEST、FINAL TEST、retrieval、Opening fusion 或 model selection。原測試的 synthetic fixtures 僅屬 regression tests。

## Container

教授原 `scripts/start-container.sh` 確實指定 `kds285/minizero:latest`。本輪沒有使用其他 image tag，也沒有自動安裝 Docker Desktop。

- Windows：PATH 無 docker／podman；`C:/Program Files/Docker/Docker/resources/bin/docker.exe` 與 `C:/Program Files/RedHat/Podman/podman.exe` 不存在。未存在的命令沒有偽造 version 或 info。
- WSL：runtime path 均為 null。Linux x86_64、Python 3.14.4、g++ 15.2.0、pkg-config 2.5.1、Git 2.53.0 可用。
- WSL 缺 CMake、Python Torch、Boost header、OpenCV pkg-config、ALE files；搜尋 `/usr/local`、`/opt` 未找到 Torch／ALE CMake configs。
- Windows 既有 portable Python 的 Torch 2.14.1+cpu 可供測試；它不是 WSL LibTorch，不能用來宣稱 Linux binding dependencies 已齊。
- 沙箱起初拒絕 WSL service access；取得工具核准後，已在沙箱外完成唯讀 WSL probe，沒有把權限錯誤誤報成 WSL 不存在。

證據：[container_audit.json](../outputs/phase31b/container_audit.json)、[manual_environment_audit.json](../outputs/phase31b/manual_environment_audit.json)、[wsl_probe_command.json](../outputs/phase31b/wsl_probe_command.json)。[container_environment.json](../outputs/phase31b/container_environment.json) 明確是 BLOCKED，digest／size／pull time／container software versions 全部 null。

## Build 與 Binding

WSL Git 確認教授 commit 為 `b44b70f53de6cdaa7e2f44a590d541492148a9ad`，tracked source status 為空。原 network、environment、SGF loader 沒有修改。

`scripts/build.sh go release` **未執行**；command_executed=false、exit_code=null、elapsed=null。沒有 `build/go/minizero_py*` artifact，未嘗試把 Python mock 當作 binding import。

證據：[build.log](../outputs/phase31b/build.log)、[build_audit.json](../outputs/phase31b/build_audit.json)、[binding_contract.json](../outputs/phase31b/binding_contract.json)、[source_audit.json](../outputs/phase31b/source_audit.json)。若後續需要 compatibility patch，必須 STOP 並記錄 SOURCE_PATCH_REQUIRED。

## Real Env 與 18-plane semantics

Real Env reset／get_features／act 均 **未執行**，無 real tensor 或 plane sums。

原 `go.cpp::GoEnv::getFeatures()` 定義已讀取：planes 0..15 為最近八個 post-action boards，偶數／奇數 plane 分別使用 **當下 side-to-move** 的 own／opponent 視角。Plane 16 為 Black turn 全 1，plane 17 為 White turn 全 1；另一個 turn plane 全 0。Reset history 為空；pass 會把未變的 board 加入 history、切換 turn 並移動 history。

以上是 **SOURCE_ONLY_NOT_RUNTIME_VERIFIED**。不能當作實際 binding semantics PASS。[real_feature_trace.json](../outputs/phase31b/real_feature_trace.json) 保存定義與空 records。

從教授現有 `example.cfg` 僅取 `env_board_size=19`、`env_go_komi=7`、`env_go_ko_rule=positional`，產生 [feature_only.cfg](../outputs/phase31b/feature_only.cfg)。標記 **FEATURE PIPELINE ONLY**、not_checkpoint_matching=true；未載入 binding 驗證。[feature_cfg_audit.json](../outputs/phase31b/feature_cfg_audit.json) 包含 source／derived cfg SHA256。

## Pre-move replay 與 Coordinate validation

新增 [phase31b_pipeline.py](../src/phase31b_pipeline.py) 提供真實 binding import、source semantics assertions、四角／中心／pass placement 檢查，以及使用既有 `replay_pre_move()` 的 instrumentation 與兩次 exact equality replay。只有 compiled binding 可使用，不提供 feature extractor fallback。

`audit_real_env()` 檢查 move 1／2／3、pass 前後、B/W、history shift 與全部 18 planes。Expected masks 僅為無 capture 的短序列 assertions，絕不當作 network input。`audit_replay()` 在 callback 檢查已 act 的 moves 數必須等於 actual move number − 1，保存 metadata／plane diagnostic／tensor hash，之後才 act actual move；每盤最多 6 target moves。

目前 aa→A19、sa→T19、as→A1、ss→T1、jj→K10、pass→PASS 僅通過 Python mapping contract。Real SGF→GTP→Env agreement、B/W TRAIN replay、pass、reproducibility **仍 BLOCKED**。本輪沒有建立 TRAIN replay manifest，因為 binding 不存在；不得宣稱完成 ≤5 players、≤2 games/player 的 real replay。新增整合測試使用明確標記的短 SGF fixture，也因 binding 不存在而 skipped。

## Tiny bootstrap、CPU runtime 與 checkpoint

**NOT EXECUTED**。Real feature pipeline 沒有 PASS，bootstrap eligibility 必須拒絕。

`trainpolicy.sh` 引用名稱包含 6bx256_n18 的 cfg，但該 cfg 不存在；原 AlphaZeroNetwork class 接受 blocks／channels 等參數，沒有提供 missing cfg 的實際內容。本輪未把檔名推導當作三方完整核對，也未猜測 value head／regularization 設定。

已確認 `train2.py` 非 gumbel policy loss 為 one-hot human action 與 log-softmax 的 cross entropy；該 script 還會使用 VAL loader，因此本輪沒有直接執行它。本輪沒有新增訓練器、optimizer、Player-ID label 或 checkpoint。

Steps=0；training runtime／loss=N/A；save/load、TRAIN inference、468 fingerprint aggregation、real-feature hidden hook 均 NOT_RUN。沒有以 generic checkpoint test 或 synthetic tensor forward 代替 real bootstrap gate。Professor checkpoint 重新搜尋仍 **NOT AVAILABLE**，matching cfg 仍不存在。

證據：[runtime.json](../outputs/phase31b/runtime.json)、[source_audit.json](../outputs/phase31b/source_audit.json)。CPU safety cutoff 尚無適用的 training run。

## Gates

| # | Gate | Status |
|---|---|---|
| 1 | container runtime | BLOCKED |
| 2 | professor container | BLOCKED |
| 3 | C++ build | BLOCKED |
| 4 | binding import | BLOCKED |
| 5 | Env reset | BLOCKED |
| 6 | real 18-plane features | BLOCKED |
| 7 | coordinate agreement | BLOCKED |
| 8 | pre-move ordering | BLOCKED |
| 9 | Black replay | BLOCKED |
| 10 | White replay | BLOCKED |
| 11 | pass | BLOCKED |
| 12 | history planes | BLOCKED |
| 13 | reproducibility | BLOCKED |
| 14 | real network forward | BLOCKED |
| 15 | bootstrap training | NOT_RUN |
| 16 | checkpoint save/load | NOT_RUN |
| 17 | move statistics | NOT_RUN |
| 18 | fingerprint aggregation engineering | NOT_RUN |

[gates.json](../outputs/phase31b/gates.json) 為機器可讀結果。Phase 3.1 的 structural forward PASS 保留歷史意義，但不是本輪 real-feature forward PASS。

## Tests 與保留歷史

[check_phase31b.py](../scripts/check_phase31b.py) 重跑原 216 tests，再跑新增 boundary／integration tests。涵蓋 TRAIN-only、no identity loss、禁止 Player-ID score、VAL／CLOSED TEST／FINAL TEST／Stability denial、教授 source patch denial、Phase 3.1 artifacts immutable、bootstrap 必須 real gates 與 verified architecture、real shape／coordinate／pass／pre-move／reproducibility／B/W／history／binding import 與 checkpoint integration。

結果：**216 原測試通過 + 14 新測試通過 + 9 skipped**。Imports／syntax 通過；outputs/phase212、phase213、phase30、phase31 前後 hash 不變，frozen source bindings 完整。[verification.json](../outputs/phase31b/verification.json)、[tests.log](../outputs/phase31b/tests.log)。既有 check_phase31.py 不會收集新增 phase31b tests，以維持其 original191+25 contract。

## Blocking issues 與 Next readiness

仍有多項 blocker：①缺 container runtime／完整 build dependencies；②尚無 compiled binding 與真實 feature audits；③教授 checkpoint／matching cfg 不存在；④bootstrap architecture 未完成三方核對。**不是只剩 professor checkpoint blocker。NOT READY FOR PHASE 3.2 ENGINEERING**。

環境由使用者提供後，使用官方 latest image 做 CPU smoke，mount 教授 repo 到 `/workspace`，記錄 image digest／size／pull time、container environment，確認 pinned commit 後執行 `scripts/build.sh go release` 並保存真正 stdout／stderr／exit code／elapsed。不要要求 NVIDIA GPU；CUDA build 與 CUDA available 分開記錄。若需 source patch 立即停止。

Build 成功才載入 feature-only cfg、執行 real Env／semantics／coordinate checks，以及 DEV3 TRAIN seed42、≤5 players×≤2 games、每盤≤6 target moves 的實際 replay，保存 manifest 與 metadata、兩次 exact equality。這些全部 PASS 後，才重新檢查 bootstrap architecture 與 TRAIN-only 10 players×≤5 games、≤5000 positions eligibility。現有稽核脚本若偵測到 runtime，會拒絕產生「runtime 不存在」的 blocked report，後續仍需完成真正 container orchestration。

本輪狀態 **ENVIRONMENT BLOCKED**。沒有授權外的系統安裝，也沒有進入 Phase 3.2 或 FINAL TEST。
