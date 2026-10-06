# Go Player Identification：教授會議更新

## Phase 2.8 最新摘要（DEV2，非 TEST）

DEV2：200 TRAIN players × 20 games、100 VAL players × candidate 30/query 10，身份、game_id 與 exact SGF 互斥。

| 固定方法 | Top-1 | Top-3 | Top-5 | Score |
|---|---:|---:|---:|---:|
| Color-aware-player-5 | 0.69 | 0.84 | 0.88 | 0.739568 |
| Triplet-Hard-100（epoch 18） | 0.38 | 0.50 | 0.57 | 0.417385 |
| Fusion（alpha=0.9，z-score） | 0.71 | 0.85 | 0.91 | 0.758266 |

- Color-aware Opening 改善 **0.103179**；Fusion 比 Opening 改善 **0.018698**。
- Opening 錯的 31 題中，Triplet 單獨補對 **9 題**。
- 固定 54 人的 Opening 同色−跨色 gap **0.754040**，跨色依然很弱。
- **Embedding collapse 尚未解決**：Hard-100 separation 約 3.53e-9，全部 128 維觸發近零 variance 警示。
- **CrossColor positive 未帶來穩定改善**：配對 99 位 TRAIN，Hard 0.409307、CrossColor 0.348375；B→W 退步，同色−跨色 gap 未縮小。

Phase 2.9：100 位全新 Stability 身份，Opening／Triplet／Fusion score 為 **0.839382／0.408832／0.867889**。固定 Fusion 改善 **+0.028506**，1,000 次 paired bootstrap 差值 95% CI **[−0.010277, 0.068760]**，仍未排除零改善；embedding collapse 持續存在。92 tests 全通過，未調整模型或 alpha。

限制與下一步：DEV2 已反覆選窗口、epoch、alpha，Stability 的正向結果仍有抽樣不確定性。依指定三條件為 **READY FOR FINAL TEST 3**，下一步可規劃獨立 one-shot 驗證；本輪沒有建立或執行。Round 1 TEST、FINAL TEST 2 永久 CLOSED；不加入新模型、Strength Estimator 或 MiniZero。
