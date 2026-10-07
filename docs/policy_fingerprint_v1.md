# Policy Fingerprint v1：設計契約，尚未模型推論

目的：描述玩家在給定局面下，實際選手相對於固定human policy的偏離模式。H1同玩家模式可重現；H2與opening位置偏好不同；H3能補opening錯題；H4中盤增加context資訊；H5黑白行為不同。以上都是**待驗證假說**，不是Phase3.0結果。

## 每手記錄

只取CSV目標color的實際move，在落子**前**取得18-plane context。記錄game_id、player_color、total_move_number（雙方含pass）、target_player_move_number（本方含pass）、actual_move_coordinate、actual_action_index、policy_actual_probability、policy_actual_log_probability、policy_actual_rank、policy_top1_probability、policy_top1_action、policy_entropy、actual_top1/3/5、probability_gap_to_top1、rank_normalized、game_phase、is_pass。identity metadata只供grouping，不能送入network。

固定T=1、raw362-action softmax，使用穩定log_softmax。p=p(actual)，logp=log probability，rank=按logit降序排名（ties按action index升序），topK=rank≤K，entropy=−Σp logp，gap=max(p_top1−p,0)，rank_normalized=(rank−1)/361。未做legal masking；不能把raw rank解讀成合法手排名。label move不放入input。per-move TopK是feature，並非玩家retrieval accuracy。

階段固定opening1–30、middle31–150、late151+，另存overall bank；不是最佳threshold，不搜尋。Python dataclass接受已存在的per-move statistics；未實作或呼叫policy network。optional context fields由adapter／future extractor填入，topK、gap、normalized rank由aggregation導出；game_phase由總手數導出。

## 完整schema

`src.policy_fingerprint.feature_schema()`與 `outputs/phase30/policy_schema_v1.json`列出全部 **468個 numeric fields**。12banks：black/white/combined × opening/middle/late/overall。每bank有：

| metric（7） | summary（每項5） |
|---|---|
| actual_probability、actual_rank、negative_log_probability、entropy、top1_probability、probability_gap、rank_normalized | mean、std、median、p25、p75 |

再加top1_rate、top3_rate、top5_rate、pass_rate，共12×(7×5+4)=468。命名`{color}_{phase}_{summary}_{metric}`；例`black_opening_top5_rate`、`white_middle_mean_actual_rank`、`black_late_mean_entropy`。coverage欄位包含move_count、各metric valid_count、valid_game_count、missing_mask，**不加入similarity向量**。

position→game：只保留target-side，分色分階段統計，std用population ddof0。game→player/query：每個有此feature的game等權平均，避免長盤支配；player的median欄位是game medians的平均，std欄位是game stds的平均，並非把所有move pooling後重算。caller須先依player_id分組；同一game不得混player或換色，owner_id只留metadata並在player聚合時驗證，不寫進numericfeature。沒有公開identity的query可用question_id作opaquegroup，不能取得groundtruth後回填owner。

None/NaN視為缺值；empty bank全部numeric=None而非0。單項有效數據決定分母；rank缺值不算topK失敗。Infinity／非法範圍拒絕。p=0且缺logp時NLL使用−ln(max(p,1e−12))；若有finite logp用−logp以避免softmax underflow。未來extractor須保存是否clipped。short/empty/error game記coverage，不能假成功；缺某phase不丟整盤。bank missing_mask表示沒有move，per-feature缺值以None及valid_count判定。

## Candidate/query similarity：第一版最簡單方案

首先只用B/W×三phase banks；combined與overall保留診斷，不同時重複計權。TRAIN-only估計每個feature的mean/std，std floor预先固定。candidate/query都用同一transform，僅共同有效feature計算標準化均方Euclidean distance，score=−distance。bank等權、bank內features等權；缺色／缺phase用預先記錄的shared-bank mask及coverage，不臨時用看到query/VAL後的fallback。無共同bank須回報不可比較，future protocol須先決定minimum coverage與tie rule，不用假零補齊。

| 方案 | 判斷 |
|---|---|
| standardized Euclidean | v1首選：尺度可控、無新模型／learnedweights |
| cosine | baseline診斷；先normalize，各種prob/rank混尺度前不可直接cosine |
| Mahalanobis | 暫緩：維度大、缺值、covariance不穩，需TRAIN-onlyshrinkage |
| feature-wise normalized distance | 可解釋版本，等權preset，不能在這輪搜尋weights |

本輪只schema／aggregation，尚未實作fit scaler或retrieval，不產生新performance。

## 座標與adapter

SGF x/y自左上，MiniZero index=(18−y)×19+x；aa342、sa360、as0、ss18、jj180；B/W共用。pass SGF空字串↔361。全部362 actions round-trip已測。教授policy_play非pass公式一致；pass必須特判，不能產生'a`'。

adapter输出one-line MiniZero dialect `GM[go_19x19]SZ[19]`，保留B/W順序、pass、komi，metadata留在record而非PB/PW輸入。非19board拒絕；setup/handicap/PL/AE明確拒絕以防丟失局面；主線外variation未納入。不是完整棋規驗證器。`sgf_before_move`排除actual與future moves；第一手用empty-prefix＋Env.reset/get_features，禁止legacyTestDataLoader空prefix直接當有效features。沒有mass SGF匯出或原CSV修改。

## Hidden fingerprint（後续獨立方案）

read-only forward hook可取最後residual output B×C×19×19，再spatial average→C-vector→同色同phase game平均→player平均→L2/cosine。hidden也可能只是board分布，對actual choice敏感度比prob/rank弱；不能直接替代Triplet且不宣稱抗collapse。需要已驗證checkpoint、cfg及GPUsmoke，這輪沒有hook execution。
