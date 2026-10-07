# Strength integration：design only

Strength學的是rank-ordered skill/context signal，不是identity。BT scalar score、RankNetwork rank distribution、hidden representation是不同輸出，不能互相冒稱。原repo Go loader從BR分組、要求BR/WR/PB/PW/KM/RE/SZ；本專案target可為白，CSV rank是A族群，不足以合法填全套metadata。**不產生aicup_strength_adapter.py**：目前缺失資訊不能藉adapter偽造。

| 資料場景 | 可能價值 | 限制／優先度 |
|---|---|---|
| A：只有train_A=1–3k | 固定預訓練score的一致性／context residual可能補style | narrow skill範圍；A跨原repo兩個kyu bins；無exact/opponent rank。不優先自訓Strength |
| B：D/C/B/A/1D…6D全rank | rank calibration、跨skill候選過濾與rerank較合理 | 須先稽核每bin映射、rank內game_id重複、identity跨rank、來源污染、metadata；coarse bins不等於exact rank |

| Integration | input／aggregation | score與風險 |
|---|---|---|
| A coarse filtering | 已校準rank distribution或coarse candidate metadata | soft gate優於hard排除，否則rank錯估把真玩家刪掉；單A幾乎無過濾價值 |
| B strength consistency | 同色同phase的game score/weight→equal-game candidate/query distribution | −標準化分布distance，skill相近者仍不可區分；TRAIN calibration固定 |
| C auxiliary representation | verifiedscore、rankprobs或read-onlyhidden，接獨立bank | 相似度獨立計算，避免隨意串高維style；hidden是否有identity待驗證 |
| D reranking | Opening/Policy shortlist後使用soft strength score | 固定score normalization/fusion，可能引入skillbias／domainshift |

SE與SE_infinity不是免supervision的player-ID工具。pseudo-nonhuman最低族群只能提供低端對比，不能取代多個真實棋力rank。`se_go.cfg`及`se_infty_go.cfg`實際都bt_use_weight=false；BTNetwork仍回weight，不能因此宣稱目前cfg訓練了有效weighted aggregation。

前置條件：取得verified Go checkpoint＋matchingcfg＋SHA256＋ranksetup＋training provenance＋授權；確認輸入feature前後手與target perspective；釐清匿名CSV缺metadata能否只走不依metadata的environment inference。不可填假的WR、把A寫成2k、捏造RE/PB/PW來通過training loader。真實值缺失須fail／標記不可用。來源GPL-3.0與權重授權分開審核，模型是否碰AI CUP identities尚未確證。

**PRETRAINED GO CHECKPOINT NOT VERIFIED**。本輪沒有Strength training／inference。稍後multi-rank research比現在單A接入更有意義；仍須先建立DEV-only protocol，不用CLOSED TEST驗證。
