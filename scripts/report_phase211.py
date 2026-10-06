"""Explain saved DEV2 geometry only; no CLOSED TEST reads, inference or model selection."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase211_guard import dev_only
from src.metric_utils import write_json,file_digest


def table(rows,columns,precision=6):
    """Use scientific notation when geometry magnitudes would round to zero."""
    def display(value):
        if isinstance(value,(float,np.floating)):
            return f'{value:.{precision}g}' if precision>6 or value!=0 and abs(value)<1e-5 else f'{value:.6f}'
        return str(value)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(display(row[key]) for key in columns)+' |' for row in rows)
    return '\n'.join(lines)


def main():
    """Produce technical forensics and a concise professor brief from existing DEV artifacts."""
    with dev_only():
        d=ROOT/'outputs/phase211'
        s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        v=json.loads((d/'verification.json').read_text(encoding='utf-8'))
        if s['status']!='COMPLETE EXPLORATORY DEV-ONLY FORENSICS' or v['failures'] or v['errors']:
            raise ValueError('Reporting requires completed DEV diagnosis and passing tests')
        if file_digest('outputs/phase28/experiments/Triplet-Hard-100/best.pt')!=s['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        game=[row for row in s['geometry'] if row['level']=='game']
        position=[row for row in s['geometry'] if row['level']=='position']
        lookup={row['layer']:row for row in game}
        bb,raw,norm=[lookup[layer] for layer in ['backbone_feature','raw_embedding','normalized_embedding']]
        pca=s['pca']
        # Complete the requested single-file PCA schema from already saved TRAIN statistics.
        for row in pca:
            with np.load(d/(row['layer']+'_train_statistics.npz')) as fitted:
                row.update({f'eigenvalue_{i+1}':float(value) for i,value in enumerate(fitted['eigenvalues'][:20])})
        pd.DataFrame(pca).to_csv(d/'pca_diagnostics.csv',index=False,encoding='utf-8-sig')
        eigenvalues=pd.read_csv(d/'pca_top20_eigenvalues.csv').to_dict('records')
        metrics_columns=['method','top1','top3','top5','competition_score']
        fit_manifest={'fit_partition':'DEV2 TRAIN','fit_games':4000,'fit_players':200,
                      'source_sha256':s['split_sha256']['train'],'no_val_labels_or_stability_used':True,
                      'checkpoint_sha256':s['checkpoint_sha256'],
                      'statistics_sha256':{row['layer']:file_digest(d/(row['layer']+'_train_statistics.npz')) for row in pca},
                      'fitting_granularity':'equal-position game means; PCA population covariance; B/W means split by TRAIN color',
                      'whitening_eigenvalue_floor':1e-5,'removed_pcs':[0,1,2,4,8]}
        write_json(fit_manifest,d/'fit_statistics_manifest.json')
        best=s['diagnostic_best_postprocessing']
        prior,new=s['diagnostic_fusion']
        fusion_delta=new['competition_score']-prior['competition_score']
        warnings='\n'.join('- '+warning for warning in s['warnings'])
        text=f'''# Phase 2.11：Embedding Collapse Forensics

## 教授 1 分鐘摘要

已保存的正式 TEST3 結果：Color-aware Opening score **0.773919**；frozen Fusion **0.790240**，提升 +0.016321 的 95% CI **[−0.022885, 0.052726]** 包含 0。這些數字只作研究背景，不作本輪選擇依據；没有開啟任何 CLOSED TEST inputs／scores，也沒有重新 inference。

本輪只用 DEV2。共同方向在 **pooled backbone 就已出現**，projection 後角度集中更嚴重：game cosine separation 從 **{bb['cosine_separation']:.4g}** 降至 **{raw['cosine_separation']:.4g}**。Normalize 前 raw cosine 已接近 1，因此 **L2 normalize 不是主因**。現有 projection 是單一 `Linear(576,128)`，沒有新增 MLP 或改 forward。

Raw 的 variance 沒有歸零，但其 TRAIN PC1 吃掉 **{pca[0]['top1_explained']:.2%}** variance，且與共同均值方向高度對齊；normalized effective rank **{pca[1]['effective_rank']:.2f}**，仍有高維微小訊號，不能稱為完全恆定輸出。問題更像 **共同均值方向／projection 的角度集中，加上色彩域差異與身份訊號弱**。

Current／RawMean／RawGameNormalize score 都是 **0.417385**；remove-PC、whitening 均未超過 Current。Color-aware + global centering 的 DEV2 Triplet score **{best['competition_score']:.6f}**，但固定 fusion 從 **{prior['competition_score']:.6f}** 降至 **{new['competition_score']:.6f}**。沒有選出新正式模型。

## 1. 隔離與固定 checkpoint

Round 1、FINAL TEST 2、FINAL TEST 3 的 CLOSED／consumed markers 均存在。所有 Phase 2.11 I/O entry points 拒絕其 candidate／query／truth／score matrix，以及 Stability raw inputs。只允許 DEV2 TRAIN／VAL；歷史正式數字取自需求與已整理文件，不載入 CLOSED score arrays。

- Checkpoint：`outputs/phase28/experiments/Triplet-Hard-100/best.pt`，epoch 18。
- SHA256：`{s['checkpoint_sha256']}`，執行前後完全相同。
- 64 channels／8 residual blocks／128 embedding；production model/config 原始 SHA256 通過。
- DEV2 TRAIN：200 players／4,000 games／64,000 positions；VAL：100 players／candidate 3,000／query 1,000／100 questions，共 64,000 positions。
- 每盤固定 16 positions，全部 cache provenance 與既有 split SHA256 相符；没有重新抽樣／改棋譜。
- TRAIN 與 VAL player identities 交集為 0；原 checkpoint 訓練的 100 位玩家是此 TRAIN pool 的子集。

{table(s['feature_coverage'],['partition','total_games','successful_games','failed_games','positions'])}

## 2. Representation 擷取與計算定義

`PlayerEncoder.forward()` 完全未修改。Diagnostic hooks 捕捉 projection 的輸入（pooled backbone 576 維）、projection output（raw 128 維），以及原 forward 最終 normalized 128 維；同一次 forward 取得三者。`model.eval()`／no_grad/inference_mode，不建立 optimizer、不 backward、不更新 BatchNorm。

Variance/norm 分別計算 **所有 64,000 VAL positions** 與 **4,000 equal-position game means**。Position 同／異玩家各固定 seed 42 抽 20,000 pairs；game-level 使用全部 30,000 same-player／2,970,000 different-player candidate-query pairs。兩者不混用。Cosine separation=same−different；Euclidean separation=different distance−same distance。

Cosine 在 float64 中重新以每個向量的 norm 計算；Raw/Backbone 的 norm 和 variance 保留原尺度。Euclidean 先減共同原點再算距離，避免共同大均值造成 catastrophic cancellation。Variance near-zero 門檻固定 1e-8，但此指標依尺度改變，不能單獨用它定位 collapse。

### Position geometry

{table(position,['layer','variance_mean','variance_median','variance_min','variance_max','near_zero_fraction','norm_mean','norm_median','norm_std','norm_min','norm_max'],12)}

{table(position,['layer','same_cosine_mean','same_cosine_median','same_cosine_std','same_cosine_p25','same_cosine_p75','different_cosine_mean','different_cosine_median','different_cosine_std','different_cosine_p25','different_cosine_p75'],12)}

{table(position,['layer','cosine_separation','same_euclidean_mean','different_euclidean_mean','euclidean_separation'],12)}

### Game geometry

{table(game,['layer','variance_mean','variance_median','variance_min','variance_max','near_zero_fraction','norm_mean','norm_median','norm_std','norm_min','norm_max'],12)}

{table(game,['layer','same_cosine_mean','same_cosine_median','same_cosine_std','same_cosine_p25','same_cosine_p75','different_cosine_mean','different_cosine_median','different_cosine_std','different_cosine_p25','different_cosine_p75'],12)}

{table(game,['layer','cosine_separation','same_euclidean_mean','different_euclidean_mean','euclidean_separation','mean_vector_norm','centered_rms_norm','common_mean_to_centered_rms'],12)}

最早可觀察的共同方向集中在 pooled backbone，game cosine 約 {bb['same_cosine_mean']:.9f}／{bb['different_cosine_mean']:.9f}。這不代表 backbone 全部維度完全相同：只有 {bb['near_zero_fraction']:.2%} 維 near-zero。Projection raw 的 mean/scatter ratio 從 backbone {bb['common_mean_to_centered_rms']:.1f} 升到 {raw['common_mean_to_centered_rms']:.1f}，cosine 更集中。Raw/normalized cosine 幾乎相同，說明 normalize 沒有創造這個角度問題。本輪未擷取 stem／各 residual block，不能進一步斷言發生在哪個卷積層。

## 3. TRAIN-only covariance／PCA

所有 statistics 用 **DEV2 TRAIN 4,000 個 game means** fitting；先平均每盤 positions，使每盤等權。不是在 VAL／Stability／TEST fitting。PCA 用中心化 population covariance，eigenvectors 以最大絕對元素正號固定 sign；effective rank=exp(entropy(eigenvalue proportions))，participation ratio=trace²/sum(eigenvalue²)。

{table(pca,['layer','top1_explained','top5_cumulative','top10_cumulative','top20_cumulative','effective_rank','participation_ratio','total_variance','anisotropy_warning'],12)}

{table(pca,['layer','uncentered_top1_energy','mean_norm','centered_rms','mean_direction_first_centered_pc_alignment'],12)}

{table(eigenvalues,['layer','rank','eigenvalue','explained_variance'],12)}

`pca_diagnostics.csv` 同時包含 top-20 eigenvalue columns，逐 rank 版本另存 `pca_top20_eigenvalues.csv`。TRAIN mean／B/W means／components 存 NPZ，來源與 SHA256 存 `fit_statistics_manifest.json`。

**ANISOTROPY WARNING** 門檻固定 top1≥50% 或 top5≥80%；raw 觸發、normalized 不觸發。兩者未中心化 second moment 的 PC1 energy 都約 99.999987%，說明共同**均值方向**主導；不能把它和中心化後的 covariance PC1 混為一談。Raw PC1 與 TRAIN mean 對齊約 {pca[0]['mean_direction_first_centered_pc_alignment']:.6f}，大部分變化偏向長度／徑向；normalize 消除徑向變化後，剩餘很小但高 effective-rank 的角度訊號。

## 4. Aggregation／normalization

{table(s['aggregation'],metrics_columns)}

Current：normalized positions→game mean→player/query mean→最後 normalize/cosine。
RawMean：raw positions→game mean→player/query mean→最後 normalize/cosine。
RawGameNormalize：raw positions→game mean→game normalize→player/query mean→normalize/cosine。
Backbone：pooled position features→game mean→player/query mean→normalize/cosine。

三個 Triplet aggregation 的 Top-k/score 完全相同，且 Current 精確重現原 checkpoint DEV2 score；沒有證據支持「改 normalize 的時機就能解決」。Backbone score 較低；projection 有壓縮角度差異，也保留／重組部分 retrieval 訊號，不能直接稱為無用。

## 5. Mean-centering

{table(s['centering'],metrics_columns)}

每個 game embedding 減對應 TRAIN game mean後 L2 normalize，再平均同一 player/question 的 games並 L2 normalize。Raw／Normalized fitting 各使用自己的 TRAIN representation，無 VAL label fitting。Centered-Raw 明顯變差，Centered-Normalized 也低於 Current。

## 6. Remove top PCs

{table(s['remove_pc'],['removed_pcs','top1','top3','top5','competition_score','same_diff_separation','effective_rank'],10)}

固定 k=0/1/2/4/8，只 fit TRAIN raw PCA。Game raw−TRAIN mean→移除 PCs→game normalize→平均 games→player/query normalize。表內 effective rank 是同一 transform 後 TRAIN game covariance 的 effective rank，沒有利用 VAL fitting。k=0 與 Centered-Raw 相同。

移除 PC1／PC2 相對 k=0 回復部分表現，但所有 k 都沒有超過 Current。因此 dominant PC 是幾何問題的一部分，去掉它不足以恢復更強的玩家辨識；也可能丟失有用訊號。

## 7. Whitening

{table(s['whitening'],['comparison','method','top1','top3','top5','competition_score'])}

TRAIN raw mean/PCA fitting；game raw−TRAIN mean→PCA rotation→除 sqrt(max(eigenvalue,**1e-5**))→game normalize→player/query mean normalize。固定 epsilon，沒有搜尋。

{s['whitening_floored_eigenvalues']}/128 eigenvalues 被 floor；所有輸出 finite，沒有數值不穩定 warning。Whitening 改善相對純 raw centering，但仍低於 Current；未把它選成正式模型。

## 8. Color-aware Triplet／color-specific centering

{table(s['color_triplet'],metrics_columns)}

{table(s['color_centering'],metrics_columns)}

Baseline 使用 normalized-position game means，B query 只對 candidate B，W 只對 W，按 query B/W game 數加權。Missing color bank 用零 similarity，禁止跨色 fallback。本批沒有缺色 candidate bank。

Global／color-specific center 使用 TRAIN **normalized game means**；先在 game 層級減全球 TRAIN mean，或相應 TRAIN B/W mean並 normalize，再分色 aggregation。Color-aware Triplet 高於 Current；global centering 最高，color-specific 比未中心化 color-aware 好，但沒有超過 global centering。表示 color-domain shift 有影響，尚不能說每色中心才是主要修復方式。

## 9. Within／Between player separability

{table(s['separability'],['method','within_mean','between_mean','ratio'],12)}

按需求使用未平方 Euclidean mean distance；這不是統計學的 squared variance。Within=各 candidate game 到其 player centroid 距離平均；between=不同 player centroid 間距離平均。Current／RawMean 保留各自 game 尺度；best diagnostic 使用 transform 後 game vectors。Ratio 無尺度，但 color-aware 的表列 centroid 是 mixed game centroid，無法完整反映分色 score banks。

三者 between/within 都約 0.34–0.36：玩家間差異小於同玩家內的棋局差異。即使後處理提高 DEV retrieval，也沒有讓 pooled within/between ratio 明顯變好。

## 10. 固定 alpha diagnostic fusion

{table(s['diagnostic_fusion'],['method','triplet_diagnostic','alpha','normalization','top1','top3','top5','competition_score'])}

只對 DEV2，Opening 固定 Color-aware-player-5、alpha=0.9／z-score。最高 DEV-only post-processing 是 `{best['method']}`，但 new fusion 比 old fusion **{fusion_delta:+.6f}**。Triplet 單獨 retrieval 提升不保證能提供更有互補性的 fusion 訊號。

這個最高值是已使用多次的 DEV2 上 exploratory comparison，沒有獨立泛化保證；不建立 `phase211_selected.yaml`，不更新任何 frozen competition model。

## 11. 技術結論 A–H

A. 最早在 pooled backbone 已觀察到共同方向集中，projection 後角度差異進一步縮小；固定 threshold 的近零 variance 警示只在 normalized 出現，不能因此把原因歸給 normalize。
B. Raw 絕對 variance 較大／無 near-zero 維，但方向仍接近恆定、centered covariance 強 anisotropy，不能說它較健康；normalized 不是零秩，其 TRAIN effective rank 仍約 93。
C. 共同均值方向強烈主導 uncentered energy；raw covariance PC1 也對齊均值方向。兩種 dominant-direction 指標不同，但皆支持徑向／共同方向成分很大。
D. Remove-PC 只比 centered raw 好，沒有超過 Current。
E. Whitening 數值穩定、沒有超過 Current。
F. Color-aware Triplet 改善 DEV2 retrieval。
G. Color-specific centering 比單純 color-aware 改善，但低於 global centering；未支持其為最優修復。
H. 更像 **多個因素共同存在：encoder/projection 的角度集中、色彩域差異與玩家間／玩家內分離不足**；不支持 normalize／aggregation 是單一主因，也不是完全恆定 representation。

這些是固定 checkpoint 的觀察，沒有因果 intervention training，不能斷言某 loss／某 residual block 造成問題。

## 12. Tests、warnings、執行與下一步

完整 **{v['tests']} tests passed**，failure/error/skipped=0；imports／syntax check 通過。Production forward 與全部 buffers 不變；raw projection／norm≈1、TRAIN-only center/PCA/whitening、deterministic remove-PC、B/W-only matching、所有 CLOSED paths 拒絕、同一 questions 與無 optimizer/backward 全部涵蓋。

正式 DEV-only inference＋診斷耗時 **{s['elapsed_seconds']:.2f} 秒**，未訓練任何模型。Warnings：

{warnings}

明天最值得討論的模型方向：**在新的 DEV 研究中，以可監測 mean direction／variance／within-between separation 的身份辨識目標，研究防止表示角度集中，並把 B/W 色彩域因素納入控制**。先設計 diagnostics 與驗證 protocol，再考慮 loss／sampling／projection 的受控改動；不要從 TEST3 分數倒推修改。

本輪不實作新模型／loss、不改 checkpoint／epoch、不加入 Strength Estimator／MiniZero、不建立新 FINAL TEST。

```powershell
python scripts/check_phase211.py
# 已完成；runner 拒絕重做，使用保存的 DEV artifacts
python -m src.phase211_forensics --config configs/phase211.yaml
python scripts/report_phase211.py
```
'''
        (ROOT/'docs/phase211_results.md').write_text(text,encoding='utf-8')
        professor=f'''# Go Player Identification：教授會議更新

## 任務與目前正式結果

由多盤棋譜在 100 位候選玩家中辨識身份。官方資料：100,000 games／1,582 players；TRAIN／VAL／TEST 身份分開。Round 1 TEST、FINAL TEST 2、FINAL TEST 3 全部永久 CLOSED。

已保存的 TEST3：Color-aware Opening Top-1 **0.72**／Score **0.773919**；Frozen Fusion Top-1 **0.74**／Score **0.790240**。Fusion 提升 **+0.016321**，paired bootstrap 95% CI **[−0.022885, 0.052726]** 包含 0，尚不能確認改善穩定。

## Baseline evolution 與主要發現

Opening 用目標玩家實際開局落點 heatmap 識別風格；DEV2 分黑／白後 score 提升 **+0.103179**。Triplet 提供部分互補，但表示高度集中。Stability 也見到正向 fusion point estimate，CI 同樣包含 0；所有正式方法保持 frozen。

## Phase 2.11：collapse 發生在哪裡？

這輪只用 DEV2 TRAIN fitting／VAL exploratory evaluation；只載入原 Triplet-Hard-100 epoch 18 checkpoint，SHA256 不變，沒有 training、optimizer 或 backward。沒有讀 CLOSED TEST 棋局／truth／score matrix，也沒有用 Stability truth 選模型。

**共同方向在 pooled backbone 已出現，projection 後更集中；L2 normalize 不是主要原因。** Raw embedding 在 normalize 前 cosine 已接近 1。改用 raw mean 或改 normalize 時機，DEV score 都維持 **0.417385**。

Raw 的 PC1 吃掉約 **86%** variance，與均值方向對齊；normalized effective rank 約 **93**，表示仍保留微小的高維角度訊號，並非完全恆定輸出。絕對 near-zero variance 門檻受尺度影響，不能單靠它定位 collapse。

去中心、移除 PCs、whitening 都沒有超過原 Triplet；單純 remove dominant direction 不是足夠的修復。Color-aware Triplet＋global centering 的 DEV score 升到 **0.480703**，但固定 alpha=0.9 fusion 反而由 **0.758266 降至 0.743930**。Triplet 更高的獨立 score 不保證互補性更好；color-specific centering 也沒有超過 global centering。

Within／between ratio 仍約 **0.35**：玩家內棋局差異大於玩家間差異。本輪結論是 encoder/projection 角度集中、color-domain shift 與身份訊號分離不足共同存在；不支持 normalization／aggregation 是單一主因。

## 限制與下一個研究方向

DEV2 已反覆探索，本轮最高後處理結果不是新 final model；沒有新 selected YAML／frozen model。只量測 pooled backbone，尚不能定位到某 residual block；沒有訓練 intervention，不能證明特定 loss 是原因。

最值得討論：**先建立新的 DEV protocol，研究能監測並控制共同均值方向／角度集中與玩家分離度的身份辨識目標，並控制黑／白色彩域因素。** 再規劃 sampling／loss／projection 的受控研究，不以 CLOSED TEST 結果調參。

**{v['tests']} tests／imports／syntax 全通過；checkpoint／forward／historical split 全保留。** 本輪未訓練新模型、不加入 Strength Estimator／MiniZero、不建立新 FINAL TEST。完整數據見 [phase211_results.md](phase211_results.md)。
'''
        (ROOT/'docs/professor_update.md').write_text(professor,encoding='utf-8')
        print('Saved docs/phase211_results.md and concise docs/professor_update.md; DEV artifacts only')


if __name__=='__main__':
    main()
