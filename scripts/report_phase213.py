"""Render saved multi-seed results without model selection or additional evaluation."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase213_guard import research_only
from src.anti_collapse import NAMES
from src.phase213_analysis import GEOMETRY,METRICS


def table(frame,columns):
    """Keep tiny gradient and geometry quantities visible with scientific notation."""
    def fmt(x):
        if isinstance(x,float):
            if .99999<x<1:
                return f'{x:.12f}'
            return f'{x:.6g}'
        return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(fmt(row[col]) for col in columns)+' |' for row in frame.to_dict('records')])


def main():
    """Append a compact professor update while retaining the complete Phase 2.12 report."""
    with research_only():
        d=ROOT/'outputs/phase213'
        s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        verification=json.loads((d/'verification.json').read_text(encoding='utf-8'))
        audit=json.loads((d/'protocol_verification.json').read_text(encoding='utf-8'))
        if verification['failures'] or verification['errors'] or not s['phase212_all_artifacts_unchanged']:
            raise ValueError('Require successful verified experiments and tests')
        frames={name:pd.read_csv(d/(name+'.csv')) for name in
                ['per_seed_results','per_seed_geometry','paired_deltas','aggregate_results','sign_consistency',
                 'mean_gradient_diagnostic','variance_gradient_diagnostic','color_results','fusion_results','error_overlap']}
        labels=frames['sign_consistency']
        aggregated=frames['aggregate_results']
        color=frames['color_results']
        fusion=frames['fusion_results']
        overlap=frames['error_overlap']
        color_aggregate=color.groupby(['experiment','retrieval'],sort=False)[METRICS].agg(['mean','std'])
        color_aggregate.columns=['_'.join(c) for c in color_aggregate.columns]
        color_aggregate=color_aggregate.reset_index()
        fusion_aggregate=fusion.groupby('experiment',sort=False)[METRICS].agg(['mean','std'])
        fusion_aggregate.columns=['_'.join(c) for c in fusion_aggregate.columns]
        fusion_aggregate=fusion_aggregate.reset_index()
        overlap_cols=['opening_wrong_triplet_correct','opening_correct_triplet_wrong','both_correct','both_wrong',
                      'fusion_correct_when_opening_wrong','fusion_wrong_when_opening_correct']
        overlap_aggregate=overlap.groupby('experiment',sort=False)[overlap_cols].mean().reset_index()
        aggregate_cols=['experiment','top1_mean','top1_std','competition_score_mean','competition_score_std','competition_score_min','competition_score_max']
        geo_cols=['seed','experiment']+GEOMETRY
        delta_cols=['seed','experiment','competition_score_delta','mean_direction_norm_delta','raw_pc1_explained_delta',
                    'cosine_separation_delta','between_within_ratio_delta','geometry_improved_directions']
        mean_cols=['seed','best_epoch','mean_loss','mean_gradient_normalized_norm','mean_gradient_raw_norm',
                   'mean_gradient_raw_tangential_norm','raw_norm_mean']
        var_cols=['seed','experiment','variance_loss','variance_gradient_raw_norm','actual_normalized_std_mean',
                  'actual_normalized_std_min','actual_normalized_std_max']
        a2_label=labels[labels.experiment==NAMES[2]].classification.iloc[0]
        a3_label=labels[labels.experiment==NAMES[3]].classification.iloc[0]
        recommendation=('A2與A3均為SEED-SENSITIVE RESULT，不把seed42的promising當成穩定修復。下一輪值得研究能避免L2共同方向駐點的anti-collapse目標，或事前固定protocol的SupCon受控比較；不是在這輪追加lambda/gamma搜尋。本輪不實作。'
                        if a2_label=='SEED-SENSITIVE RESULT' and a3_label=='SEED-SENSITIVE RESULT' else
                        '依跨seed sign consistency延伸研究；geometry需看絕對量級。下一輪可討論改良anti-collapse或SupCon受控比較，本輪不實作。')
        opening=pd.DataFrame([{'method':'Color-aware-player-5',**s['opening_reference']}])
        all_collapsed=(frames['per_seed_geometry'].near_zero_fraction==1).all()
        report=f'''# Phase 2.13：Multi-Seed Anti-Collapse Stability

所有模型 **DEV RESEARCH ONLY**。沒有final model／FINAL TEST4，沒有使用任何CLOSED TEST、DEV2或Stability作selection。

## 1. Frozen protocol與執行紀錄

DEV3 TRAIN100×20=2,000games；VAL50×candidate30/query10=1,500/500games。split seed永久42。直接沿用Phase2.12的CSV、16positions/game cache與固定TRAIN diagnostic subset，沒有重新抽identity/game、沒有重新preprocess positions。

四training seeds：42/123/2026/31415。42僅reference metadata，沒有copy checkpoint或training；其餘各四objective×20epochs，共240新epochs。training seed控制model initialization、epoch player permutation、cached-position sampling、PyTorch與CUDA RNG；cache sampling seed仍42。

所有hyperparameters與Phase2.12一致：batch16、lr.001、weight_decay.0001、margin.2、每player20triplets/epoch，2,000anchors/epoch、batch-hard。A1 lambda_mean=.1；A2 lambda_var=1；A3兩者；gamma=1/sqrt128、epsilon1e-4、population variance。regularizer在48-row source forward pool計算一次；hard-negative reuse不增加regularizer權重。

每seed四組初始化SHA相同，四seed初始化不同；best epoch只看normal DEV3 score，tie保留最早。Color/fusion/geometry都不參與selection。

Wall-clock：**{s['elapsed_seconds']/60:.2f}分鐘**（訓練、cache載入、diagnostics與hash checks）。原Phase2.12所有artifacts/hash不變；完整protocol見`outputs/phase213/protocol.json`。

## 2. 四seeds × 四experiments retrieval

{table(frames['per_seed_results'],['seed','experiment','best_epoch']+METRICS)}

## 3. Aggregate與sign consistency

STD使用跨四seeds的 **sample standard deviation（ddof=1）**，不是standard error或confidence interval。四seeds共用同一DEV3，不能視為四個獨立泛化資料集。

{table(aggregated,aggregate_cols)}

{table(labels,['experiment','retrieval_improved_count','geometry_majority_improved_count','seeds','mean_retrieval_improved','classification'])}

A2穩定promising需要平均retrieval>A0、retrieval至少3/4、geometry majority至少3/4。A3穩定retrieval需要平均>A0且retrieval至少3/4；geometry另判。Geometry主要四軸mean↓、rawPC1↓、cosine separation↑、between/within↑，每seed至少3/4才算majority；不單看effective rank。

## 4. Same-seed paired deltas

{table(frames['paired_deltas'],delta_cols)}

每row減去同seed A0，沒有跨seed錯配。全部geometry mean/std保存於aggregate_results.csv。

## 5. Best checkpoint geometry

{table(frames['per_seed_geometry'],geo_cols)}

所有best checkpoints near-zero維度fraction均100%：**{all_collapsed}**。mean direction與cosine separation需看絕對量級，方向性改善不等於collapse已解決。相同game-level幾何定義沿用Phase2.12。

## 6. TRAIN-only mean gradient diagnostic

固定TRAIN48-row forward batch：原TRAIN cache、seed42/epoch0前16triplets的a/p/n，不隨training seed或模型更換。診斷用eval-mode、無optimizer step、autograd只求embedding梯度，parameter .grad保持None；不寫checkpoint或參與selection。

{table(frames['mean_gradient_diagnostic'],mean_cols)}

L_mean=||mean(z)||²；對normalized z的梯度2mean(z)/N。對raw x須通過L2的Jacobian：(I-zzᵀ)/||x||。完全同方向時normalized梯度可以非零，但raw tangential梯度為零，正是synthetic test驗證的駐點。實際數值以表為準；不能把非零normalized gradient解讀成有效旋轉raw方向。

## 7. Variance gradient diagnostic

{table(frames['variance_gradient_diagnostic'],var_cols)}

actual std不加epsilon；loss仍sqrt(population variance+1e-4)，沒有改epsilon。完全collapsed時penalty=gamma-.01≈.07838835；非零raw梯度表示可微訊號存在，但若極小，不能推論強到足以對抗triplet dynamics。所有四objective的best checkpoints同批診斷，A2是本項主比較。

## 8. Color-aware secondary stability

{table(color,['seed','experiment','retrieval']+METRICS)}

{table(color_aggregate,['experiment','retrieval','competition_score_mean','competition_score_std'])}

Bquery只對Bbank，Wquery只對Wbank，按query game數加權；missing bank固定similarity=0，無cross-color fallback。這是secondary，沒有選color-aware checkpoint。

## 9. Fixed Opening／Fusion stability

{table(opening,['method']+METRICS)}

Opening只reference一次Phase2.12保存的score matrix，沒有重新計算。固定alpha=.9，逐question z-score fusion。

{table(fusion,['seed','experiment','alpha','normalization']+METRICS)}

{table(fusion_aggregate,['experiment','competition_score_mean','competition_score_std'])}

## 10. Opening complementarity（Top1）

{table(overlap,['seed','experiment']+overlap_cols)}

跨seeds平均question counts：

{table(overlap_aggregate,['experiment']+overlap_cols)}

Opening共50questions，6題Top1錯。Triplet補對不一定表示fusion補對：Opening90%權重與score scale可能維持原prediction。另記fusion破壞Opening正確題數，避免只報rescue。

## 11. 結論、限制與下一步

{table(labels,['experiment','classification'])}

A2平均score低於A0，只有2/4 retrieval改善、1/4 geometry majority改善；不符合VARIANCE REGULARIZATION STABLE PROMISING。A3平均略高於A0，但只有2/4 retrieval改善、1/4 geometry majority改善；不符合COMBINED RETRIEVAL IMPROVEMENT STABLE。A1 retrieval只有1/4改善，geometry改善的兩seed並未提高retrieval。

A0 fusion平均score最高（.909650），比Opening .903435高.006215；A2/A3 fusion平均皆.903172，略低於Opening。A3純Triplet補對Opening錯題平均2題，A0/A1/A2為1.75題；但A0 fusion補對.75題、破壞.25題，net+.5題；A2補對1題但破壞1題，net0。單獨Triplet retrieval較高不等於fusion complementarity較好。

{recommendation}

僅4training seeds、同一50question DEV3，而且seed42與此DEV已用於研究；不把最高single-seed score稱final model，不作formal統計顯著性宣稱。保留不改善/退步seeds，不重抽split、不延長epochs、不改lambda/gamma/alpha。

## 12. Tests與warnings

**{verification['tests']} tests passed**，failures/errors/skipped=0，imports/syntax通過。Saved-result audit驗證240新增epochs＋80referenceepochs、same/different initialization、normal-only best、paired deltas/sample STD、固定TRAIN batch與Phase2.12 artifacts未改。

原Opening有1個missing color bank，沿用規則；沒有SGF preprocessing error，因本輪沒有preprocessing。未加入SupCon、ArcFace、classification、Strength Estimator、MiniZero或新TEST。

暫存aggregate在僅seed42完成時，ddof=1的sample STD無法估計，曾產生NumPy degrees-of-freedom／invalid-divide warnings並暫存NaN。這不影響training；正式四seed aggregate需全部finite，saved-result verifier會核對。沒有因warning重跑seed42或改STD定義。

訓練前第一輪新增tests有1個error：新hash verifier把舊audit的CSV檔名誤當成project-relative path。修正路徑解析後完整160 tests通過，才開始訓練；沒有更改舊split或降低測試標準。正式training failures=0。
'''
        (ROOT/'docs/phase213_results.md').write_text(report,encoding='utf-8')
        marker='\n## Phase 2.13 multi-seed stability\n'
        professor=ROOT/'docs/professor_update.md'
        previous=professor.read_text(encoding='utf-8').split(marker)[0]
        selected=aggregated[aggregated.experiment.isin([NAMES[0],NAMES[2],NAMES[3]])]
        compact=selected.merge(labels[['experiment','retrieval_improved_count','geometry_majority_improved_count']],on='experiment',how='left').fillna('reference')
        complement=fusion_aggregate.merge(overlap_aggregate,on='experiment',validate='one_to_one')
        section=f'''{marker}
固定DEV3 split/cache，seed42引用保存結果，新增123/2026/31415；每seed四objective同初始化，20epochs。只改training RNG，沒有新TEST或final model。

{table(compact,['experiment','competition_score_mean','competition_score_std','retrieval_improved_count','geometry_majority_improved_count'])}

A2/A3皆 **SEED-SENSITIVE RESULT**，未達各自穩定門檻。A3平均略升、但2/4改善，A2平均下降；四seed/16模型near-zero dimensions皆100%，collapse未解決。

固定alpha=.9 Fusion與Opening complementarity（跨seed平均題數；Opening score=.903435）：

{table(complement,['experiment','competition_score_mean','competition_score_std','opening_wrong_triplet_correct','fusion_correct_when_opening_wrong','fusion_wrong_when_opening_correct'])}

Mean loss接近1時，L2 Jacobian可能讓raw tangential gradient接近零；這輪TRAIN-only gradient diagnostic與synthetic test分開驗證normalized/raw梯度。Geometry仍須看絕對量級，不能以effective rank或最高seed score宣稱collapse修復。{recommendation}

{verification['tests']} tests通過；完整結果：[phase213_results.md](phase213_results.md)。
'''
        professor.write_text(previous+section,encoding='utf-8')
        print('Updated Phase 2.13 full report and compact professor section; no additional evaluation')


if __name__=='__main__':
    main()
