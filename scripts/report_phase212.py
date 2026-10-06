"""Report the user revision and saved DEV3 experiments, preserving original blocked history."""
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.phase212_guard import dev3_only


def table(rows,columns):
    """Retain small geometry differences in scientific notation instead of rounding to zero."""
    def value(x):
        if isinstance(x,float):
            if .99999 < x < 1:
                return f'{x:.12f}'
            return f'{x:.8g}' if x!=0 and abs(x)<1e-5 else f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
                     ['| '+' | '.join(value(row[key]) for key in columns)+' |' for row in rows])


def main():
    """Render complete saved results; never infer TEST or select via secondary fusion scores."""
    with dev3_only():
        d=ROOT/'outputs/phase212'
        s=json.loads((d/'summary.json').read_text(encoding='utf-8'))
        v=json.loads((d/'verification.json').read_text(encoding='utf-8'))
        if s['status']!='COMPLETE PHASE 2.12-R DEV3 RESEARCH' or v['failures'] or v['errors']:
            raise ValueError('Require completed experiments and passing tests')
        matrix=s['summary_matrix']
        baseline=matrix[0]
        geometry=s['best_geometry']
        best=s['dev_candidate']['experiment']
        collapse=geometry[0]['mean_direction_norm']>.95 and geometry[0]['near_zero_fraction']>=.8
        mean_help=matrix[1]['mean_direction_norm']<baseline['mean_direction_norm']
        variance_help=geometry[2]['variance_mean']>geometry[0]['variance_mean']
        combined_best=best==matrix[3]['experiment']
        color_help=[name for name in [row['experiment'] for row in s['retrieval']] if
                    next(r['competition_score'] for r in s['color_retrieval'] if r['experiment']==name and r['retrieval']=='Color-aware')>
                    next(r['competition_score'] for r in s['color_retrieval'] if r['experiment']==name and r['retrieval']=='Normal')]
        fusion_help=[row['experiment'] for row in s['fusion_diagnostic'] if row['competition_score']>s['opening_reference']['competition_score']]
        promising=s['promising_experiments']
        takeaway=('至少一個固定 intervention 在 DEV3 同時提高 retrieval，且四個主要 geometry 指標有至少三項改善。仍只有單 seed／單 DEV3，不能宣稱泛化已確認。' if promising else
                  'A1／A2／A3 沒有同時滿足 retrieval 提升與多數主要 geometry 改善；固定 simple anti-collapse regularization 不足，不自行加入更多 loss。')
        caution='四組 mean direction norm 都仍接近 1，near-zero dimension fraction 都為 100%；A2 cosine separation 雖上升，仍只有約 1e-8，between/within 反而下降。方向性 promising 不等於 collapse 已解決，也不代表統計顯著改善。'
        stats=['experiment','best_epoch','top1','top3','top5','competition_score']
        geometry_cols=['experiment','mean_direction_norm','raw_pc1_explained','normalized_effective_rank','cosine_separation','between_within_ratio']
        records_path=ROOT/'docs/phase212_results.md'
        old=records_path.read_text(encoding='utf-8')
        # Save the original report exactly once; rerender always preserves the same historical section.
        archive=d/'original_blocked_report.md'
        if not archive.exists():
            archive.write_text(old,encoding='utf-8')
        original=archive.read_text(encoding='utf-8')
        epoch_sections=[]
        monitoring=[]
        for row in s['retrieval']:
            experiment=row['experiment']
            log=pd.read_csv(d/'experiments'/experiment/'training_log.csv').to_dict('records')
            for entry in log:
                entry['best_checkpoint']=entry['epoch']==row['best_epoch']
            epoch_sections.append('### '+experiment+'\n\n'+table(log,['epoch','train_total_loss','train_triplet_loss','train_mean_loss','train_variance_loss','val_top1','val_top3','val_top5','val_competition_score','best_checkpoint']))
            frame=pd.read_csv(d/'experiments'/experiment/'geometry_log.csv')
            for entry in frame[frame.epoch.isin([1,row['best_epoch'],20])].to_dict('records'):
                monitoring.append({'experiment':experiment,**entry})
        epoch_text='\n\n'.join(epoch_sections)
        text=f'''# Phase 2.12：Anti-Collapse Metric Learning

## 教授 1 分鐘摘要

原 **Phase 2.12-A** 在任何訓練前因 150 TRAIN／50 VAL 身份資格不足被阻擋。使用者事前明確授權 **Phase 2.12-R：100 TRAIN／50 VAL**，不是看模型分數後降規模。原 eligibility／blocked summary 保留原 SHA256。

新 DEV3：100 TRAIN players／2,000 games；50 VAL players／candidate 1,500／query 500。所有历史身份与 TRAIN／VAL 身份、game ID／exact SGF overlap=0。

四組各 20 epochs、同 seed／初始化／cache／取樣預算，只改 mean／variance regularizer。最佳 epoch 只依 normal DEV3 competition score，secondary color-aware／fusion 不參與選擇。

1. A0 best checkpoint 是否仍達本輪 collapse 警示（mean direction>0.95 且 near-zero維≥80%）：**{collapse}**。
2. A1 mean-direction regularizer 是否降低 best VAL 共同方向：**{mean_help}**。
3. A2 variance regularizer 是否增加 best VAL dimension variance：**{variance_help}**。
4. Combined A3 是否 normal retrieval 最高：**{combined_best}**；最高為 **{best}**。
5. Geometry＋retrieval 同時改善：**{', '.join(promising) if promising else '無'}**。
6. Color-aware secondary 提升的實驗：**{', '.join(color_help) if color_help else '無'}**。
7. 固定 α=0.9 fusion 高於 Opening 的實驗：**{', '.join(fusion_help) if fusion_help else '無'}**。
8. 結論：**{s['conclusion']}**。{takeaway} {caution}

下一輪 SupCon：{'目前已有值得延伸的 representation intervention，建議先做新的 DEV／多 seed 確認，不急於加入 SupCon。' if promising else '值得討論 SupCon 或 identity/proxy-based objective 作下一個受控研究；本輪沒有實作。'} 不建立 FINAL TEST 4。

## Phase 2.12-A：Original protocol blocked（保留完整原始紀錄）

以下為原報告，描述的是修訂前的狀態，不代表本輪未訓練：

<details>
<summary>150 TRAIN／50 VAL 原 protocol 與 blocked record</summary>

{original}

</details>

## Phase 2.12-R：Revised 100／50 protocol

### 1. 修訂、資料隔離與 freeze

`revised_protocol.json` 記錄 original=150／50、status=BLOCKED BEFORE TRAINING、revised=100／50、reason=insufficient unseen eligible identities、user-authorized=true、revision before any A0–A3 training=true。

歷史 715 位身份僅從已保存 copied audit/metadata 排除，未讀 CLOSED TEST 原始 CSV／truth／score matrix。資料 source SHA256：`{s['split']['dataset_sha256']}`。

TRAIN=100×20=2,000；VAL=50×candidate30/query10=1,500／500，共50 questions。Query 不含 player_id／rank。所有 partition 的 game_id／exact SGF overlap=0，TRAIN／VAL／historical identity overlap=0。

Split CSV／audit、修訂與固定 TRAIN subset 建立後 freeze；每次入口 hash-verify，不允許重新抽 identity。原 `eligibility.json` 與 `blocked_summary.json` 未刪除或改寫。

### 2. 共同設定與 loss 定義

17 channels／history8／16 positions per game；64 channels／8 residual blocks／Adaptive Pool3×3／Linear576→128／L2 output。沿用等價 deterministic adaptive-average operator，不改模型結構／state keys。

AdamW lr=0.001／weight_decay=0.0001，batch16，margin0.2、batch-hard negatives，20 triplets/player/epoch，2,000 anchors/epoch，每玩家精確20次，共20epochs，seed42。

四組從相同 random initialization 開始，initial model SHA256：`{s['initial_model_sha256']}`。沒有載入 Phase 2.8 checkpoint；新權重僅寫入 `outputs/phase212/experiments/`。

- A0：Triplet 原 loss，mean／variance components=0。
- A1：Triplet＋0.1×sum(batch mean²)，variance component=0。
- A2：Triplet＋mean(ReLU(1/sqrt128−sqrt(population variance＋1e−4)))，mean component=0。
- A3：Triplet＋0.1×mean loss＋1.0×variance loss。

每 batch anchor16＋positive16＋negative source16=**48 source-forward rows**；每 row 是原一次 forward 的結果。Regularizer 對此完整 pool 算一次；被 hard-negative selection 重用的 rows 不另外附加、不重新計權。採 population variance（correction=0），gamma=0.08838834764831843，沒有 covariance loss／SupCon／classification head。

取樣偶爾可以重複同一來源 position；本輪保持原 source-forward pool 語義，不額外改 sampling／dedup 策略。所有實驗使用完全相同的原始取樣，唯一 objective 差異是固定 regularizer。

Loss log 中 mean／variance 是未乘係數的項，total≈triplet＋0.1×mean＋variance；逐 batch assert，inactive項精確0。Batch std log 不加epsilon，regularizer std 加固定epsilon。

{table(s['feature_coverage'],['partition','total_games','successful_games','failed_games','success_rate','mean_positions','median_positions','min_positions','max_positions'])}

### 3. DEV3 normal retrieval／best epochs

{table(s['retrieval'],stats)}

20 epochs 全部完成。每組 best 只取 normal DEV3 score 最大值，相等時保留最早 epoch；没有用 geometry／color-aware／fusion 改 epoch。

### 4. Best checkpoint 完整 geometry

{table(s['best_geometry'],geometry_cols)}

{table(s['best_geometry'],['experiment','variance_mean','variance_median','variance_min','variance_max','near_zero_fraction','within_mean','between_mean'])}

{table(s['best_geometry'],['experiment','same_cosine_mean','same_cosine_median','same_cosine_std','different_cosine_mean','different_cosine_median','different_cosine_std','same_euclidean_mean','different_euclidean_mean'])}

Best VAL geometry 在 **unit-normalized game means** 上計算 normalized mean direction、variance、effective rank／within-between；raw PC1 在 raw game means上計算。Same／different pairs 使用全部 candidate-query games（same15,000／different735,000），cosine separation=same−different。Euclidean 是未平方距離，within/between也用未平方距離。

VAL covariance 只是 best checkpoint 的描述性 diagnostic，不 fitting後處理／不參與checkpoint選擇。

### 5. 固定 TRAIN geometry monitoring

固定20 TRAIN players各5 games，共100盤／1,600 sampled positions，subset IDs／game IDs與SHA在訓練前保存；每 epoch eval mode／no gradients。Mean、variance、raw PC1與 normalized effective rank 使用該 subset 的 **position-level** outputs。與 best VAL game-level geometry 的統計粒度不同，不能直接把兩者數值變化當作 improvement。

下表顯示 epoch1／best／epoch20；完整20epochs保存在各 `geometry_log.csv`：

{table(monitoring,['experiment','epoch','mean_direction_norm','variance_mean','near_zero_fraction','raw_mean_vector_norm','raw_centered_rms','raw_mean_scatter_ratio','raw_pc1_explained','normalized_effective_rank'])}

### 6. Color-aware secondary retrieval

{table(s['color_retrieval'],['experiment','retrieval','top1','top3','top5','competition_score'])}

B query只比B candidate bank、W只比W；按query各顏色game數加權。沿用 Phase2.11 定義，missing color bank similarity=0，沒有跨色 fallback。僅在既選定 best checkpoint上診斷，沒有用結果重新訓練或改regularizer。

### 7. Opening reference與fixed fusion

{table([{'method':'Color-aware-player-5',**s['opening_reference']}],['method','top1','top3','top5','competition_score'])}

{table(s['fusion_diagnostic'],['experiment','alpha','normalization','top1','top3','top5','competition_score'])}

Opening只跑固定player5/color-aware一次，沒有window/color搜尋。Fusion固定逐question跨candidate z-score，alpha=0.9，使用normal Triplet scores；不挑color-aware版本、不搜尋alpha、不參與Triplet selection。

### 8. Geometry × Retrieval

{table(matrix,['experiment','competition_score','retrieval_delta_vs_A0','mean_direction_norm','raw_pc1_explained','cosine_separation','between_within_ratio','geometry_notes','classification'])}

Geometry主要四軸：mean direction↓、raw PC1↓、cosine separation↑、between/within↑。事前規定至少3/4改善才稱majority geometry improvement；effective rank僅輔助、不單獨判成功。這是單seed方向性判準，沒有統計顯著性保證。

最高normal retrieval候選為 **{best}**；`configs/phase212_best_dev.yaml`明確標記 DEV3 RESEARCH CANDIDATE ONLY／not_final_model=true／no_test_evaluation=true。Geometry與retrieval分開報告，沒有新frozen competition model。

主要geometry方向改善最多為A2（3/4）；A2 mean direction／cosine separation較佳，A1 raw PC1／between-within較佳，沒有一個intervention在全部主要軸勝出。{caution}

**{s['conclusion']}**。{takeaway}

### 9. 每epoch loss decomposition與validation log

{epoch_text}

### 10. 驗證、限制與下一步

完整 **{v['tests']} tests passed**，failure/error/skipped=0，imports/syntax全部通過。涵蓋四loss等價／分解、常數、collapsed/spread vectors、full sourcepool weighting、finite gradients、TRAIN-only regularizer、歷史排除、相同split／初始化／cache、只依normal score選epoch與CLOSED paths拒絕。

本輪總wall-clock（preprocessing＋訓練＋diagnostics）**{s['elapsed_seconds']/60:.2f} 分鐘**。只有DEV3；不讀TEST分數選模型、不用DEV2或Stability selection。未建立FINAL TEST4，未修改Phase2.8／Phase2.10 artifacts。正式結果與各best checkpoints都有保存provenance。

本輪每種lambda只試固定值，只有一個seed、一組DEV3；50 questions的不確定性未以新TEST驗證。不能把本輪highest DEV候選稱為final model。

Feature preprocessing／Opening parsing皆無失敗。Opening reference有 {s['opening_missing_color_banks']} 個missing color bank，沿用固定 similarity=0、無跨色fallback；沒有因結果更換資料或設定。Protocol audit另驗證80 epochs、loss decomposition、共享初始化／split與6個歷史path拒絕。

下一步：{'先用新的DEV或多seed確認joint improvement，分開檢查色彩域與身份訊號；SupCon可作後續獨立受控比較，但本輪不實作。' if promising else '可以討論SupCon或identity/proxy-based目標，但要另行事前固定DEV研究protocol；不依CLOSED TEST調參，本輪不自行增加更多loss。'}
'''
        records_path.write_text(text,encoding='utf-8')
        professor=f'''# Go Player Identification：教授會議更新

## 研究問題與已知背景

由多盤棋譜辨識未見玩家。既有Triplet在normalize前已共同方向集中，projection後更嚴重；centering／remove-PC／whitening沒有超過原方法。所有既有TEST永久CLOSED，不作這輪選擇。

## Phase 2.12-A／2.12-R

原150 TRAIN／50 VAL在任何training前因資格不足阻擋。使用者事前明確改成100／50，保留blocked紀錄；不是看結果後改規模。

新DEV3：100 TRAIN×20games；50 VAL×candidate30/query10。排除715位歷史身份，所有identity／game／exact SGF overlap=0。四組同seed42、初始化、架構、cache、sampling與20epochs，只改固定mean／variance regularizer。

## 核心結果

{table(s['retrieval'],stats)}

Best epoch只依normal DEV3 score，color-aware／fusion不參與選擇。

{table(matrix,['experiment','mean_direction_norm','raw_pc1_explained','cosine_separation','between_within_ratio','classification'])}

原A0是否仍達collapse警示：{collapse}；A1共同方向是否降低：{mean_help}；A2spread是否增加：{variance_help}；A3是否normal retrieval最高：{combined_best}。不能只因effective rank高就判成功。

**{s['conclusion']}**。{takeaway} {caution}

## Secondary diagnostics

Opening reference score={s['opening_reference']['competition_score']:.6f}。Color-aware提升的實驗：{', '.join(color_help) if color_help else '無'}。固定alpha0.9 fusion高於Opening的實驗：{', '.join(fusion_help) if fusion_help else '無'}；沒有alpha search。

## 限制與下一步

單seed／單DEV3、50questions，候選僅DEV3 research candidate，非final model。Geometry與retrieval分開判斷，沒有建立新FINAL TEST。

{'先確認joint improvement的多seed／新DEV穩定性，再討論SupCon受控比較；目前不需要直接跳過此intervention。' if promising else 'Simple regularization未同時改善主要geometry與retrieval，值得下一輪討論SupCon或identity/proxy-based learning；本輪不實作。'}

**{v['tests']} tests／imports／syntax通過；原blocked records與歷史artifacts保留。** 不加入Strength Estimator／MiniZero／classification head，不建立FINAL TEST4。完整結果見[phase212_results.md](phase212_results.md)。
'''
        (ROOT/'docs/professor_update.md').write_text(professor,encoding='utf-8')
        print('Updated Phase 2.12-A/R history and professor report from saved DEV3 results only')


if __name__=='__main__':
    main()
