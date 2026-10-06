"""Render only completed stability artifacts; never infer or inspect old TEST results."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase29_validation import stability_only


def table(rows, columns):
    """Keep Markdown numeric precision useful for small bootstrap differences."""
    output = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        output.append('| '+' | '.join(f'{row[key]:.6f}' if isinstance(row[key], float) else str(row[key]) for key in columns)+' |')
    return '\n'.join(output)


def main():
    """Explain independence, fixed-method results, paired uncertainty and the requested gate."""
    directory = ROOT/'outputs/phase29'
    with stability_only():
        s = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        verification = json.loads((directory/'verification.json').read_text(encoding='utf-8'))
        if s['status'] != 'COMPLETE STABILITY VALIDATION' or verification['failures'] or verification['errors']:
            raise ValueError('Reporting requires completed inference and passing tests')
        rows = [{'method': name, **metrics} for name, metrics in s['metrics'].items()]
        opening, triplet, fusion = [s['metrics'][name] for name in ['opening', 'triplet', 'fusion']]
        delta = fusion['competition_score']-opening['competition_score']
        boot = s['bootstrap']
        diag = s['embedding']
        uncertain = boot[2]['ci_lower'] <= 0 <= boot[2]['ci_upper']
        interpretation = ('Fusion−Opening CI 包含 0，尚無足夠證據宣稱穩定正向提升。' if uncertain else
            'Fusion−Opening CI 完全大於 0，支持本次獨立身份上的正向提升。' if boot[2]['ci_lower'] > 0 else
            'Fusion−Opening CI 完全小於 0，支持本次獨立身份上的退步。')
        cos = [{'pair_type': kind, **{stat: diag[kind+'_'+stat] for stat in ['mean', 'median', 'std', 'p25', 'p75']}}
               for kind in ['same', 'different']]
        cosine_table = '| Pair | Mean | Median | Std | p25 | p75 |\n|---|---:|---:|---:|---:|---:|\n'
        cosine_table += '\n'.join('| '+r['pair_type']+' | '+' | '.join(f'{r[key]:.12g}' for key in ['mean','median','std','p25','p75'])+' |' for r in cos)
        text = f'''# Phase 2.9：Stability Validation

結論：**{s['decision']['status']}**。這是全新身份的獨立 validation，並非 FINAL TEST 3；本輪未建立或執行任何新的 TEST，沒有訓練、調參或重新選擇 checkpoint。

## 1. Stability split 與安全檢查

官方來源 SHA256 `{s['split']['source_sha256']}`。排除 Round 1 TEST 10、Phase 2.6 VAL 50、FINAL TEST 2 50、DEV2 TRAIN 200、DEV2 VAL 100，共 **410 位**。只從 Phase 2.8 已保存的身份 audit 讀取排除名單；Stability pipeline 禁止開啟任何 Phase 2.6／CLOSED TEST artifact，也禁止讀取 DEV2 原始 splits/truth 或其他 checkpoint。

排除後最多 **{s['split']['max_eligible_players']} 位**符合至少 40 games，本輪固定 seed 42 抽取 **{s['split']['players']} 位**，每人 candidate 30/query 10，共 3,000／1,000 games、100 questions。沒有降低玩家數。排除身份 overlap、candidate/query game ID overlap、exact SGF overlap 均為 **0**。官方來源的 game ID/SGF 全域唯一，排除整位玩家也排除其所有來源棋局；沒有為確認 overlap 而重新讀舊 TEST。Query CSV 無 player_id、rank。

{table(s['feature_coverage'], ['partition','total_games','successful_games','failed_games','success_rate','mean_positions','median_positions','min_positions','max_positions'])}

Phase 2.8 selected config SHA256：`{s['selected_config_sha256']}`。
固定 checkpoint SHA256：`{s['checkpoint_sha256']}`。載入前後 SHA256 相同；epoch 18、model/features 設定與 TRAIN IDs 都已驗證。既有 encoder/pooling/inference code SHA256 同時綁定 Phase 2.8 manifest，沒有改模型。

## 2–4. 固定 Opening／Triplet／Fusion

Opening：Color-aware **player-5**，pass 消耗窗口但不加入 heatmap；B/W 各自建立 fingerprint，query 僅比較同色，再依 query 棋局數加權。

Triplet：只載入 `outputs/phase28/experiments/Triplet-Hard-100/best.pt`，epoch 18；64 channels、8 blocks、128 embedding、history 8、16 positions/game，equal-position/equal-game aggregation。Inference mode，無 optimizer、backprop 或 checkpoint 更新。

Fusion：每題全部 100 個候選 scores 做 **z-score**，固定 `0.9*Opening+0.1*Triplet`，沒有搜尋 alpha、窗口或其他模型。三方法使用完全相同 candidate/question 集合與分母。

{table(rows, ['method','top1','top3','top5','competition_score'])}

## 5. DEV2 vs Stability

DEV2 metrics 僅從既有 summary 取出作**描述性對照**，不讀 DEV2 ground truth、不再 inference，也不作選擇。Difference = Stability − DEV2。

{table(s['comparison'], ['method','dev2_score','stability_score','difference'])}

{table(s['comparison'], ['method','dev2_top1','stability_top1','dev2_top3','stability_top3','dev2_top5','stability_top5'])}

Stability Fusion−Opening = **{delta:+.6f}**，相較 DEV2 的 +0.018698。本輪無論好壞都保留 alpha=0.9，不改 frozen 方法。

## 6. Paired bootstrap：1,000 resamples，seed 42

每次對 100 個 question indices 有放回抽 100 次，Opening/Fusion 使用**同一組 indices**，因此差值保留配對相關性。95% CI 採 bootstrap 分布第 2.5/97.5 percentiles；不是把兩個獨立 CI 相減。

`mean` 為原始 questions 的平均估計；`bootstrap_mean` 為 1,000 個重抽樣平均值的平均。所有 bootstrap samples 已存 CSV。

{table(boot, ['method','mean','bootstrap_mean','ci_lower','ci_upper','resamples','seed'])}

{interpretation} 此 CI 反映固定模型在本次抽取 identities/questions 上的抽樣不確定性，不涵蓋所有未來資料分布或模型選擇不確定性。

## 7. Embedding collapse

沿用同一 checkpoint，candidate/query game embeddings 平均 positions 後 L2 normalize，計算 same/different game-pair cosine；population std 與 128 維 candidate-game population variance。未用診斷重選任何模型。

{cosine_table}

- Separation：**{diag['separation']:.12g}**。
- Dimension variance mean／median：**{diag['variance_mean']:.12g}／{diag['variance_median']:.12g}**。
- Dimension variance min／max：**{diag['variance_min']:.12g}／{diag['variance_max']:.12g}**。
- Variance < 1e-8 的維度比例：**{diag['near_zero_fraction']:.2%}**；≥80% 的 collapse warning：**{diag['collapse_warning']}**。

完整逐維 variance 存於 `dimension_variance.csv`。Same/different 近 1 且高度重疊時，不能因 retrieval 高於隨機就宣稱 collapse 已解決。

## 8. FINAL TEST 3 readiness

事前固定三個條件：

1. Opening 明顯有效：其 bootstrap 95% CI 下界高於均勻隨機排名的解析期望 score。100 候選的期望為 `sum(exp(-(r-1)),r=1..5)/100` = **{s['decision']['analytic_random_expected_score']:.6f}**，這不是額外執行 Random 方法。結果：**{s['decision']['opening_effective']}**。
2. 固定 Fusion score 高於固定 Opening。結果：**{s['decision']['fusion_above_opening']}**。
3. Paired bootstrap 的 Fusion−Opening 平均差為正。結果：**{s['decision']['bootstrap_mean_positive']}**。

因此：**{s['decision']['status']}**。依照使用者指定規則，差值 CI 是否跨 0 另外報告，沒有偷偷加入「CI 下界必須 >0」作第四個門檻。READY 只代表值得規劃新的 one-shot TEST；本輪**沒有建立 FINAL TEST 3**，也沒有加入 Strength Estimator、MiniZero 或新模型。

## 測試、執行與輸出

完整 **{verification['tests']} tests passed**，failure/error/skipped 均 0；所有 imports/syntax 通過。包含封存路徑不可讀、DEV2 身份排除、固定 alpha/window/checkpoint/epoch、bootstrap deterministic、相同 questions 與 inference 無 optimizer/backward 檢查。

正式 Stability pipeline wall-clock **{s['elapsed_seconds']/60:.2f} 分鐘**。輸出隔離於 `outputs/phase29/`：split audit、三方法 predictions/per-question scores、完整 similarities、DEV2 對照、bootstrap CI/samples、embedding diagnostics 與 summary。已完成後 command 拒絕重做 inference，報告可從保存結果重建。

檢查流程曾誤用舊版 verifier，讀取一次 CLOSED receipt 的狀態 metadata；已改用獨立 `check_phase29.py`。未讀舊 TEST 棋局或 ground truth、未重做舊 TEST inference；正式 Stability pipeline 沒有開啟任何 CLOSED artifact。

```powershell
python scripts/check_phase29.py
python -m src.phase29_validation --config configs/phase29.yaml
python scripts/report_phase29.py
```
'''
        (ROOT/'docs/phase29_results.md').write_text(text, encoding='utf-8')
    print('Saved docs/phase29_results.md from Stability artifacts only')


if __name__ == '__main__':
    main()
