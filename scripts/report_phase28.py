"""Produce a meeting-friendly report from completed DEV2 artifacts; no inference."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase28_common import dev2_only


def table(frame, columns):
    """Render dependency-free Markdown tables with explicit numeric precision."""
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            value = row[key]
            if isinstance(value, float):
                if key.startswith('variance') or key in ['separation', 'std']:
                    value = f'{value:.6g}'
                elif key in ['mean', 'median', 'p25', 'p75']:
                    value = f'{value:.12f}'
                else:
                    value = f'{value:.6f}'
            values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    """Explain only completed, genuine DEV2 results and preserve historical reports."""
    directory = ROOT/'outputs/phase28'
    with dev2_only():
        summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
        if summary['status'] != 'COMPLETE DEV2 ONLY':
            raise ValueError('Report requires a completed DEV2 experiment')
        frames = {name: pd.read_csv(directory/(name+'.csv')) for name in ['opening_results', 'exposure_results',
            'color_control_results', 'cross_color_triplet_results', 'embedding_diagnostics', 'fusion_results', 'error_overlap', 'feature_coverage']}
        op, exposure, color, cross, diag, fusion, error, coverage = [frames[name] for name in frames]
        metrics = ['top1', 'top3', 'top5', 'competition_score']
        total = float(op.loc[op.method == 'Opening-total-10', 'competition_score'].iloc[0])
        player5 = float(op.loc[op.method == 'Opening-player-5', 'competition_score'].iloc[0])
        color5 = float(op.loc[op.method == 'Color-aware-player-5', 'competition_score'].iloc[0])
        color10 = float(op.loc[op.method == 'Color-aware-player-10', 'competition_score'].iloc[0])
        player10 = float(op.loc[op.method == 'Opening-player-10', 'competition_score'].iloc[0])
        hard_score, cross_score = cross.competition_score.tolist()
        cohort = json.loads((directory/'color_controls/audit.json').read_text(encoding='utf-8'))
        training_cohort = json.loads((directory/'cross_color_cohort.json').read_text(encoding='utf-8'))
        tests = json.loads((directory/'verification.json').read_text(encoding='utf-8'))
        ob, tb, fb, selected = [summary[k] for k in ['opening_best', 'triplet_best', 'best_fusion', 'selected']]
        both_wrong = int(error.loc[error.category == 'both_wrong', 'count'].iloc[0])
        triplet_only = int(error.loc[error.category == 'triplet_only_correct', 'count'].iloc[0])
        opening_wrong = 100-int(round(ob['top1']*100))
        cosine_rows = []
        for _, row in diag.iterrows():
            for kind in ['same', 'different']:
                cosine_rows.append({'experiment': row.experiment, 'pair_type': kind,
                    **{stat: row[kind+'_'+stat] for stat in ['mean', 'median', 'std', 'p25', 'p75']}})
        color_scores = color.pivot(index='condition', columns='method', values='competition_score').reset_index()
        order = ['B_to_B', 'W_to_W', 'B_to_W', 'W_to_B', 'Mixed_to_Mixed']
        color_scores = color_scores.set_index('condition').loc[order].reset_index()
        gaps = []
        for method in color.method.unique():
            rows = color[color.method == method].set_index('condition')
            same = (rows.loc['B_to_B', 'competition_score'] + rows.loc['W_to_W', 'competition_score'])/2
            different = (rows.loc['B_to_W', 'competition_score'] + rows.loc['W_to_B', 'competition_score'])/2
            gaps.append({'method': method, 'same_color_mean': same, 'cross_color_mean': different, 'gap': same-different})
        gap_frame = pd.DataFrame(gaps)
        xgap = float(gap_frame.loc[gap_frame.method == 'Triplet-Hard-CrossColor', 'gap'].iloc[0])
        hgap = float(gap_frame.loc[gap_frame.method == cross.experiment.iloc[0], 'gap'].iloc[0])
        exposure30 = float(exposure.loc[exposure.train_players == 30, 'competition_score'].iloc[0])
        exposure200 = float(exposure.loc[exposure.train_players == 200, 'competition_score'].iloc[0])
        opening_gap = float(gap_frame.iloc[0]['gap'])
        collapse_count = int(diag.collapse_warning.sum())
        color_hard = color[color.method == cross.experiment.iloc[0]].set_index('condition')
        color_cross = color[color.method == 'Triplet-Hard-CrossColor'].set_index('condition')
        robustness_changes = {name: float(color_cross.loc[name, 'competition_score']-color_hard.loc[name, 'competition_score'])
                              for name in ['B_to_W', 'W_to_B', 'Mixed_to_Mixed']}
        timings = []
        for name in diag.experiment:
            history = pd.read_csv(directory/'experiments'/name/'training_log.csv')
            timings.append({'experiment': name, 'epoch_median_seconds': float(history.elapsed_seconds.median()),
                'epoch_min_seconds': float(history.elapsed_seconds.min()), 'epoch_max_seconds': float(history.elapsed_seconds.max()),
                'epochs_total_seconds': float(history.elapsed_seconds.sum())})
        contents = f'''# Phase 2.8：Opening-aware / Color-aware DEV2 結果

本輪最佳方法：**{selected['method']}**，DEV2 competition score **{selected['competition_score']:.6f}**。只使用 DEV2 選方法、epoch 與 alpha；Round 1 TEST、FINAL TEST 2 保持 CLOSED，未讀取其 ground truth、未執行其 inference，也未建立 FINAL TEST 3。這是 validation 結果，不能宣稱新的 held-out TEST 效果。

## 會議重點

| 問題 | 本輪答案 |
| --- | --- |
| Total vs player window | total-10 與 player-5 相同；player-10／20 較弱 |
| Color-aware 是否改善 | player-5 分色 score 提高 {color5-player5:.6f} |
| 同色 vs 跨色 | 固定 {cohort['players']} 人的 Opening 同色−跨色 score gap 為 {opening_gap:.6f} |
| Exposure 是否改善 | 200 人相較 30 人差 {exposure200-exposure30:+.6f}；尚無同 DEV fixed-512 對照 |
| CrossColor robustness | 主 VAL score 差 {cross_score-hard_score:+.6f}；配對色控 gap 從 {hgap:.6f} 到 {xgap:.6f} |
| Collapse 是否解決 | 尚未解決；{len(diag)} 個實驗中 {collapse_count} 個觸發近零維度 warning，未觸發的 30-player cosine 也接近 1 |
| Fusion 是否超過 Opening | 最佳 alpha={fb['alpha']:.1f}，score 差 {fb['competition_score']-ob['competition_score']:+.6f} |
| Error 是否互補 | Opening 錯的 {opening_wrong} 題中，Triplet 單獨補對 {triplet_only} 題 |

## 資料與可重現性

來源：`data/training/train_A.csv`；SHA256 `{summary['split_audit']['source_sha256']}`。Seed 42。排除 Round 1 TEST 10 位、Phase 2.6 VAL 50 位與 FINAL TEST 2 的 50 位，共 110 位。最後一組身份只由封存 predictions 取得；receipt 只驗證封存狀態與玩家數，不拿舊 TEST metrics 做選擇。

| Partition | Players | Games/player | Total games |
| --- | --- | --- | --- |
| DEV2 TRAIN | 200 | 20 | 4000 |
| DEV2 VAL candidate | 100 | 30 | 3000 |
| DEV2 VAL query | 100 | 10 | 1000 |

TRAIN/VAL identity overlap = 0；與排除集合 identity overlap = 0；所有 TRAIN/candidate/query pair 的 game ID 與 SGF overlap = 0。原始 CSV 的 game ID 與 exact SGF 字串全域唯一，排除整位玩家也排除了其所有來源棋局。Queries 無 player_id、rank。完整 IDs、CSV SHA256 與 audit 保存在 `outputs/phase28/splits/`。

{table(coverage, ['partition', 'total_games', 'successful_games', 'failed_games', 'success_rate', 'mean_positions', 'median_positions', 'min_positions', 'max_positions'])}

## 1. Opening 窗口與黑白分開比對

total-N 計算整盤前 N 個 B/W move nodes；player-N 只計算目標玩家的前 N 次輪到落子。兩者都讓 pass 消耗窗口額度，但 pass 不加入 heatmap；setup 不消耗額度；只讀主變化。每盤 heatmap 等權平均，每位玩家/每題比較完整候選集合。沒有合法落點的棋譜記錄錯誤，不讓單一 SGF 停止流程。

Color-aware 將 candidate B/W 分開平均，query 的 B 僅比 B、W 僅比 W，再依該題原始 query B/W 棋局數加權；缺少某色 fingerprint 時該色貢獻為 0，保留全部候選人。未加入 rank、姓名等身份特徵。

{table(op, ['method']+metrics)}

total-10 與 player-5 差 **{player5-total:+.6f}**；一般輪流下棋時它們包含相同目標手數，本次結果相同。player-10／20 的結果見表，窗口變長不保證風格更易辨識。Color-aware-player-5 比非分色 player-5 改善 **{color5-player5:+.6f}**；player-10 分色改善 **{color10-player10:+.6f}**。

## 2. 相同 cohort 的同色／跨色控制

主 VAL 100 位中，只有 **{cohort['players']} 位**同時有至少 40 盤 B 與 40 盤 W；因此明確縮小到這一固定 cohort，五組皆使用相同 IDs、每人 candidate 30/query 10。Mixed 固定 candidate 15B+15W、query 5B+5W。每組內 candidate/query 無 game/SGF overlap；不同控制條件可重用棋局，這是配對控制設計。使用主 VAL 事先選定的最佳**非分色 Opening**，避免跨色組本來就沒有同色 candidate bank 而得到機械式 0 分。

{table(color_scores, list(color_scores.columns))}

{table(gap_frame, list(gap_frame.columns))}

gap = 同色 B→B/W→W 平均 score − 跨色 B→W/W→B 平均 score。候選人數與身份組成已固定，因此這些差異支持棋色／棋局分布有影響；棋局內容仍有差異，不能直接宣稱純棋色的因果效果。這個小 cohort 的 score 也不能直接跟主 VAL 100 人混合比較。

## 3. Exposure-controlled Triplet

固定 T=20 **anchor triplets/player/epoch**，每個 epoch 驗證實際每人 exposure 的 min=max=20；30/100/200 人為 nested TRAIN 子集，20 games/player、16 positions/game、64 channels、8 blocks、128 embedding、batch-hard negative、最多 20 epochs，其餘參數一致。正樣本從同玩家不同棋局取位置，negative 在 batch pool 中選最近的其他 TRAIN 玩家；採樣與 mining 都不使用 VAL identity。

{table(exposure, ['experiment', 'train_players', 'samples_per_epoch', 'best_epoch']+metrics)}

只依 DEV2 competition score 取最高 epoch，平手取較早 epoch。這修正了每人的 anchor exposure，但總 optimizer steps 隨人數上升，所以不能將差異完全歸因於玩家多樣性。沒有同一 DEV2、相同 GPU 環境下的 fixed-512 對照；不拿 Phase 2.6 不同 VAL 的 score 推論 exposure 改善。

## 4. Cross-color positive 的配對對照

父實驗 `{training_cohort['parent_experiment']}` 共 {training_cohort['parent_players']} 位；至少各 2 盤 B/W 的實際 eligible TRAIN 玩家 **{training_cohort['actual_players']} 位**。Hard 與 CrossColor 使用完全相同 eligible IDs、棋局、每人 T=20 與其他參數；若父實驗有不合格身份，另訓練 matched Hard 對照。CrossColor 每位玩家每 epoch 固定 10 個 B-anchor/W-positive、10 個 W-anchor/B-positive，絕不退回同色 positive。

{table(cross, ['experiment', 'train_players', 'samples_per_epoch', 'best_epoch']+metrics)}

主 VAL CrossColor − matched Hard score = **{cross_score-hard_score:+.6f}**。固定色控 cohort 的同色−跨色 gap：Hard **{hgap:.6f}**、CrossColor **{xgap:.6f}**；同時看跨色絕對 score 與 gap，不能只因 gap 變小就說模型更好（也可能同色退步）。

CrossColor 的配對 score 變化：B→W **{robustness_changes['B_to_W']:+.6f}**、W→B **{robustness_changes['W_to_B']:+.6f}**、Mixed→Mixed **{robustness_changes['Mixed_to_Mixed']:+.6f}**。本次改善不一致：W→B/Mixed 較好，B→W 較差，平均同色−跨色 gap 也沒有縮小，因此尚無穩定降低棋色依賴的證據。

## 5. Embedding separation 與 collapse

診斷使用選定 epoch 的 DEV2 candidate/query **game embeddings**；先平均 positions，再 L2 normalize。Same/different 以 candidate-game/query-game 配對計算 cosine，std 使用 population std。Dimension-wise variance 在 3000 個單位 candidate-game embeddings 上計算；全部 128 維另存於各 experiment 的 `dimension_variance.csv`。

{table(pd.DataFrame(cosine_rows), ['experiment', 'pair_type', 'mean', 'median', 'std', 'p25', 'p75'])}

{table(diag, ['experiment', 'separation', 'variance_mean', 'variance_median', 'variance_min', 'variance_max', 'near_zero_fraction', 'collapse_warning'])}

工程診斷門檻：variance < 1e-8 為近零，近零維度比例 ≥ 80% 發出 collapse warning。門檻已固定於 config，不作模型選擇依據；warning 是數值退化提示，不等同可證明所有玩家完全不可分。Same/different 很接近且 variance 小時，不能只看 Top-1 便宣稱學到穩定風格。

## 6. Fusion 與 error overlap

使用最佳 Opening `{ob['method']}` 與最佳 Triplet `{tb['experiment']}`，對**每題全部 100 個 candidate scores**做 z-score，然後 `alpha*Opening+(1-alpha)*Triplet`。支援 min-max，但本輪固定 z-score，不另外搜尋 normalization。alpha=0/1 排名已驗證完全等於純 Triplet/Opening；Top-5 沒有重複 IDs。

{table(fusion, ['alpha']+metrics)}

最佳 alpha **{fb['alpha']:.1f}**，score **{fb['competition_score']:.6f}**；比純最佳 Opening 差 **{fb['competition_score']-ob['competition_score']:+.6f}**。這是同一個 DEV2 上的模型與 alpha 搜尋，需新 held-out evaluation 才能確認泛化，不能把 validation 小幅提高視為已證實。

{table(error, ['category', 'count', 'proportion'])}

Opening 的 {opening_wrong} 個 Top-1 錯誤中，Triplet 單獨補對 {triplet_only} 個；兩者都錯 {both_wrong} 個。因此是否互補應以這些逐題錯誤與融合實際分數一起判讀，互補不保證加權分數一定改善。

## 選定設定、測試與下一步

`configs/phase28_selected.yaml` 已 frozen，依據只有 DEV2 competition score。選定 **{selected['method']}**，score **{selected['competition_score']:.6f}**。若融合端點與純 Opening 平手，保留純 Opening；其他平手按預先固定順序／較小 alpha。記錄來源、split、Triplet checkpoint SHA256 與各窗口定義；純 Opening 選中時不需要載入 Triplet。

完整 tests **{tests['tests']} passed**，failure/error/skipped 均 0；所有 src Python imports 與 syntax check 通過。GPU `{summary['environment']['device']}`，Torch `{summary['environment']['torch']}`，deterministic=True。既有 encoder 結構與 675648 參數不變；CUDA deterministic adaptive pooling backward 的限制以數學等價的 floor/ceil window mean 實作處理，CPU/CUDA forward 與 gradient 已驗證，未修改歷史模型程式或封存 checkpoint。

正式 pipeline wall-clock 時間 **{summary['elapsed_seconds']/60:.2f} 分鐘**，包含 preprocessing、訓練、DEV 評估與觀察到的時間波動，不能等同純 GPU compute time。完整逐 epoch logs 與 provenance 保存在 `outputs/phase28/experiments/`。測試時的 mock 只位於 `.test-tmp/`，不是正式 DEV2 結果。完整原始 metrics 在本報告同名 CSV。

{table(pd.DataFrame(timings), list(timings[0]))}

時間警示：部分 epochs 明顯較慢，最長 **{max(t['epoch_max_seconds'] for t in timings):.1f} 秒**；log 未顯示對應訓練例外，原因未經確認，因此只記錄 wall-clock 波動，沒有因此重跑、改 seed、換 split 或改參數。

下一步先確認 DEV2 選擇是否穩定，再決定一次性 FINAL TEST 3；本輪沒有建立其 split 或執行其 inference。建議：

1. 在獨立 DEV identities／預先固定多個 seeds 上確認 color-aware Opening 與選定 alpha 的穩定性，避免同一 VAL 的多次選擇偏差。
2. 另做相同 DEV、matched optimizer steps 的 fixed-512 vs T=20 對照，釐清總更新數與每人 exposure 的作用。
3. 針對 collapse、同色／跨色差距及互補錯誤進行資料與 embedding 分析，再考慮後續模型研究；尚未加入 Strength Estimator、MiniZero 或其他 Phase 3 元件。

重跑與報告指令（僅 DEV2；已完成的相同實驗會驗證 provenance 後重用 checkpoint，不重訓）：

```powershell
python scripts/check_phase28.py
python scripts/check_phase28_cuda.py
python -m src.run_phase28 --config configs/phase28.yaml
python scripts/report_phase28.py
```
'''
        (ROOT/'docs/phase28_results.md').write_text(contents, encoding='utf-8')
    print('Saved docs/phase28_results.md (DEV2 artifacts only)')


if __name__ == '__main__':
    main()
