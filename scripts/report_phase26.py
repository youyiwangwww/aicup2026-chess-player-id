"""Export a compact diagnosis report from saved DEV results, never any final TEST."""
import json
from pathlib import Path
import sys

import pandas as pd


def table(frame, columns=None):
    """Format saved numeric results as Markdown without an extra dependency."""
    if columns:
        frame = frame[columns]
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        values = [f'{value:.6f}' if isinstance(value, float) else str(value) for value in row]
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def main():
    """Read Phase 2.6 summaries only and verify preserved Round 1 hashes."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.phase26_common import dev_only
    from src.phase26_prepare import preserve_round1
    from src.metric_utils import file_digest, write_json
    from src.utils import load_config
    directory = root / 'outputs/phase26'
    with dev_only():
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        selected = json.loads((directory / 'selection.json').read_text(encoding='utf-8'))
        for key in ['selected_experiment', 'best_epoch', 'dev_val_score']:
            if selected[key] != summary['selection'][key]:
                raise ValueError('Frozen selection disagrees with completed DEV experiment')
        if not selected.get('frozen'):
            raise ValueError('Freeze the selected checkpoint before exporting the final report')
        if summary['final_test2_evaluations'] != 0:
            raise ValueError('FINAL TEST 2 must remain unevaluated')
        summary['selection'] = selected
        write_json(summary, directory / 'summary.json')
        results = pd.read_csv(directory / 'dev_results.csv')
        opening = pd.read_csv(directory / 'opening_signal.csv')
        cross = pd.read_csv(directory / 'cross_color_results.csv')
        similarity = pd.read_csv(directory / 'embedding_similarity.csv')
        diagnostics = json.loads((directory / 'embedding_diagnostics.json').read_text(encoding='utf-8'))
        epochs = pd.read_csv(directory / 'experiments/C20-Random/training_log.csv')
        hard_epochs = pd.read_csv(directory / 'experiments/H20-Hard/training_log.csv')
        curve_columns = ['epoch', 'train_loss', 'val_top1', 'val_top3', 'val_top5', 'val_score', 'is_best']
        score_columns = ['experiment', 'best_epoch', 'val_top1', 'val_top3', 'val_top5', 'val_score']
        early_loss, late_loss = float(epochs.train_loss.head(3).mean()), float(epochs.train_loss.tail(3).mean())
        early_score, late_score = float(epochs.val_score.head(3).mean()), float(epochs.val_score.tail(3).mean())
        curve_best = epochs.loc[epochs.val_score.idxmax()]
        curve = {'first3_mean_loss': early_loss, 'last3_mean_loss': late_loss,
                 'first3_mean_val_score': early_score, 'last3_mean_val_score': late_score,
                 'best_epoch': int(curve_best.epoch), 'best_val_score': float(curve_best.val_score)}
        if late_loss < early_loss and late_score < early_score:
            curve['interpretation'] = 'Loss 下降，但 DEV score 未同步改善；可能過擬合或優化目標與檢索不一致，單次曲線無法區分原因。'
        elif late_loss >= early_loss:
            curve['interpretation'] = 'Loss 沒有穩定下降，優先檢查優化、sampling 與 representation；不足以直接宣稱過擬合。'
        else:
            curve['interpretation'] = 'Loss 與 DEV score 的平均趨勢改善；仍應依最高 DEV score 而非最低 loss 選 checkpoint。'
        (directory / 'epochs_analysis.json').write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding='utf-8')
        tests = json.loads((root / '.test-tmp/phase26_verification.json').read_text(encoding='utf-8'))
        audit = json.loads((directory / 'splits/split_audit.json').read_text(encoding='utf-8'))
        if file_digest(root / 'data/training/train_A.csv') != audit['source_csv_sha256']:
            raise ValueError('Official training CSV changed since split creation')
        cross_audit = json.loads((directory / 'cross_color_audit.json').read_text(encoding='utf-8'))
        sections = ['# Phase 2.6：Triplet Baseline Diagnosis（DEV ONLY）',
                    'Round 1 TEST 為 CLOSED TEST；FINAL TEST 2 為 LOCKED，評估次數 0。所有選擇只依 DEV VALIDATION。',
                    '初次 split 建立時曾重開 FINAL TEST 2 truth 檔案計算 SHA-256；未解析身份或評分。已改為在記憶體計算 hash，所有 tuning 步驟均未開啟該 truth 檔。',
                    f'驗證：{tests["tests"]} tests 通過，failures {tests["failures"]}、errors {tests["errors"]}；syntax passed {tests["syntax_passed"]}。',
                    '## Split audit',
                    f'DEV TRAIN / DEV VAL / FINAL TEST 2 players = {audit["train_player_count"]} / {audit["val_player_count"]} / {audit["test_player_count"]}。',
                    'TRAIN 4,000 games；DEV VAL candidate/query 1,000/500；FINAL TEST 2 candidate/query 2,000/500。23 項交集為 0，排除 10 位 Round 1 CLOSED TEST 玩家。',
                    '## Opening windows', table(opening),
                    '## Training data size', table(results[results.experiment.isin(['A1', 'A2', 'A3'])], ['experiment', 'train_players', 'games_per_player'] + score_columns[1:]),
                    '## Positions', table(results[results.experiment.isin(['B4', 'B8', 'B16'])], ['experiment', 'positions_per_game'] + score_columns[1:]),
                    '## Random epochs curve', table(epochs, curve_columns), curve['interpretation'],
                    '## Hard epochs curve', table(hard_epochs, curve_columns),
                    '## Random vs batch hard', table(results[results.experiment.isin(['C20-Random', 'H20-Hard'])], score_columns),
                    '## Cross-color', table(cross), f'Cross-color eligibility/audit: `{json.dumps(cross_audit, ensure_ascii=False)}`',
                    '## Embedding diagnostics', table(similarity),
                    f'Mean separation = {diagnostics["separation"]:.6f}；IQR overlap = {diagnostics["iqr_overlap"]}；standardized separation = {diagnostics["standardized_separation"]:.6f}。',
                    diagnostics['interpretation'],
                    '## DEV-selected configuration',
                    f'選定 {selected["selected_experiment"]}，best epoch {selected["best_epoch"]}；DEV Score {selected["dev_val_score"]:.6f}。',
                    f'同 DEV Phase 2.5 reference Score {selected["reference_dev_val_score"]:.6f}；絕對改善 {selected["score_improvement_same_dev"]:.6f}。',
                    '設定存於 `configs/phase26_selected.yaml`。不能與 Round 1 CLOSED TEST 分數直接比較。',
                    '## 限制與下一步',
                    'A1→A2 同時改玩家數與每人棋譜數；A2→A3 才能較直接檢查身份數。每 epoch 固定 512 triplets，資料量增大會降低每位玩家的平均曝光。',
                    '多次 DEV 模型選擇可能對 DEV 過擬合；跨棋局 similarity pairs 並非獨立樣本。Cross-color 只用符合黑白棋譜數門檻的 DEV 玩家，不可與不同候選數的原 DEV 分數直接比較。',
                    '固定 seed、相同 sampler 的不同 K 會產生不同取樣位置，位置集合並非巢狀；positions ablation 仍可能有取樣波動。',
                    '不建議立即進 Phase 3：先固定設定完成獨立 FINAL TEST 2 的一次評估，並確認重現性、資料曝光與 representation。FINAL TEST 2 評估需另行使用者授權。',
                    f'本輪 runner elapsed = {summary["elapsed_seconds"]:.2f} seconds。']
        hard_score = float(results.loc[results.experiment == 'H20-Hard', 'val_score'].iloc[0])
        random_score = float(results.loc[results.experiment == 'C20-Random', 'val_score'].iloc[0])
        opening_best = opening.loc[opening.val_score.idxmax()]
        sections.extend([
            '## 結果解讀',
            f'Opening 最佳為 {opening_best.experiment}（{opening_best.val_score:.6f}），短開局優於全局 heatmap，支持優先研究開局取樣；不能據此證明身份訊號只來自開局。',
            f'Hard negative 比其他設定相同的 Random negative 改善 {hard_score-random_score:.6f}。它仍低於最佳 Opening；高於 reference 的幅度混合了容量、positions、epochs 與 mining 的變化，不能全歸因於 mining。',
            '資料量 ablation 未顯示較大資料帶來穩定提升，但固定 512 triplets/epoch 導致較大資料的每玩家曝光較少，因此無法排除資料不足。',
            '跨顏色辨識偏弱。兩方向使用同一組 34 位玩家，仍缺少這 34 位玩家的同色與混色配對控制，不能只靠與原 50 位 DEV 的分數差斷言顏色是唯一原因。',
            'Embedding cosine 高度集中且同／不同玩家分布重疊，尚未學出清楚的玩家可分性。暫不建議進 Phase 3。',
            '## 下一輪建議（未實作）',
            '1. 固定 hard negative 與每玩家訓練曝光，再比較 30／100／200 training identities。',
            '2. 固定 nested sampled positions，比較前 10／20 手與全局 positions，維持其他設定不變。',
            '3. 在相同 34 位玩家建立 B→B、W→W、混色及跨色控制組，隔離顏色與候選人組成的效果。',
            'FINAL TEST 2 是否執行一次，等待使用者明確確認；本輪不會自動執行。',
            'Runtime warnings and the separate CUDA limitation: [phase26_runtime.md](phase26_runtime.md). All reported experiments use CPU.'
        ])
        (root / 'docs/phase26_results.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    preserve_round1(load_config(root / 'configs/phase26.yaml'))
    print('Saved docs/phase26_results.md; all Round 1 hashes unchanged; FINAL TEST 2 remains locked.')


if __name__ == '__main__':
    main()
