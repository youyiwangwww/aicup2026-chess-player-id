"""Freeze the DEV comparison reference, then document saved one-shot TEST results."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate
from src.metric_utils import file_digest, write_json
from src.phase26_common import dev_only
from src.random_baseline import random_predictions
from src.utils import read_csv, write_csv


def table(rows):
    """Render metric dictionaries as a concise Markdown comparison."""
    lines = ['| Method | Top-1 | Top-3 | Top-5 | Score |', '|---|---:|---:|---:|---:|']
    for method, values in rows.items():
        lines.append(f'| {method} | {values["top_1_accuracy"]:.4f} | {values["top_3_accuracy"]:.4f} | {values["top_5_accuracy"]:.4f} | {values["competition_score"]:.6f} |')
    return '\n'.join(lines)


def prepare_dev_reference():
    """Add the fixed random DEV reference; reuse all saved Opening/Triplet DEV scores."""
    from src.final_test2_lock import require_unconsumed
    require_unconsumed()
    path = ROOT / 'outputs/phase26/phase27_dev_reference.json'
    with dev_only():
        sources = {name: file_digest(ROOT / f'outputs/phase26/splits/dev_val_{name}.csv')
                   for name in ['candidates', 'queries', 'ground_truth']}
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            if saved['source_sha256'] != sources or saved['seed'] != 42:
                raise ValueError('Existing DEV comparison provenance mismatch')
            return saved
        c = read_csv('outputs/phase26/splits/dev_val_candidates.csv', ['player_id', 'game_id'])
        q = read_csv('outputs/phase26/splits/dev_val_queries.csv', ['question_id', 'game_id'])
        truth = read_csv('outputs/phase26/splits/dev_val_ground_truth.csv', ['question_id', 'player_id'])
        predictions = random_predictions(c.player_id, q.question_id, seed=42, top_k=5)
        random, _ = evaluate(predictions, truth)
        saved_results = pd.read_csv(ROOT / 'outputs/phase26/dev_results.csv')
        values = {'random': random}
        for name, experiment in [('opening10', 'Opening-10'), ('triplet_hard', 'H20-Hard')]:
            row = saved_results[saved_results.experiment == experiment].iloc[0]
            values[name] = {key: float(row[column]) for key, column in [
                ('top_1_accuracy', 'val_top1'), ('top_3_accuracy', 'val_top3'),
                ('top_5_accuracy', 'val_top5'), ('competition_score', 'val_score')]}
        reference = {'seed': 42, 'source_sha256': sources, 'metrics': values,
                     'note': 'Random DEV reference computed before FINAL TEST 2; model DEV scores reused, no selection changes.'}
        write_json(reference, path)
        write_csv(predictions, 'outputs/phase26/phase27_dev_random_predictions.csv')
        print('Saved fixed DEV comparison reference; no FINAL TEST inputs opened.')
        return reference


def report():
    """Update documents using saved metrics only; never reopen closed TEST inputs."""
    directory = ROOT / 'outputs/phase26/final_test2'
    receipt = json.loads((directory / 'final_test2_receipt.json').read_text(encoding='utf-8'))
    if receipt['evaluation_count'] != 1 or receipt['status'] != 'CLOSED TEST':
        raise ValueError('Need a successful closed one-shot receipt')
    reference = json.loads((ROOT / 'outputs/phase26/phase27_dev_reference.json').read_text(encoding='utf-8'))
    dev, test = reference['metrics'], receipt['metrics']
    delta = test['triplet_hard']['competition_score'] - dev['triplet_hard']['competition_score']
    dev_gap = dev['opening10']['competition_score'] - dev['triplet_hard']['competition_score']
    test_gap = test['opening10']['competition_score'] - test['triplet_hard']['competition_score']
    rank = lambda values: sorted(values, key=lambda method: values[method]['competition_score'], reverse=True)
    analyses = {'triplet_test_minus_dev': delta, 'dev_opening_triplet_gap': dev_gap,
                'test_opening_triplet_gap': test_gap, 'ranking_consistent': rank(dev) == rank(test),
                'dev_ranking': rank(dev), 'test_ranking': rank(test)}
    conclusions = (f'Triplet 比 Random 高 {test["triplet_hard"]["competition_score"]-test["random"]["competition_score"]:.6f}。'
                   f'最佳 TEST 方法為 {rank(test)[0]}。'
                   f'Opening − Triplet 差距相較 DEV {"縮小" if test_gap < dev_gap else "擴大"} {abs(test_gap-dev_gap):.6f}。')
    write_json(analyses, directory / 'generalization_analysis.json')
    tests = json.loads((ROOT / '.test-tmp/phase26_verification.json').read_text(encoding='utf-8'))
    section = '\n\n'.join([
        '## FINAL TEST 2 One-Shot Result',
        'FINAL TEST 2 已永久 CLOSED TEST，evaluation_count = 1。Frozen config、checkpoint、hyperparameters、seed 與 split 全部保持原值；本輪沒有 training、backprop、optimizer step 或再次選模。',
        f'Selected H20-Hard，best epoch 17；checkpoint SHA256 `{receipt["checkpoint_sha256"]}`。Config SHA256 `{receipt["config_sha256"]}`。',
        '### Feature coverage',
        '\n'.join(f'- {row["partition"]}: total {row["total_games"]}, successful {row["successful_games"]}, failed {row["failed_games"]}, success rate {row["success_rate"]:.2%}' for row in receipt['feature_coverage']),
        '### Final TEST metrics', table(test), '### DEV reference', table(dev),
        f'Triplet TEST − DEV = {delta:+.6f}。Opening − Triplet gap：DEV {dev_gap:.6f}；TEST {test_gap:.6f}。Ranking consistent = {analyses["ranking_consistent"]}。',
        conclusions,
        'DEV candidate 每人 20 盤，TEST 每人 40 盤，且玩家不同；score 差異同時包含未見玩家與候選 fingerprint 品質的影響。本報告未做顯著性檢定；單 seed、50 questions 仍不足以估計跨 split 的穩定性。',
        'Phase 2.6 原 Triplet 的改進以同一 DEV reference 衡量；未在 FINAL TEST 2 重跑未選定的 Random-Negative checkpoint，因此不能把 TEST 差異直接歸因於 hard negative。',
        '只解讀本次固定設定結果，不用 CLOSED TEST 做後續調參。暫不進 Phase 3：embedding 可分性與跨顏色 robustness 仍需在新的 DEV 設計中驗證。',
        f'CPU tests: {tests["tests"]} passed, failures {tests["failures"]}, errors {tests["errors"]}; syntax {tests["syntax_passed"]}。One-shot elapsed {receipt["elapsed_seconds"]:.2f} seconds。',
        '一次性入口：`python -m src.phase27_final_test2`。再次呼叫（包括 check-only）將拒絕；共用 CSV readers 也拒絕重開已消耗的 FINAL TEST 2 split。Frozen YAML 中的 LOCKED 描述是評估前快照；最新狀態以 final_test2_receipt.json 為準。'
    ])
    path = ROOT / 'docs/phase26_results.md'
    text = path.read_text(encoding='utf-8')
    marker = '## FINAL TEST 2 One-Shot Result'
    text = text.split(marker)[0].rstrip()
    # Preserve DEV tables; clarify that their zero-count statement describes the earlier phase.
    text = text.replace('Round 1 TEST 為 CLOSED TEST；FINAL TEST 2 為 LOCKED，評估次數 0。所有選擇只依 DEV VALIDATION。',
                        'Phase 2.6 完成時：Round 1 TEST 為 CLOSED TEST；FINAL TEST 2 為 LOCKED，評估次數 0。所有選擇只依 DEV VALIDATION。Phase 2.7 最終結果與 CLOSED 狀態見文末。')
    text = text.replace('FINAL TEST 2 是否執行一次，等待使用者明確確認；本輪不會自動執行。',
                        'Phase 2.6 結束時等待使用者確認；其後已獲授權執行 Phase 2.7 一次性評估，結果見文末。')
    path.write_text(text + '\n\n' + section + '\n', encoding='utf-8')
    professor = '\n\n'.join([
        '# Go Player Identification：教授會議更新',
        '## Round 1（CLOSED TEST）',
        '正式資料 100,000 games／1,582 players；TRAIN／VAL／TEST players = 30／10／10，identity 不重疊。Random Score 0.019028，Opening 0.715343，Triplet 0.230301。此 TEST 已關閉，不再調參。',
        '## Phase 2.6 diagnosis（DEV ONLY）',
        '新 split：DEV TRAIN／DEV VAL／FINAL TEST 2 players = 200／50／50，三組玩家完全不同；game_id、SGF 亦無交集，排除 Round 1 CLOSED TEST 玩家。',
        'Opening 前 10 手最佳 DEV Score 0.663844，全局 heatmap 0.143219，顯示開局偏好值得優先研究。',
        '相同設定下，Hard Negative DEV Score 0.376701 高於 Random Negative 0.186007，最佳 epoch 17；同 DEV Phase 2.5 reference 0.133546。增加資料未改善，但固定 512 triplets/epoch 有曝光次數混淆。',
        '選定模型實際使用 master TRAIN 中的 30 players × 10 games、16 positions/game、64 channels／8 blocks／128 dimensions；並非使用全部 200 位 TRAIN 玩家。',
        'Embedding 有 collapse 警訊：same／different cosine 都約 0.999992，separation 僅 2.05e-7，分布高度重疊。Cross-color 的 34-player Score 僅 0.044–0.098，需要配對控制。',
        '## FINAL TEST 2（ONE-SHOT，已 CLOSED）', table(test),
        f'固定 H20-Hard epoch 17，沒有訓練或修改 frozen artifacts。Triplet TEST−DEV = {delta:+.6f}；DEV／TEST 方法排序一致：{analyses["ranking_consistent"]}。',
        conclusions,
        '## 限制與下一步研究問題',
        '單 seed、50 queries、DEV 多組比較，且 DEV／TEST candidate games 不同。CPU tests 全通過；獨立 CUDA 訓練的 deterministic pooling 限制未放寬。',
        '暫不進 Phase 3。下一輪在新 DEV 上研究：等每玩家曝光下的 training identities 數量、固定 nested 開局／全局 positions、同一玩家集合的同色／跨色 robustness。兩個 CLOSED TEST 都不得再作調參依據。'
    ])
    (ROOT / 'docs/professor_update.md').write_text(professor + '\n', encoding='utf-8')
    print('Updated phase26_results.md and professor_update.md from saved receipt; no TEST inference repeated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--prepare-dev-reference', action='store_true')
    args = parser.parse_args()
    prepare_dev_reference() if args.prepare_dev_reference else report()
