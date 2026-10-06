"""Render FINAL TEST 3 saved results; never open closed CSVs or rerun inference."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.metric_utils import file_digest


def table(rows,columns):
    """Format saved measurements at six decimals without altering their values."""
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        lines.append('| '+' | '.join(f'{row[key]:.6f}' if isinstance(row[key],float) else str(row[key]) for key in columns)+' |')
    return '\n'.join(lines)


def main():
    """Explain one-shot isolation, complementarity and uncertainty from receipts only."""
    d=ROOT/'outputs/phase210'
    s=json.loads((d/'final_test3_receipt.json').read_text(encoding='utf-8'))
    v=json.loads((d/'verification.json').read_text(encoding='utf-8'))
    preflight=json.loads((d/'preflight.json').read_text(encoding='utf-8'))
    if s['status']!='CLOSED TEST' or s['evaluation_count']!=1 or v['failures'] or v['errors'] or not preflight['passed']:
        raise ValueError('Require closed successful one-shot and passing tests')
    if file_digest(d/'preregistered_protocol.json')!=s['protocol_sha256']:
        raise ValueError('Immutable preregistration no longer matches receipt')
    m=s['metrics']
    o,t,f=[m[name] for name in ['opening_color_player5','triplet_hard100','fusion_alpha09']]
    rows=[{'method':name,**value} for name,value in m.items()]
    boot=s['bootstrap']
    delta=f['competition_score']-o['competition_score']
    general=s['generalization']
    differences=[{'partition':part,'fusion_minus_opening':general[2][key]-general[0][key]}
                 for part,key in [('DEV2','dev2'),('Stability','stability'),('FINAL TEST 3','final_test3')]]
    crossing=boot[2]['ci_lower']<=0<=boot[2]['ci_upper']
    overlap=s['error_overlap']
    same_rank=all(general[2][k]>general[0][k]>general[1][k] for k in ['dev2','stability','final_test3'])
    ci_statement=('差值 95% CI 跨 0，不能確認 Fusion 提升穩定，且不能依結果重選 alpha。' if crossing else
                  '差值 95% CI 全為正，支持本批 question 上的正向改善；仍不包含模型選擇與跨資料集的不確定性。' if boot[2]['ci_lower']>0 else
                  '差值 95% CI 全為負，本輪顯示失敗泛化；只記錄結果，不修改 alpha。')
    final_table=table(rows,['method','top1','top3','top5','competition_score'])
    general_table=table(general,['method','dev2','stability','final_test3'])
    bootstrap_table=table(boot,['method','mean','bootstrap_mean','ci_lower','ci_upper','resamples','seed'])
    text=f'''# Phase 2.10：FINAL TEST 3 One-Shot Evaluation

## 1. 身份資格、split 與 preflight

官方資料有 100,000 games／1,582 players。排除所有歷史 TRAIN／VAL／TEST／Stability 身份，共 **{s['split_counts']['historical_unique_players']} 位不同玩家**；剩餘 **{s['split_counts']['unseen_players']} 位 unseen players**，其中 **{s['split_counts']['max_eligible_players']} 位至少有 40 games**。沒有降低玩家數。

FINAL TEST 3：100 players，每人 candidate 30／query 10，共 3,000 candidate games／1,000 query games／100 questions。所有歷史 player intersection、candidate/query game_id／exact SGF intersection 均為 0；來源 game_id／exact SGF 全域唯一。Query CSV 沒有 player_id／rank。

除了指定的六組排除名單，也排除 Round 1 TRAIN／VAL、Phase 2.6 TRAIN 及其各實驗 TRAIN。旧 TEST 身份只從先前複製的 audit 名單取得；歷史 TRAIN CSV 只讀 player_id 並 hash，沒有開啟舊 TEST CSV／ground truth，也沒有重新 inference。

在 inference 前已完成 immutable preregistration，status=PREREGISTERED／evaluation_count=0。SHA256 綁定 dataset、split audit、三份 CSV、兩份 configs、checkpoint、模型／inference code 與歷史身份來源。**13 項 preflight 全通過**。

- Dataset SHA256：`{s['dataset_sha256']}`
- Split SHA256：`{s['split_sha256']}`
- Protocol SHA256：`{s['protocol_sha256']}`
- Frozen config SHA256：`{s['config_sha256']}`
- Run config SHA256：`{s['run_config_sha256']}`
- Checkpoint SHA256：`{s['checkpoint_sha256']}`

## 2. 固定方法與 feature coverage

Opening 固定 Color-aware-player-5；Triplet 僅載入 Phase 2.8 Triplet-Hard-100 epoch 18 checkpoint（64 channels／8 blocks／128 embedding），history 8／16 positions/game。Fusion 固定逐 question、100 candidates 的 z-score：`0.9*Opening+0.1*Triplet`。Random seed=42，不參與選擇。

沒有 training、optimizer、backward、checkpoint／epoch／window／color-aware／alpha／normalization search。`model.eval()`／`torch.no_grad()`；checkpoint 和 frozen config 前後 SHA256 相同。

{table(s['feature_coverage'],['partition','total_games','successful_games','failed_games','success_rate','mean_positions','median_positions','min_positions','max_positions'])}

Opening 解析失敗：{s['opening_failed_games']}；候選缺少某顏色 fingerprint 的組數：{s['missing_candidate_colors']}，沿用已 frozen 的既有處理規則，未修改 split。

## 3. FINAL TEST 3 — ONE-SHOT RESULT

{final_table}

四方法使用同一組 100 questions／100 candidate players。Opening／Triplet／Fusion 各保存完整 100×100 score matrix；所有 prediction 保存後才解析 ground_truth.csv。四方法在同一次評估 invocation 中各評分一次。

## 4. Paired bootstrap

固定 1,000 resamples／seed 42；每次同一 question index 同時抽取 Opening 與 Fusion，保留差值 covariance。95% CI 使用 percentile 2.5／97.5；mean 是原始 question 平均，bootstrap_mean 是重抽樣平均。

{bootstrap_table}

{ci_statement}

## 5. Generalization comparison

只使用保存的歷史 summary，沒有讀取歷史原始 validation／TEST 做 inference。

{general_table}

{table(differences,['partition','fusion_minus_opening'])}

三批 frozen 方法排名一致：**{same_rank}**（Fusion > Opening > Triplet）。不同玩家批次的 difficulty 可能不同，不把不同 partition 的分數差直接當成模型進步。

## 6. Error overlap（Top-1）

{table([{'category':key,'questions':value} for key,value in overlap.items()],['category','questions'])}

Triplet 單獨補對 Opening 錯誤：{overlap['triplet_only_correct']} 題；融合後補對 Opening 錯誤：{overlap['fusion_correct_when_opening_wrong']} 題。兩者不等同；融合也可能改錯原本 Opening 正確的題目。

## 7. 結果解讀

1. Color-aware Opening Top-1={o['top1']:.2%}／score={o['competition_score']:.6f}，相對 Random score={m['random']['competition_score']:.6f} 仍有高辨識能力。
2. Triplet score 高於 Random：{t['competition_score']>m['random']['competition_score']}；但先前 Stability 的 representation collapse 警示仍未解決，本輪沒有依 TEST 結果修改模型。
3. Fusion score 高於 Opening：{delta>0}，差值 **{delta:+.6f}**。{ci_statement}
4. DEV2／Stability／TEST3 排名一致：{same_rank}；正向 point estimate 不等同於提升已獲統計確認。
5. 互補性存在於部分問題，詳見 error overlap；未重選 alpha 或重跑。

## 8. CLOSED TEST、測試與操作

永久 **CLOSED TEST／evaluation_count=1**；保留原始 PREREGISTERED manifest，不修改其 count。Receipt 是完成狀態的權威紀錄。

Atomic reservation 在 preprocessing 前建立。中途失敗也標記 FAILED_CLOSED，禁止自動重跑。完成後所有 split CSV（包括 ground truth）再開啟及 inference command 再 invocation 都被 shared utils 的 process-wide audit hook 拒絕。報告只讀 saved receipt／summary，不重新評分。

完整 **{v['tests']} tests passed**，failure/error/skipped 均為 0；所有 Python imports／syntax check 通過。涵蓋 identity 排除、固定方法／checkpoint、no optimizer/backward、truth 延後、atomic registration、成功／失敗均不可二次 invocation、paired bootstrap 與相同 question/candidate axes。

正式 one-shot wall-clock：**{s['elapsed_seconds']:.2f} 秒（{s['elapsed_seconds']/60:.2f} 分鐘）**。

準備階段曾發現鎖阻擋首次 audit JSON 原子寫入；在 preregistration 前修正並保留相同 split SHA，新增 regression test。當時未保留 inference slot、未做推論／評分。正式 one-shot 沒有重跑。

```powershell
python scripts/check_phase210.py
# 已 CLOSED；下列 prepare / evaluation commands 現在都必須拒絕
python -m src.phase210_final_test3 --config configs/phase210.yaml --prepare
python -m src.phase210_final_test3 --config configs/phase210.yaml
# 只重建報告，不打開 CLOSED CSV，也不重新評分
python scripts/report_phase210.py
```

## 9. 下一個研究問題（本輪不實作）

- 在新的 DEV protocol 中找出 color-aware opening 的有效訊號是否依玩家顏色習慣／開局偏好而異，控制同色與跨色難度。
- 在 TRAIN／新的 DEV 上診斷 Triplet collapse、極小 cosine 差距與 aggregation，先建立 representation 判準，再提出改進。
- 事前固定 Fusion 假設與主要 endpoint，規劃更多獨立身份的驗證以縮小 paired difference CI；不使用 TEST3 調參。

本輪到此停止，不進 Phase 3，不加入 Strength Estimator／MiniZero／新模型，也不自動建立下一個 TEST。
'''
    (ROOT/'docs/phase210_results.md').write_text(text,encoding='utf-8')
    professor=f'''# Go Player Identification：教授會議更新

## Problem／Dataset

任務是由多盤 query 棋譜，辨識 100 位 candidate players 中的同一玩家。官方 train_A.csv：100,000 games／1,582 players。指標為 Top-1／3／5 與 competition score（正確名次 r≤5：exp(-(r-1))，未入 Top-5 為 0）。

TRAIN／VAL／TEST 分開玩家身份，避免識別已見玩家。Round 1 TEST、FINAL TEST 2 永久 CLOSED；本輪沒有重新讀其棋局、truth 或做 inference。

## Baseline evolution／關鍵發現

- Opening Fingerprint：用目標玩家實際落子建立 heatmap，平均多盤後 cosine retrieval。開局偏好已提供強訊號。
- Phase 2.8 color-aware-player-5：黑／白分開 fingerprint，DEV2 score **0.739568**，比非 color-aware player-5 **+0.103179**。
- Triplet-Hard-100：原 64-channel／8-block encoder，128 embedding；DEV2 best epoch 18、score **0.417385**。Opening 錯的 31 題中，Triplet 單獨補對 9 題。
- Cross-color gap 很大；配對 CrossColor positive 沒有改善。Stability 同／異玩家 cosine 都接近 1，separation 4.63e-9，全部 128 維近零 variance：**embedding collapse 尚未解決**。
- Frozen Fusion：每題 candidate scores 分別 z-score，alpha=0.9；DEV2 比 Opening **+0.018698**。模型、epoch、window、alpha 全部 frozen。

## Stability／FINAL TEST 3

Stability 使用 100 位新玩家，每人 candidate 30／query 10；score 差 **+0.028506**，paired bootstrap 95% CI **[−0.010277, 0.068760]**，未排除零改善。

FINAL TEST 3 排除歷史全部 **{s['split_counts']['historical_unique_players']} 位身份**，剩餘 {s['split_counts']['max_eligible_players']} 位至少有 40 盤；抽取 100 位完全新玩家，同樣 30／10 games。所有 identity／game／SGF overlap=0。Preregistration SHA256 綁定資料／split／config／checkpoint，13 項 preflight 通過，只有一次 inference／evaluation。

{general_table}

FINAL TEST 3 四方法正式結果：

{final_table}

Fusion−Opening **{delta:+.6f}**，paired bootstrap 95% CI **[{boot[2]['ci_lower']:.6f}, {boot[2]['ci_upper']:.6f}]**。{ci_statement} 三批方法排名一致：{same_rank}。Triplet 單獨補對 Opening 錯誤 {overlap['triplet_only_correct']} 題；Fusion 補對 {overlap['fusion_correct_when_opening_wrong']} 題。

## 限制與下一個研究問題

各 partition 玩家難度可能不同；100 questions 的 CI 仍寬，DEV2 已反覆選模型／alpha，不能把 point estimate 改善當作最終定論。Triplet collapse 尚未解決，跨色效果仍弱。

下一步只提出研究建議：在新的 DEV protocol 控制同色／跨色難度；先診斷 representation collapse 與 aggregation；事前固定 Fusion 假設，以更多獨立身份驗證其改善。

**FINAL TEST 3 已永久 CLOSED，evaluation_count=1。{v['tests']} tests／imports／syntax 全通過。**不使用 TEST3 調參，不重跑，不進 Phase 3；不加入 Strength Estimator、MiniZero 或新模型。
'''
    (ROOT/'docs/professor_update.md').write_text(professor,encoding='utf-8')
    print('Updated docs/phase210_results.md and docs/professor_update.md from saved receipt only')


if __name__=='__main__':
    main()
