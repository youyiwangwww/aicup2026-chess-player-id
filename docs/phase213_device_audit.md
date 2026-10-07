# Phase 2.13 device provenance audit

Audit date：2026-10-07（Asia/Taipei）。只讀保存紀錄，沒有重跑training，沒有載入GPU模型或查詢目前GPU來推斷歷史環境。

| 欄位 | 可確認程度 | 既有證據／限制 |
| --- | --- | --- |
| training device | 強證據支持CUDA | `src/run_phase213.py:train_one`要求`choose_device('cuda')`，helper在CUDA unavailable時直接raise，沒有CPU fallback；各新增seed完成20epochs；123/2026/31415的A0 best.pt pickle storage location tags均包含`cuda:0`。用zipfile/pickletools讀metadata，沒有執行pickle或inference。 |
| GPU model | **無法追溯** | run.log、protocol、state、seed_audit未保存device_name。`cuda:0`只表示logical device，不能推出GPU型號。 |
| PyTorch version | **無法從實驗紀錄確證** | 未保存torch_version；checkpoint ZIP的serialization version不是PyTorch package version。run.log中CUDA Python目錄名稱也不是版本證據。 |
| CUDA version | **無法從實驗紀錄確證** | 未保存torch.version.cuda、driver/runtime資訊。不能從CUDA成功或`cuda:0`推出版本。 |

Seed42為Phase2.12保存結果reference，沒有在Phase2.13重訓。不可把現在環境查詢、對話中曾提到的GPU資訊或今日套件版本回填成當時device provenance。

結論：能證明實驗要求CUDA，並有保存CUDA tensors與成功run的佐證；**不足以提供完整GPU型號／PyTorch／CUDA版本鏈**。不補造歷史欄位、不重跑training。

## 未來強制provenance schema

在任何GPU experiment開始前保存並綁定run signature：

```json
{
  "device_type": "cuda",
  "device_name": "<torch.cuda.get_device_name(actual_device)>",
  "torch_version": "<torch.__version__>",
  "cuda_version": "<torch.version.cuda>"
}
```

另外建議保存device index、driver、cuDNN版本、Python／OS、deterministic settings、config/split/cache/code/checkpoint SHA256、timestamp。若四個必要欄位不足，未來runner須在training前拒絕，而不是結果完成後猜測。Phase3.0沒有GPU experiment，這是integration contract，不修改已frozen Phase2 runner。
