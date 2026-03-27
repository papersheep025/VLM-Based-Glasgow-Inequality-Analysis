# Glasgow 城市不平等分析
## 训练前准备
    
将文件夹dataset和SIMD放入主文件夹

## 目录说明

- `scripts/build_streetview_prefix_satellite_alignment.py`
  - 生成街景与遥感 patch 的对齐数据
- `scripts/build_streetview_ntl_alignment.py`
  - 生成街景与夜光 patch 的对齐数据
- `scripts/build_vlm_jsonl.py`
  - 把对齐结果整理成模型可读的 JSONL
- `scripts/smoke_test_vlm_pipeline.py`
  - 检查数据、图片和 JSONL 是否正常
- `scripts/predict_qwen3_vl_plus_api.py`
  - 直接调用 `Qwen3-VL-Plus` API 做小批量试运行和批量预测
- `scripts/evaluate_predictions.py`
  - 计算分类与回归指标
- `scripts/aggregate_datazone_predictions.py`
  - 把样本级预测聚合成 `datazone` 级结果

## 当前数据结构

原始数据和派生数据的建议目录如下：

- `dataset/streetview_dataset`
  - 原始街景图
- `dataset/satellite_dataset`
  - 原始遥感数据
  - `satellite_patches` 为原始卫星 patch
  - `satellite_ntl_patches` 为夜光 patch
- `dataset/streetview_satellite_aligned`
  - 街景 + 遥感 对齐结果
- `dataset/streetview_ntl_aligned`
  - 街景 + 夜光 对齐结果
- `outputs/predictions`
  - 模型预测输出
- `src/glasgow_vlm`
  - 提示词、数据集封装、指标计算

## 环境准备

如果你已经有 `.venv`，直接安装当前依赖即可：

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```


## API Key

`scripts/predict_qwen3_vl_plus_api.py` 内已经预留了 API Key 位置，将API Key填入：

```python
DASHSCOPE_API_KEY = "你的DashScope API Key"
```


## 街景-遥感双模态分析
### 1. 生成街景与遥感对齐表

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_prefix_satellite_alignment.py
```

输出会放在：

- `dataset/streetview_satellite_aligned/streetview_prefix_satellite_alignment.csv`
- `dataset/streetview_satellite_aligned/streetview_prefix_satellite_alignment.json`
- `dataset/streetview_satellite_aligned/alignment_summary.json`

### 2. 生成 VLM JSONL

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_aligned/streetview_prefix_satellite_alignment.csv --output-dir dataset/streetview_satellite_aligned/vlm_data --input-mode dual --task ordinal --secondary-modality satellite
```

会生成：

- `dual_ordinal_train.jsonl`
- `dual_ordinal_val.jsonl`
- `dual_ordinal_test.jsonl`
- `dual_ordinal_all.jsonl`

### 3. 做烟雾测试

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_vlm_pipeline.py --jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_train.jsonl --num-samples 8
```

### 4. 用 Qwen3-VL-Plus API 做小批量预览

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_preview.jsonl --max-samples 5
```

这一步会先做一个很小的 API 连通性检查，再输出前 5 条预测结果。

### 5. 跑完整预测

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_test.jsonl
```

### 6. 计算指标

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_test.jsonl
```

### 7. 聚合到 datazone 级别

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_datazone_predictions.py --pred-jsonl outputs/predictions/qwen3_vl_plus_test.jsonl --gold-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --output-csv outputs/datazone_predictions.csv
```

## 街景-夜光双模态分析


### 1. 生成街景-夜光对齐表

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_ntl_alignment.py
```

输出会放在：

- `dataset/streetview_ntl_aligned/streetview_ntl_alignment.csv`
- `dataset/streetview_ntl_aligned/streetview_ntl_alignment.json`
- `dataset/streetview_ntl_aligned/alignment_summary.json`
- `dataset/streetview_ntl_aligned/satellite_ntl_patches`

### 2. 生成夜光版 VLM JSONL

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_ntl_aligned/streetview_ntl_alignment.csv --output-dir dataset/streetview_ntl_aligned/vlm_data --input-mode dual --task ordinal --secondary-modality ntl
```

这条命令和原来的双模态流程是同一套结构，但 prompt 里会把第二张图明确写成夜光图。

### 3. 做夜光版烟雾测试

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_vlm_pipeline.py --jsonl dataset/streetview_ntl_aligned/vlm_data/dual_ordinal_train.jsonl --num-samples 8
```

### 4. 用 Qwen3-VL-Plus API 做夜光预览

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_ntl_preview.jsonl --max-samples 5
```

## 结果预期

如果流程跑通，你最终会得到：

- 原始卫星版的预测结果
- 夜光版的预测结果
- 两套各自的评估指标
- `datazone` 级聚合表
- 可用于论文的空间不平等分析图表




## 街景-遥感-夜光三模态分析
如果你想把 streetview + satellite + nightlight 一起送进模型，可以先做小批量试运行，确认三张图的读取和 API 输出都正常。

### 1. 生成三模态对齐表

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_satellite_ntl_alignment.py
```

输出会放在：

- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv`
- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.jsonl`
- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_summary.json`

### 2. 生成三模态 VLM JSONL

```powershell
.\.venv\Scripts\python.exe scripts/build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv --output-dir dataset/streetview_satellite_ntl_aligned/vlm_data --input-mode triple --task ordinal
```

会生成：

- `triple_ordinal_train.jsonl`
- `triple_ordinal_val.jsonl`
- `triple_ordinal_test.jsonl`
- `triple_ordinal_all.jsonl`

### 3. 三模态 smoke test

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_vlm_pipeline.py --jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_ordinal_train.jsonl --num-samples 5
```

### 4. 三模态小批量 API 预览

```powershell
.\.venv\Scripts\python.exe scripts/predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_preview.jsonl --input-mode triple --max-samples 5
```

这条命令会同时发送街景、卫星和夜光三张图，适合在正式跑完整集之前先看模型输出风格和稳定性。

## 全量运行说明



### 1. 街景-遥感全量运行

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_test.jsonl
```

### 2. 街景-夜光全量运行

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_ntl_test.jsonl
```

### 3. 街景-遥感-夜光三模态全量运行

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_ordinal_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --input-mode triple
```

### 4. 全量结果评估

街景-遥感评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_ordinal_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_test.jsonl
```
街景-夜光评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_ordinal_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_ntl_test.jsonl
```

街景-遥感-夜光评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_ordinal_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl
```

全量结果汇总到 `datazone` 层，也可以继续接 `scripts/aggregate_datazone_predictions.py`。
