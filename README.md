# VLM Based Glasgow Inequality Analysis

本项目使用视觉语言模型研究格拉斯哥城市不平等与贫困分布。

主流程：
- 构建空间对齐表
- 生成 VLM JSONL 文件
- 用 Qwen3-VL-Plus 做全量 API 预览
- 评估预测结果
- 按 datazone 进行聚合分析


## 项目结构

- `dataset/streetview_dataset`
  - 原始街景图像
- `dataset/satellite_dataset`
  - 原始遥感 TIFF 
  - `satellite_patches`遥感图像采样图片（256*256）
  - `satellite_ntl_patches`夜光图像采样图片（512*512）
- `dataset/streetview_satellite_aligned`
  - 街景 + 遥感双模态对齐结果
- `dataset/streetview_ntl_aligned`
  - 街景 + 夜光双模态对齐结果
- `dataset/satellite_ntl_aligned`
  - 遥感 + 夜光双模态对齐结果
- `dataset/streetview_satellite_ntl_aligned`
  - 街景 + 遥感 + 夜光三模态对齐结果
- `outputs/predictions`
  - API 预测预览和完整输出
- `src/glasgow_vlm`
  - prompt、数据读取和指标计算等辅助代码

## 环境配置

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 空间对齐

仓库提供四个空间对齐脚本：

### 街景 + 遥感：

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_prefix_satellite_alignment.py
```

### 街景 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_ntl_alignment.py
```

### 遥感 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\build_satellite_ntl_alignment.py
```

### 街景 + 遥感 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_satellite_ntl_alignment.py
```

这些脚本生成对齐 CSV 和摘要文件，并且不合并 SIMD 标签

## 生成 JSONL

`scripts/build_vlm_jsonl.py` 会把对齐 CSV 转成 VLM JSONL 文件。

支持的 `--input-mode`：
- `streetview`
- `satellite`
- `dual`
- `satellite_ntl`
- `triple`

支持的 `--task`：
- `explain`

街景 + 遥感 explain 示例：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_aligned/streetview_prefix_satellite_alignment.csv --output-dir dataset/streetview_satellite_aligned/vlm_data --input-mode dual --task explain
```

街景 + 夜光 explain 示例：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_ntl_aligned/streetview_ntl_alignment.csv --output-dir dataset/streetview_ntl_aligned/vlm_data --input-mode dual --task explain
```

遥感 + 夜光 explain 示例：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/satellite_ntl_aligned/satellite_ntl_alignment.csv --output-dir dataset/satellite_ntl_aligned/vlm_data --input-mode satellite_ntl --task explain
```
街景 + 遥感 + 夜光三模态 explain 示例：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv --output-dir dataset/streetview_satellite_ntl_aligned/vlm_data --input-mode triple --task explain
```

## Smoke Test 检查链路能否跑通

对生成的 JSONL 做快速检查：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_vlm_pipeline.py --jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_train.jsonl --num-samples 5
```

smoke test 会检查：
- 必需字段是否存在
- 图像路径是否可读
- prompt 是否可解析
- 在当前“纯空间对齐”流程中，`answer_json` 是可选项

## 跑小批量，检查输出格式和内容

主推理脚本是 `scripts/predict_qwen3_vl_plus_api.py`，可以修改`--max-samples`的数量调整需要预览的样本个数

### 街景 + 遥感

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_satellite_preview.jsonl --input-mode dual --task explain --max-samples 5
```

### 街景 + 夜光

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_ntl_preview.jsonl --input-mode dual --task explain --max-samples 5
```

### 遥感 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/satellite_ntl_aligned/vlm_data/satellite_ntl_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_satellite_ntl_preview.jsonl --input-mode satellite_ntl --task explain --max-samples 5
```

### 街景 + 遥感 + 夜光三模态：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_preview.jsonl --input-mode triple --task explain --max-samples 5
```

## 全量运行

全量跑要去掉 `--max-samples`：

### 街景 + 遥感：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_satellite.jsonl --input-mode dual --task explain
```

### 街景 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_ntl.jsonl --input-mode dual --task explain
```

### 遥感 + 夜光：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/satellite_ntl_aligned/vlm_data/satellite_ntl_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_satellite_ntl.jsonl --input-mode satellite_ntl --task explain
```

### 街景 + 遥感 + 夜光三模态：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple.jsonl --input-mode triple --task explain

### 断点续跑

`predict_qwen3_vl_plus_api.py` 支持断点续跑：

- 已完成的 `id` 会自动跳过
- 空或不完整的记录会重新尝试
- 想从头开始可以加 `--overwrite`

示例：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --input-mode triple --task explain --overwrite
```
## Prompt 设计要点

`src/glasgow_vlm/prompts.py` 定义了 prompt 模板。

当前 prompt 的目标：
- 限定 `evidence` 的输出均为短语，大幅减少 token 使用
- 要求结构化 evidence，分成 `streetview`, `satellite`, `nightlight` 分别记录
- 要求模型从图像中推断 visual indicators
- 避免 Datazone ID 泄漏，大模型读取 ID 信息后调取 SIMD 数据干扰正常分析


## 预测JSON输出格式

结构化结果保存在 `prediction_json` 中。

字段层级如下：

```text
prediction_json
- predicted_quintile or above_median_deprivation
- predicted_rank_band
- confidence
- visual_indicators
  - density
  - greenery
  - lighting
  - infrastructure
  - building_condition
  - land_use_mix
  - cleanliness
  - accessibility
  - vehicle_presence
  - housing_type
  - vacancy
- evidence
  - streetview
  - satellite
  - nightlight
```
## 评估

对预测结果进行评估：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl
```

## Datazone 聚合

按 datazone 聚合预测结果：

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_datazone_predictions.py --pred-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --gold-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-csv outputs/datazone_predictions.csv
```

