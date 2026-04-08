# VLM Based Glasgow Inequality Analysis

本项目使用视觉语言模型研究格拉斯哥城市不平等与贫困分布。

主流程：
- 构建空间对齐表
- 生成 VLM JSONL 文件
- 用 Qwen3-VL-Plus 的 API 预测
- 评估预测结果
- 按 datazone 进行聚合分析

## 项目结构

- `dataset/streetview_dataset`
  - 原始街景图像
- `dataset/satellite_dataset`
  - 原始遥感 TIFF 和派生 patch
  - `satellite_patches` 遥感图像采样图片（732*732，按街景图片名命名）
  - `satellite_ntl_patches` 夜光图像采样图片（256*256，按街景图片名命名）
- `dataset/streetview_satellite_aligned`
  - 街景 + 遥感双模态对齐结果
- `dataset/streetview_ntl_aligned`
  - 街景 + 夜光双模态对齐结果
- `dataset/satellite_ntl_aligned`
  - 遥感 + 夜光双模态对齐结果
- `dataset/streetview_satellite_ntl_aligned`
  - 街景 + 遥感 + 夜光三模态对齐结果
- `outputs/predictions`
  - API 预览和全部输出
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

这些脚本只生成对齐 CSV 和摘要文件，不再合并 SIMD 标签。

## OpenStreetMap POI 提取

可以直接根据 `glasgow_datazone/glasgow_datazone.shp` 提取 OSM POI 数据：

```powershell
.\.venv\Scripts\python.exe data_processing\osm_poi.py --output-dir outputs\osm_poi
```

默认会输出：
- `outputs/osm_poi/osm_poi.csv`
- `outputs/osm_poi/osm_poi_by_datazone.csv`

## 生成 JSONL

`scripts/build_vlm_jsonl.py` 会把对齐 CSV 转成 VLM JSONL 文件。

支持的 `--input-mode`：
- `streetview`
- `satellite`
- `satellite_ntl`
- `triple`


街景 + 遥感 explain 示例生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_aligned/streetview_prefix_satellite_alignment.csv --output-dir dataset/streetview_satellite_aligned/vlm_data --input-mode dual --task explain
```

街景 + 夜光 explain 示例生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_ntl_aligned/streetview_ntl_alignment.csv --output-dir dataset/streetview_ntl_aligned/vlm_data --input-mode dual --task explain
```

遥感 + 夜光 explain 示例生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/satellite_ntl_aligned/satellite_ntl_alignment.csv --output-dir dataset/satellite_ntl_aligned/vlm_data --input-mode satellite_ntl --task explain
```

街景 + 遥感 + 夜光三模态 explain 示例生成：

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv --output-dir dataset/streetview_satellite_ntl_aligned/vlm_data --input-mode triple --task explain
```

## Smoke Test

对生成的 JSONL 做快速检查：

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_vlm_pipeline.py --jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_train.jsonl --num-samples 5
```

smoke test 会检查：
- 必需字段是否存在
- 图像路径是否可读
- prompt 是否可解析
- 在当前“纯空间对齐”流程中，`answer_json` 是可选项

街景 + 遥感 + 夜光三模态

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_preview.jsonl --input-mode triple --task explain --max-samples 5
```

## 全量运行

全量跑要去掉 `--max-samples`。

### 街景 + 遥感

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_satellite.jsonl --input-mode dual --task explain
```

### 街景 + 夜光

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_ntl_aligned/vlm_data/dual_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_streetview_ntl.jsonl --input-mode dual --task explain
```

### 遥感 + 夜光

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/satellite_ntl_aligned/vlm_data/satellite_ntl_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_satellite_ntl.jsonl --input-mode satellite_ntl --task explain
```

### 街景 + 遥感 + 夜光三模态

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple.jsonl --input-mode triple --task explain
```

### 断点续跑

`predict_qwen3_vl_plus_api.py` 支持断点续跑：

- 已完成的 `id` 会自动跳过
- 空或不完整的记录会重新尝试
- 想从头开始可以在最后加 `--overwrite`


## Prompt 设计要点

`src/glasgow_vlm/prompts.py` 定义了 prompt 模板。

当前 prompt 的目标：
- 限定 `evidence` 的输出均为短语，大幅减少 token 使用
- 要求结构化 evidence，分成 `streetview`、`satellite`、`nightlight` 分别记录
- 要求模型从图像中推断 visual indicators
- 避免 Datazone ID “剧透”，防止大模型读取 ID 后调取 SIMD 数据干扰正常分析

## 预测指标和衍生指标

结构化结果保存在 `prediction_json` 中，字段层级如下：

```text
prediction_json
- predicted_quintile
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
### 基本指标
```text
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
```
### 衍生指标：空间异质性与相对指标
后处理时，可以使用 `data_processing/process_prediction_jsonl.py` 对任何结构相同的 prediction JSONL 进行处理。

这个脚本会保留关键字段、保留 `prediction_json`、按 `id` 去重，并增加以下派生字段：

- `spatial_heterogeneity`
- `relative_density`
- `relative_greenery`
- `relative_lighting`

#### 处理命令

```powershell
.\.venv\Scripts\python.exe data_processing\process_jsonl.py --input-jsonl outputs\predictions\qwen3_vl_plus_triple_preview.jsonl --output-jsonl processed_data\processed_data.jsonl
```



#### 空间异质性

对于某个 `datazone` d，设 x_i,k 表示该 `datazone` 中第 i 个样本在视觉指标 k 上的取值，K 表示可用视觉指标集合。

公式：

$SH_d = (1 / |K|) * sum_{k in K} (sigma_d,k / 0.5)$


其中，sigma_d,k 是同一 `datazone` 内指标 k 的标准差。由于所有指标都被缩放到 [0, 1]，除以 0.5 可以把结果近似归一化到 [0, 1]。

#### 相对指标

对于每个样本 i、`datazone` d，以及指标 k ≈ {density, greenery, lighting}，定义：

$
Relative_i,k = (x_i,k - min_d(x_k)) / (max_d(x_k) - min_d(x_k) + epsilon)
$

其中 epsilon 是一个很小的常数，用于避免除零。如果某个 `datazone` 内所有值都相同，则脚本会返回 0.5 作为默认相对分数。

## 城市不平等指标

### Income


### Employment


### Environment


### Education


### Health


### Housing


### Crime

