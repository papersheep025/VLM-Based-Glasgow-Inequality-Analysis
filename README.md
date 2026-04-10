# VLM Based Glasgow Inequality Analysis

本项目使用视觉语言模型研究格拉斯哥城市不平等与贫困分布。

主流程：
- 构建空间对齐表
- 生成 VLM JSONL 文件
- 用 Qwen3-VL-Plus 的 API 预测
- 评估预测结果
- 按 datazone 进行聚合分析

## 项目结构

- `assets`
  - 格拉斯哥数据区域、遥感和夜光示意图
- `data_processing`
  - 数据清洗、聚合、POI 提取和不平等分析脚本
- `data_scraper`
  - 数据集构建与街景/遥感对齐脚本
- `dataset`
  - `streetview_dataset`
    - 原始街景图像和 `metadata.csv`
  - `satellite_dataset`
    - 原始遥感数据、街景同名遥感 patch、夜光 patch 和相关元数据
    - `sat_dataset/sat_image`
      - 遥感 patch 的另一份整理目录
    - `sat_dataset/ntl_image`
      - 夜光 patch 的另一份整理目录
    - `satellite_patches`
      - 遥感图像采样图片（按街景图片名命名）
    - `satellite_ntl_patches`
      - 夜光图像采样图片（按街景图片名命名）
    - `satellite_metadata.csv`
      - 遥感 patch 元数据，包含 `id`、`datazone` 和 333m 方形区域边界
    - `ntl_metadata.csv`
      - 夜光 patch 元数据，bbox 直接沿用遥感 patch 的边界
    - `sat_ntl_metadata.csv`
      - 遥感 + 夜光联合元数据，包含两种 patch 路径和共享 bbox
  - `sat_ntl_svi_aligned`
    - 不含 SIMD 的街景 + 遥感 + 夜光对齐结果
    - `vlm_data`
      - 生成的 VLM JSONL 文件
  - `sat_ntl_svi_aligned_with_simd`
    - 含 SIMD 的街景 + 遥感 + 夜光对齐结果
    - `vlm_data`
      - 生成的 VLM JSONL 文件
  - `dashscope_preview`
    - DashScope 打包预览文件
- `outputs`
  - 模型预测、SIMD 导出和其它实验输出
- `processed_data`
  - `datazone_data`
    - 按 datazone 整理后的中间结果
  - `inequality_data`
    - 不平等分析相关中间结果
- `scripts`
  - 对齐、JSONL 生成、评估和预测脚本
- `src/glasgow_vlm`
  - prompt、数据读取、分层切分和指标计算等辅助代码
- `WORKFLOW.md`
  - 数据处理和建模流程说明

## 环境配置

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 空间对齐

当前仓库保留的空间对齐脚本是三模态对齐：

### 遥感 + 夜光 + 街景：

```powershell
.\.venv\Scripts\python.exe scripts\build_sat_ntl_svi_alignment.py
```

输出默认写入 `dataset/sat_ntl_svi_aligned/alignment.csv`。

如果需要再合并 SIMD 标签，可以基于 `dataset/sat_ntl_svi_aligned/alignment_without_simd.csv` 做后处理，最终结果会放到 `dataset/sat_ntl_svi_aligned_with_simd/alignment.csv`。

### 遥感 patch 元数据

如果你需要单独导出遥感 patch 的元数据，可以运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_satellite_metadata.py
```

默认会读取 `dataset/streetview_dataset/metadata.csv`，并生成 `dataset/satellite_dataset/satellite_metadata.csv`。

### 夜光 patch 元数据

如果你需要单独导出夜光 patch 的元数据，可以运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_ntl_metadata.py
```

默认会读取 `dataset/satellite_dataset/satellite_metadata.csv`，并生成 `dataset/satellite_dataset/ntl_metadata.csv`。

### 遥感 + 夜光联合元数据

如果你需要把遥感和夜光元数据合并成一张表，可以运行：

```powershell
.\.venv\Scripts\python.exe scripts\build_sat_ntl_metadata.py
```

默认会读取 `dataset/satellite_dataset/satellite_metadata.csv` 和 `dataset/satellite_dataset/ntl_metadata.csv`，并生成 `dataset/satellite_dataset/sat_ntl_metadata.csv`。

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

### 跑小批量进行测试
街景 + 遥感 + 夜光三模态
--max-samples可以改测试数量
Win
```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_preview.jsonl --input-mode triple --task explain --max-samples 5
```
Mac
```powershell
python scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/sat_ntl_svi_triple_preview.jsonl \
  --input-mode triple --task explain --max-samples 5
```


## 全量运行

### 街景 + 遥感 + 夜光三模态
Win
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


$Relative_i,k = (x_i,k - min_d(x_k)) / (max_d(x_k) - min_d(x_k) + epsilon)$


其中 epsilon 是一个很小的常数，用于避免除零。如果某个 `datazone` 内所有值都相同，则脚本会返回 0.5 作为默认相对分数。

## 城市不平等指标

### Income

### Employment

### Environment

### Education

### Health

### Housing

### Crime
