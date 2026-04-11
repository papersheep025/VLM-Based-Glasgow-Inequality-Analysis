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


## Prompt 模块

所有 prompt 模板放在 `src/glasgow_vlm/prompts/`，通过 `--prompt` 参数切换：

| `--prompt` 值 | 文件 | 输出字段 | 建议 `--max-new-tokens` |
| --- | --- | --- | --- |
| `simple` | `prompts/simple.py` | 7个域评分 + overall（无 CoT） | 512 |
| `structured` | `prompts/structured.py` | 7个域评分 + overall | 1024（默认） |
| `structured_plus` | `prompts/structured_plus.py` | 7个域评分 + overall + 8个细分变量评分 | 1024 |
| `structured_reasoning` | `prompts/structured_reasoning.py` | reasoning + 7个域评分 + overall | 2048 |
| `default` | `prompts/default.py` | evidence + visual_indicators | 1024 |

所有 prompt 的输出均包含：`income`, `employment`, `health`, `education`, `housing`, `access`, `crime`（1–10整数）和 `overall`（浮点，加权公式：`0.12×income + 0.12×employment + 0.02×health + 0.01×education + 0.06×housing + 0.06×access + 0.04×crime`）。

### Few-Shot 链路

除了上述 zero-shot prompt 模块，项目还提供了基于 `structured_plus` 的 few-shot 推理链路。

**原理：** 在 API 请求中构建多轮对话，将 `dataset/few_shot_examples/few_shot_examples.json` 中的示例（每个 quintile 各一个，共 5 个）作为 user+assistant 轮次插入到 system prompt 和实际查询之间，让模型参考真实标注样本进行打分。

**消息结构：**

```text
system  →  user(示例1图片+prompt) → assistant(示例1标注)
        →  user(示例2图片+prompt) → assistant(示例2标注)
        →  ...
        →  user(待预测图片+prompt)
```

**相关文件：**

- `src/glasgow_vlm/prompts/structured_plus_fewshot.py` — few-shot prompt 模块，复用 `structured_plus` 的 prompt 模板，新增示例加载、中间变量推导和示例回复构建
- `scripts/predict_fewshot_api.py` — few-shot 预测脚本，从原始 `predict_qwen3_vl_plus_api.py` 导入工具函数
- `dataset/few_shot_examples/few_shot_examples.json` — 5 个代表性样本（quintile 1–5），包含图片路径和 ground_truth_scores

**专属参数：**

- `--few-shot-json` — few-shot 数据路径（默认 `dataset/few_shot_examples/few_shot_examples.json`）
- `--few-shot-count` — 使用的示例数量（默认全部 5 个）
- `--few-shot-quintiles` — 指定使用哪些 quintile（如 `1 3 5`）

**注意：** triple 模式下每个 few-shot 示例会额外附带 3 张图片。5 个示例 = 15 张额外图片 + 查询 3 张 = 18 张，token 开销较大。如遇 API 限制可减少 `--few-shot-count`。

### Qwen3-VL-8B LoRA 微调

这条链路用于把 `structured_plus` 的评分能力从 API 推理迁移到本地 LoRA 微调。

**核心思路：**

- 微调原因：API 链路是 zero-shot / few-shot 推理，模型没有直接学习过 SIMD 标注对，LoRA 让模型在真实图片-标注样本上学习“看图打分”。
- 采用 LoRA：只训练低秩适配器，显存和参数量都比全量微调更可控，适合本地 32GB 机器。
- 冻结视觉编码器：保留已预训练的图像特征提取能力，只更新语言侧的理解、推理和输出层。

**数据流：**

```text
SIMD_data.csv
  ↓ 按 datazone 合并标签
train / val JSONL
  ↓ 转换为训练格式
mlx-vlm LoRA 数据
```

每条样本包含 `images`（3 张图片路径）和 `messages`（system + user + assistant 对话）。
assistant 目标包含 16 个 key：7 个 domain score、1 个 overall、8 个由 `derive_intermediate_scores` 推导的中间变量。

**相关文件：**

- `scripts/build_lora_training_data.py` - 合并 SIMD 标签并生成 `mlx-vlm` 训练数据
- `scripts/train_qwen3_vl_lora.py` - LoRA 训练启动脚本
- `scripts/predict_local_qwen3_vl.py` - 本地 base + adapter 推理脚本

**当前进度：**

- 已完成 `build_lora_training_data.py`
- 训练集：20,081 条样本，474 个 datazone
- 验证集：2,662 条样本，59 个 datazone
- 数据目录：`dataset/lora_training_data/`
- `messages` 目前以 JSON 字符串存储，便于规避 PyArrow 嵌套类型问题；训练加载时可能需要 `json.loads` 或自定义 wrapper

**后续步骤：**

1. 先用小子集验证 `mlx-vlm` 的训练接口是否能正常跑通
2. 再补充 LoRA 训练脚本和本地推理脚本
3. 最后在同一 test set 上和 API baseline / few-shot 结果做统一评估

### 跑小批量进行测试

`--max-samples` 控制测试数量，`--overwrite` 从头覆盖写。

**simple（直接评分，无 CoT，token 最少）：**

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_simple_preview.jsonl \
  --input-mode triple --task explain \
  --prompt simple \
  --max-new-tokens 512 \
  --max-samples 5 --overwrite
```

**structured（只输出评分，token 少，推荐用于全量）：**

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_preview.jsonl \
  --input-mode triple --task explain \
  --prompt structured \
  --max-samples 5 --overwrite
```

**structured_plus（含细分变量评分，无 reasoning 文本）：**

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_plus_preview.jsonl \
  --input-mode triple --task explain \
  --prompt structured_plus \
  --max-samples 5 --overwrite
```

**structured_reasoning（含推理过程，需要更多 token）：**

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_reasoning_preview.jsonl \
  --input-mode triple --task explain \
  --prompt structured_reasoning \
  --max-new-tokens 2048 \
  --max-samples 5 --overwrite
```

**few-shot（基于 structured_plus，附带真实标注示例）：**

```bash
python3 scripts/predict_fewshot_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_preview.jsonl \
  --input-mode triple --task explain \
  --few-shot-count 3 \
  --max-samples 5 --overwrite
```

## 全量运行

### simple（token 消耗最少）

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_simple_full.jsonl \
  --input-mode triple --task explain \
  --prompt simple \
  --max-new-tokens 512
```

### structured_plus

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_plus_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured_plus
```

### structured（推荐）

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured
```

### structured_reasoning

```bash
python3 scripts/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_reasoning_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured_reasoning \
  --max-new-tokens 2048
```

### few-shot（全部 5 个示例）

```bash
python3 scripts/predict_fewshot_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_full.jsonl \
  --input-mode triple --task explain
```

仅使用 quintile 1 和 5 作为对比示例：

```bash
python3 scripts/predict_fewshot_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_q15_full.jsonl \
  --input-mode triple --task explain \
  --few-shot-quintiles 1 5
```

### 断点续跑

`predict_qwen3_vl_plus_api.py` 支持断点续跑：

- 已完成的 `id` 会自动跳过
- 想从头开始可以在最后加 `--overwrite`

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
