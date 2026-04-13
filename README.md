# VLM Based Glasgow Inequality Analysis

本项目使用视觉语言模型研究格拉斯哥城市不平等与贫困分布。

主流程：
- 构建空间对齐表
- 生成 VLM JSONL 文件
- 调用 Qwen3-VL-Plus 的 API 或者部署本地 Qwen3-VL-8b 预测 SIMD Domain 分数
- 按 datazone 进行聚合分析

## 实验总览

| 实验名称 | 学习范式 | Prompt 模块 | 模型 | POI 上下文 | 输出字段 |
| --- | --- | --- | --- | --- | --- |
| simple | zero-shot | `simple` | Qwen3-VL-Plus (API) | 无 | 7 域评分 + overall |
| structured | zero-shot | `structured` | Qwen3-VL-Plus (API) | 无 | 7 域评分 + overall |
| structured_plus | zero-shot | `structured_plus` | Qwen3-VL-Plus (API) | 无 | 7 域评分 + overall + 8 细分变量 |
| structured_reasoning | zero-shot | `structured_reasoning` | Qwen3-VL-Plus (API) | 无 | reasoning + 7 域评分 + overall |
| few-shot 5 例 | few-shot | `structured_plus_fewshot` | Qwen3-VL-Plus (API) | 无 | 7 域评分 + overall + 8 细分变量 |
| few-shot Q1&Q5 | few-shot | `structured_plus_fewshot` | Qwen3-VL-Plus (API) | 无 | 7 域评分 + overall + 8 细分变量 |
| few-shot + POI 5 例 | few-shot | `structured_fewshot_poi` | Qwen3-VL-Plus (API) | 有 | 7 域评分 + overall + 8 细分变量 |
| few-shot + POI Q1&Q5 | few-shot | `structured_fewshot_poi` | Qwen3-VL-Plus (API) | 有 | 7 域评分 + overall + 8 细分变量 |
| LoRA 微调（纯图像） | fine-tuning | `structured_plus` | Qwen3-VL-8B-4bit (本地) | 无 | 7 域评分 + overall + 8 细分变量 |
| LoRA 微调（图像+POI） | fine-tuning | `structured_fewshot_poi` | Qwen3-VL-8B-4bit (本地) | 有 | 7 域评分 + overall + 8 细分变量 |

## 项目结构

- `assets`
  - 项目示意图与可视化资源
- `data_processing`
  - `simd_data.py`、`osm_poi.py`、`process_datazone_data.py`、`process_jsonl.py`
  - SIMD 清洗、POI 提取、不平等分析与聚合处理脚本
- `data_scraper`
  - `datazone_to_dataset_streetview.py`
  - `datazone_to_dataset_satellite.py`
  - `align_streetview_to_satellite.py`
  - 街景、遥感与夜光数据采集和对齐脚本
- `dataset`
  - `SIMD`
    - SIMD 原始数据
  - `glasgow_datazone`
    - 格拉斯哥 datazone 矢量数据
  - `streetview_dataset`
    - 原始街景图像和 `metadata.csv`
  - `satellite_dataset`
    - 原始遥感数据、街景同名遥感 patch、夜光 patch 和相关元数据
  - `sat_ntl_svi_aligned`
    - 不含 SIMD 的街景 + 遥感 + 夜光对齐结果
    - `vlm_data`
      - 生成的 VLM JSONL 文件
  - `sat_ntl_svi_aligned_with_simd`
    - 含 SIMD 的街景 + 遥感 + 夜光对齐结果
    - `vlm_data`
      - 生成的 VLM JSONL 文件
  - `few_shot_examples`
    - few-shot 示例样本
  - `osm_poi`
    - OpenStreetMap POI 中间结果
  - `lora_training_data`
    - 纯图像 LoRA 训练数据
  - `lora_training_data_poi`
    - 含 POI 上下文的 LoRA 训练数据
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
  - `data_prep`
    - 对齐、元数据构建、JSONL 生成和去重脚本
  - `evaluation`
    - 预测评估与烟雾测试脚本
  - `inference`
    - API 预测脚本
  - `training`
    - LoRA 训练与训练数据构建脚本
- `src/glasgow_vlm`
  - `config.py`
  - `data.py`
  - `metrics.py`
  - `splits.py`
  - `prompts/`
    - prompt 模板与 few-shot 辅助逻辑

## 环境配置

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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

# 调用API预测

## Zero-shot

### Zero-shot运行指令

加`--max-samples` 控制测试数量，加`--overwrite` 从头覆盖写。

### simple（直接评分，无 CoT，token 最少，每条900）

```bash
python3 scripts/inference/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_simple_full.jsonl \
  --input-mode triple --task explain \
  --prompt simple \
  --max-new-tokens 512
```

### structured（只输出评分，token 少，每条1400）

```bash
python3 scripts/inference/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured
```

### structured_plus（含中间变量评分，无 reasoning 文本，每条1600token）

```bash
python3 scripts/inference/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_structured_plus_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured_plus
```

### structured_reasoning（含推理过程，需要更多 token，每条3000）

```bash
python3 scripts/inference/predict_qwen3_vl_plus_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_reasoning_full.jsonl \
  --input-mode triple --task explain \
  --prompt structured_reasoning \
  --max-new-tokens 2048
```

## Few-Shot

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
- `scripts/inference/predict_fewshot_api.py` — few-shot 预测脚本，从原始 `predict_qwen3_vl_plus_api.py` 导入工具函数
- `dataset/few_shot_examples/few_shot_examples.json` — 5 个代表性样本（quintile 1–5），包含图片路径和 ground_truth_scores

**专属参数：**

- `--few-shot-json` — few-shot 数据路径（默认 `dataset/few_shot_examples/few_shot_examples.json`）
- `--few-shot-count` — 使用的示例数量（默认全部 5 个）
- `--few-shot-quintiles` — 指定使用哪些 quintile（如 `1 3 5`）

**注意：** triple 模式下每个 few-shot 示例会额外附带 3 张图片。5 个示例 = 15 张额外图片 + 查询 3 张 = 18 张，token 开销较大。如遇 API 限制可减少 `--few-shot-count`。

### Few-Shot + POI 链路

在 few-shot 链路的基础上，将每个 datazone 的 OpenStreetMap POI 统计信息以文本形式注入 prompt，为模型提供额外的地理空间上下文。

**原理：** 从 `dataset/osm_poi/datazone_poi.csv` 按 datazone 聚合 POI 计数，过滤噪声类型（bench、signal 等路面家具和铁路信号设备），按类别（Amenities、Shops、Public transport 等）生成摘要文本，附加在图像 prompt 末尾。few-shot 示例和实际查询均注入对应 datazone 的 POI 上下文。

**消息结构：**

```text
system  →  user(示例1图片+prompt+POI) → assistant(示例1标注)
        →  user(示例2图片+prompt+POI) → assistant(示例2标注)
        →  ...
        →  user(待预测图片+prompt+POI)
```

**相关文件：**

- `src/glasgow_vlm/prompts/structured_fewshot_poi.py` — POI prompt 模块，复用 `structured_plus_fewshot` 的 few-shot 逻辑，新增 POI 加载和格式化
- `scripts/inference/predict_fewshot_poi_api.py` — few-shot + POI 预测脚本
- `dataset/osm_poi/datazone_poi.csv` — 14558 条 POI 记录，覆盖 666 个 datazone

**专属参数：**

- `--poi-csv` — POI 数据路径（默认 `dataset/osm_poi/datazone_poi.csv`）
- 同时继承 few-shot 链路的 `--few-shot-json`、`--few-shot-count`、`--few-shot-quintiles` 参数

### few-shot 运行指令（全部 5 个示例，由于每次都要输入示例，所以token消耗量成倍增大）

```bash
python3 scripts/inference/predict_fewshot_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_full.jsonl \
  --input-mode triple --task explain
```

仅使用 quintile 1 和 5 作为对比示例：

```bash
python3 scripts/inference/predict_fewshot_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_q15_full.jsonl \
  --input-mode triple --task explain \
  --few-shot-quintiles 1 5
```

### few-shot + POI（全部 5 个示例 + POI 上下文）

```bash
python3 scripts/inference/predict_fewshot_poi_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_poi_full.jsonl \
  --input-mode triple --task explain
```

仅使用 quintile 1 和 5 作为对比示例：

```bash
python3 scripts/inference/predict_fewshot_poi_api.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/qwen3_fewshot_poi_q15_full.jsonl \
  --input-mode triple --task explain \
  --few-shot-quintiles 1 5
```

**断点续跑**
`predict_qwen3_vl_plus_api.py`、`predict_fewshot_api.py` 和 `predict_fewshot_poi_api.py` 均支持断点续跑：
- 已完成的 `id` 会自动跳过
- 想从头开始可以在最后加 `--overwrite`


# 部署本地模型 LoRA 微调（Qwen3-VL-8B）

使用 LM Studio 下载的 Qwen3-VL-8B-Instruct-MLX-4bit 进行 QLoRA 微调，在 Apple Silicon 上本地训练。

## Qwen3-VL-8B LoRA 微调

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

- `scripts/training/build_lora_training_data.py` - 合并 SIMD 标签并生成 `mlx-vlm` 训练数据
- `scripts/training/train_qwen3_vl_lora.py` - LoRA 训练启动脚本
- `scripts/inference/predict_local_qwen3_vl.py` - 本地 base + adapter 推理脚本

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

### Step 1 — 构建训练数据

将训练 JSONL 与 SIMD ground-truth 分数合并，生成 HuggingFace Dataset 格式的训练数据：

```bash
python3 scripts/training/build_lora_training_data.py
```

输出到 `dataset/lora_training_data/`，包含 train（20081 条）和 validation（2662 条）两个 split。

每条样本包含：

- `images`：3 张图片的路径（streetview + satellite + ntl）
- `messages`：Qwen3-VL 对话格式（system + user[3 图 + structured_plus prompt] + assistant[16-key JSON target]）

小样本测试：

```bash
python3 scripts/training/build_lora_training_data.py --max-samples 100
```

### Step 2 — LoRA 微调

模型路径默认为 `~/.lmstudio/models/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit`。

**小样本验证流程（推荐先跑通）：**

```bash
python3 scripts/training/train_qwen3_vl_lora.py \
  --max-samples 10 --epochs 1 --steps 5 --print-every 1
```

**正式训练：**

```bash
python3 scripts/training/train_qwen3_vl_lora.py \
  --epochs 2 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --save-every 500 \
  --grad-checkpoint
```

**如果遇到内存不足，可以降低图片分辨率：**

```bash
python3 scripts/training/train_qwen3_vl_lora.py \
  --epochs 2 --batch-size 1 --learning-rate 2e-5 \
  --image-resize-shape 256 256 --grad-checkpoint
```

**从 checkpoint 恢复训练：**

```bash
python3 scripts/training/train_qwen3_vl_lora.py \
  --adapter-path outputs/lora_adapters/checkpoint_ep1_step500.safetensors \
  --epochs 1 --learning-rate 1e-5
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
| ------ | -------- | ------ |
| `--model-path` | `~/.lmstudio/models/.../Qwen3-VL-8B-Instruct-MLX-4bit` | 本地模型路径 |
| `--dataset-dir` | `dataset/lora_training_data` | 训练数据目录 |
| `--epochs` | 2 | 训练轮数 |
| `--batch-size` | 1 | 批量大小（32GB 内存建议为 1） |
| `--learning-rate` | 2e-5 | 学习率 |
| `--lora-rank` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA alpha |
| `--lora-dropout` | 0.05 | LoRA dropout |
| `--save-every` | 500 | 每 N 步保存 checkpoint |
| `--grad-checkpoint` | false | 梯度检查点（省内存但更慢） |
| `--image-resize-shape` | 无 | 调整图片尺寸（如 `256 256`） |
| `--max-samples` | 0（全部） | 限制训练样本数 |

**输出：** LoRA adapter 保存到 `outputs/lora_adapters/adapters.safetensors`。

### 相关文件

- `scripts/training/build_lora_training_data.py` — 构建训练数据（合并 SIMD 标签 → HuggingFace Dataset）
- `scripts/training/train_qwen3_vl_lora.py` — LoRA 微调训练脚本
- `dataset/SIMD/SIMD_data.csv` — SIMD 分数数据（从 SIMD_rawdata.csv 提取，domain rank 映射为 1-10 分数）
- `dataset/lora_training_data/` — 生成的训练数据集

## 本地模型 LoRA 微调 + POI（Qwen3-VL-8B）

在纯图像 LoRA 链路基础上，将每个 datazone 的 OpenStreetMap POI 统计信息注入训练 prompt，让模型在微调阶段就学会结合 POI 上下文进行评分。

### 与纯图像链路的区别

| | 纯图像链路 | 图像 + POI 链路 |
| --- | --- | --- |
| 数据构建脚本 | `build_lora_training_data.py` | `build_lora_training_data_poi.py` |
| prompt 模块 | `structured_plus` | `structured_fewshot_poi`（附加 POI 文本） |
| 训练数据目录 | `dataset/lora_training_data/` | `dataset/lora_training_data_poi/` |
| adapter 输出 | `outputs/lora_adapters/` | `outputs/lora_adapters_poi/` |
| 训练脚本 | `train_qwen3_vl_lora.py` | `train_qwen3_vl_lora_poi.py` |

### Step 1 — 构建 POI 训练数据

将训练 JSONL 与 SIMD ground-truth 分数合并，同时注入 POI 上下文，生成 HuggingFace Dataset：

```bash
python3 scripts/training/build_lora_training_data_poi.py
```

输出到 `dataset/lora_training_data_poi/`，包含 train（20081 条）和 validation（2662 条）两个 split，POI 覆盖率 89.4%（624/746 个 datazone 有 POI 数据）。

每条样本的 user prompt 末尾会附加类似：

```text
Supplementary POI context (OpenStreetMap, this datazone):
  Amenities (12): 3x restaurant, 2x cafe, 2x fast_food, ...
  Shops (5): 2x convenience, 1x supermarket, ...
  Public transport (3): 3x bus_stop
```

小样本测试：

```bash
python3 scripts/training/build_lora_training_data_poi.py --max-samples 100
```

额外参数：

- `--poi-csv` — POI 数据路径（默认 `dataset/osm_poi/datazone_poi.csv`）

### Step 2 — LoRA 微调（图像 + POI）

训练参数与纯图像链路完全一致，仅默认路径不同。

**小样本验证流程：**

```bash
python3 scripts/training/train_qwen3_vl_lora_poi.py \
  --max-samples 10 --epochs 1 --steps 5 --print-every 1
```

**正式训练：**

```bash
python3 scripts/training/train_qwen3_vl_lora_poi.py \
  --epochs 2 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --save-every 500 \
  --grad-checkpoint
```

**从 checkpoint 恢复训练：**

```bash
python3 scripts/training/train_qwen3_vl_lora_poi.py \
  --adapter-path outputs/lora_adapters_poi/checkpoint_ep1_step500.safetensors \
  --epochs 1 --learning-rate 1e-5
```

**输出：** LoRA adapter 保存到 `outputs/lora_adapters_poi/adapters.safetensors`。

### 相关文件（图像 + POI）

- `scripts/training/build_lora_training_data_poi.py` — 构建 POI 训练数据（合并 SIMD 标签 + POI 上下文 → HuggingFace Dataset）
- `scripts/training/train_qwen3_vl_lora_poi.py` — LoRA 微调训练脚本（图像 + POI）
- `src/glasgow_vlm/prompts/structured_fewshot_poi.py` — POI prompt 模块（`load_poi_lookup`、`format_poi_context`）
- `dataset/osm_poi/datazone_poi.csv` — POI 原始数据
- `dataset/lora_training_data_poi/` — 生成的训练数据集

---

## 远程服务器 LoRA 微调 + 预测（CUDA）

本地 MLX 链路用于 Apple Silicon 开发验证，正式训练和预测在配有 GPU 的远程服务器上完成，使用 PyTorch + HuggingFace PEFT（QLoRA 4-bit）。

与本地 MLX 链路的核心区别：

| | 本地 MLX 链路 | 远程 CUDA 链路 |
| --- | --- | --- |
| 训练脚本 | `train_qwen3_vl_lora.py` | `train_qwen3_vl_lora_cuda.py` |
| POI 训练脚本 | `train_qwen3_vl_lora_poi.py` | `train_qwen3_vl_lora_poi_cuda.py` |
| adapter 格式 | mlx-vlm safetensors | PEFT（`adapter_config.json` + `adapter_model.safetensors`） |
| adapter 输出 | `outputs/lora_adapters/` | `outputs/lora_adapters_cuda/final_adapter/` |
| POI adapter 输出 | `outputs/lora_adapters_poi/` | `outputs/lora_adapters_poi_cuda/final_adapter/` |
| 推理脚本 | `predict_local_qwen3_vl.py` | `predict_lora_local.py` |

训练数据构建脚本（`build_lora_training_data.py` / `build_lora_training_data_poi.py`）两条链路共用，无需重复构建。

## Step 1 — 将项目文件同步到远程服务器

```bash
# 同步代码和数据（首次）
rsync -avz --exclude='.git' --exclude='outputs/' \
  /path/to/VLM-Based-Glasgow-Inequality-Analysis/ \
  user@server:/home/user/glasgow/

# 后续增量同步
rsync -avz dataset/ user@server:/home/user/glasgow/dataset/
```

如果服务器和本机图片路径不同，训练时加 `--path-prefix-remap` 参数重映射。

## Step 2 — 远程服务器环境

```bash
pip install torch torchvision transformers peft bitsandbytes accelerate datasets tqdm pillow
```

## Step 3 — LoRA 微调（纯图像）

**smoke test（先跑通）：**

```bash
python scripts/training/train_qwen3_vl_lora_cuda.py \
  --max-samples 20 --epochs 1 --print-every 5
```

**正式训练（单卡）：**

```bash
python scripts/training/train_qwen3_vl_lora_cuda.py \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 2e-5 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --save-every 500
```

**多卡训练（推荐）：**

```bash
accelerate launch --num_processes 2 \
  scripts/training/train_qwen3_vl_lora_cuda.py
```

**从 checkpoint 恢复：**

```bash
python scripts/training/train_qwen3_vl_lora_cuda.py \
  --resume-from outputs/lora_adapters_cuda/checkpoint_ep1_step500
```

**路径重映射（Mac → 服务器）：**

```bash
python scripts/training/train_qwen3_vl_lora_cuda.py \
  --path-prefix-remap /Users/papersheep/projects/VLM-Based-Glasgow-Inequality-Analysis:/home/user/glasgow
```

输出：adapter 保存到 `outputs/lora_adapters_cuda/final_adapter/`。

## Step 4 — LoRA 微调（图像 + POI）

与纯图像链路命令相同，仅换脚本名，默认输出到 `outputs/lora_adapters_poi_cuda/final_adapter/`：

```bash
# smoke test
python scripts/training/train_qwen3_vl_lora_poi_cuda.py \
  --max-samples 20 --epochs 1 --print-every 5

# 正式训练
python scripts/training/train_qwen3_vl_lora_poi_cuda.py \
  --epochs 2 --batch-size 1 --grad-accum-steps 4 \
  --learning-rate 2e-5 --lora-rank 16 --lora-alpha 32

# 多卡
accelerate launch --num_processes 2 \
  scripts/training/train_qwen3_vl_lora_poi_cuda.py
```

训练参数说明（两个脚本通用）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-id` | `Qwen/Qwen3-VL-8B-Instruct` | base 模型 HuggingFace ID 或本地路径 |
| `--epochs` | 2 | 训练轮数 |
| `--batch-size` | 1 | 批量大小 |
| `--grad-accum-steps` | 4 | 梯度累积步数（有效 batch = batch × accum） |
| `--learning-rate` | 2e-5 | 学习率 |
| `--max-length` | 2048 / 4096 | 最大 token 长度 |
| `--lora-rank` | 16 | LoRA 秩 |
| `--lora-alpha` | 32 | LoRA alpha |
| `--save-every` | 500 | 每 N 个 optimizer step 保存 checkpoint |
| `--no-quantization` | false | 跳过 4-bit 量化，用 BF16 加载（需更多显存） |
| `--grad-checkpoint` | false | 梯度检查点（省显存，速度变慢） |
| `--max-samples` | 0（全部） | 限制训练样本数（smoke test 用） |
| `--path-prefix-remap` | 无 | 图片路径重映射：`OLD:NEW` |
| `--resume-from` | 无 | 从指定 checkpoint 目录恢复训练 |

## Step 5 — 预测

训练完成后，`final_adapter/` 目录包含：

```text
final_adapter/
├── adapter_config.json        # LoRA 配置（rank、target_modules 等）
├── adapter_model.safetensors  # LoRA 权重
├── tokenizer.json
├── tokenizer_config.json
└── preprocessor_config.json
```

使用 `predict_lora_local.py` 将 LoRA adapter 叠加在 base model 上进行推理：

**smoke test（5 条）：**

```bash
# 纯图像 adapter
python scripts/inference/predict_lora_local.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/lora_triple_preview.jsonl \
  --adapter-path outputs/lora_adapters_cuda/final_adapter \
  --input-mode triple --task explain --max-samples 5

# 图像 + POI adapter
python scripts/inference/predict_lora_local.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/lora_poi_triple_preview.jsonl \
  --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
  --input-mode triple --task explain --max-samples 5
```

**全量预测：**

```bash
# 纯图像
python scripts/inference/predict_lora_local.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/lora_triple.jsonl \
  --adapter-path outputs/lora_adapters_cuda/final_adapter \
  --input-mode triple --task explain

# 图像 + POI
python scripts/inference/predict_lora_local.py \
  --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
  --output-jsonl outputs/predictions/lora_poi_triple.jsonl \
  --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
  --input-mode triple --task explain
```

推理参数说明：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--adapter-path` | 必填 | LoRA adapter 目录（`final_adapter/` 或 checkpoint） |
| `--base-model-id` | `Qwen/Qwen3-VL-8B-Instruct` | base 模型 ID 或本地路径 |
| `--input-mode` | `triple` | 图像模态组合 |
| `--task` | `explain` | 预测任务类型 |
| `--prompt` | `structured` | prompt 模块（需与训练数据的 prompt 一致） |
| `--max-new-tokens` | 1024 | 最大生成 token 数 |
| `--max-samples` | 0（全部） | 限制预测样本数 |
| `--overwrite` | false | 覆盖已有输出文件 |
| `--no-quantization` | false | BF16 加载（不量化） |
| `--merge-adapter` | false | 推理前将 LoRA 权重合并进 base model（更快，更耗显存） |

断点续跑：脚本自动跳过输出文件中已有的 `id`，无需额外参数。加 `--overwrite` 从头开始。

## Step 6 — 评估

预测完成后，在本地或服务器上运行评估：

```bash
python scripts/evaluation/evaluate_predictions.py \
  --predictions outputs/predictions/lora_poi_triple.jsonl \
  --alignment-csv dataset/sat_ntl_svi_aligned/streetview_satellite_ntl_alignment.csv
```
