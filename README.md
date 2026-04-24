# VLM Based Glasgow Inequality Analysis

本项目使用视觉语言模型研究格拉斯哥城市不平等与贫困分布，采用**感知→决策**两层架构：感知层从多模态图像中提取文本证据，决策层将文本证据回归为 SIMD 7 个领域分数。

---

## 系统架构

```
多模态图像（街景 + 遥感 + 夜光）
         │
         ▼
┌─────────────────────┐
│    感知层 Perception  │  Qwen3-VL-8B，为每个 datazone 生成文本证据
│  perception/        │  输出：outputs/perception/qwen3vl_8b_perception.jsonl
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    决策层 Decision    │  SBERT 编码 + 空间特征 + RidgeCV 回归
│  decision/          │  输出：outputs/decision/*/oof_predictions.jsonl
└─────────────────────┘
         │
         ▼
SIMD 7 域分数预测（Income / Employment / Health / Education / Access / Crime / Housing）
```

---

## 实验总览

| 路线 | 方法 | OOF pooled ρ | OOF pooled R² | 状态 |
|---|---|---|---|---|
| **Route C modality_sep_v0** | 分模态 SBERT + 空间滞后 + ego-gap + SAR lag + POI | **0.667** | **0.463** | 当前最优 |
| Route C ceiling_v0 | 单向量 SBERT + 空间滞后 + ego-gap + SAR lag + POI | 0.511 | 0.434 | 已复现 |
| Route C sbert_spatial_poi_v0 | 单向量 SBERT + 空间滞后 + POI | 0.507 | 0.448 | 已复现 |
| Route C sbert_minilm_v0 | 单向量 SBERT（无空间/POI/ego/SAR） | 0.443 | 0.277 | 消融基线 |
| Route A'（LightGBM） | 冻结 BGE-m3 + PCA + POI → 7×LGBM | 0.438 | 0.195 | 已复现 |
| Route A（MLP） | 冻结 BGE-m3 + attention pooling → 共享 trunk + 7 头 | fold-mean 0.517 | -0.143 | 有 fold 偏差 |
| Route B（对照） | Qwen3-8B CoT rationale + LoRA SFT | 0.123 | 0.083 | 需远程 GPU |

---

## 项目结构

```
perception/           # 感知层：VLM 多模态证据提取
  data/
  prompts/
  infer/
  README.md

decision/             # 决策层：文本嵌入 + 空间回归
  configs/
  data/
  models/
  train/
  infer/
  eval/
  README.md

dataset/
  SIMD/               # SIMD 原始数据
  glasgow_datazone/   # 格拉斯哥 datazone 矢量数据（.shp）
  streetview_dataset/ # 原始街景图像和 metadata.csv
  satellite_dataset/  # 遥感 patch、夜光 patch 和相关元数据
  TIFF/
    glasgow/          # 遥感原始 GeoTIFF
    glasgow_ntl/      # 夜光原始 GeoTIFF
  datazone_patches/   # 按 datazone centroid 提取的遥感/夜光 patch
  poi_dataset/        # OSM POI 汇总
  few_shot_examples/  # few-shot 示例样本
  lora_training_data/     # LoRA 训练数据（纯图像）
  lora_training_data_poi/ # LoRA 训练数据（含 POI）

outputs/
  perception/         # 感知层 JSONL 输出
  decision/           # 决策层 OOF 预测与评估
  predictions/        # 直接打分流水线预测 JSONL
  evaluation/         # 评估结果
  compare/            # 多模型对比报告与图表

scripts/
  data_prep/          # patch 提取、元数据构建、JSONL 生成
  evaluation/         # 预测评估与可视化脚本
  inference/          # API 与本地模型预测脚本
  training/           # LoRA 训练与训练数据构建脚本

src/glasgow_vlm/      # 核心库（config、data、metrics、splits、prompts/）
```

---

## 完整流程

详细文档见各子目录 README：
- 感知层：[perception/README.md](perception/README.md)
- 决策层：[decision/README.md](decision/README.md)

**快速上手：**

```bash
# 第一步：感知层推理（生成文本证据）
python perception/infer/perceive_local.py \
    --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
    --max-streetviews 20

# 第二步：构建决策层数据集
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v0.jsonl

# 第三步：5-fold CV 训练（Route C 主线）
python -m decision.train.cv_runner_caption \
    --config decision/configs/route_c_modality_sep_v0.yaml

# 第四步：评估
python scripts/evaluation/eval_decision_oof.py \
  --oof-jsonl outputs/decision/route_c/modality_sep_v0/oof_predictions.jsonl \
  --output-dir outputs/evaluation/decision_modality_sep_v0 \
  --no-spatial --no-pdf
```

---

## 环境配置

```bash
pip install torch torchvision transformers peft bitsandbytes accelerate \
            datasets tqdm pillow sentence-transformers scikit-learn joblib
```

感知层推理需要 CUDA GPU（4-bit 量化约需 ~10GB 显存；BF16 约需 ~20GB）。
