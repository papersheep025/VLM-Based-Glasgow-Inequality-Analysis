# VLM Based Glasgow Inequality Analysis

本项目研究格拉斯哥城市不平等与贫困分布，当前采用 **感知 → 决策** 两层架构：

1. **感知层 Perception**：从遥感、夜光、街景、POI 中提取可见文本证据，并可额外生成街景语义分割 SVF 特征。
2. **决策层 Decision**：把感知层证据编码成特征，结合 POI、空间邻接和 SAR target lag，回归 SIMD 7 个领域分数。

感知层不直接判断贫困程度；SIMD 预测只在决策层完成。

---

## 当前主线

```
streetview + satellite + nightlight + POI
        │
        ▼
perception/infer/perceive_local.py
Qwen3-VL-8B → evidence phrases + domain_indicators
        │
        ▼
decision/data/build_dataset.py
dataset_v1.jsonl
        │
        ▼
Route C
modality-separated SBERT + POI + spatial lag + ego-gap + SAR lag
        │
        ├── RidgeCV baseline
        ├── indicators 消融
        ├── SegFormer / MIT PSP SVF 消融
        ├── HGB / PCA-HGB 消融
        └── stacking 消融
        │
        ▼
outputs/decision/route_c/*/oof_predictions.jsonl
```

可选 SVF 分支：

```
raw streetview images
        │
        ▼
perception/segmentation/{segformer_infer.py, mitpsp_infer.py}
image-level SVF parquet
        │
        ▼
perception/segmentation/aggregate.py
datazone-level SVF parquet
        │
        ▼
decision/train/cv_runner_svf.py
```

---

## 实验总览

当前论文表以 `decision/experiments/manifests/paper_v1.yaml` 为准。

| 路线 | 方法 | OOF rho | OOF R2 | 状态 |
|---|---|---:|---:|---|
| **Route C + SegFormer SVF** | v1 + SegFormer SVF mean/std | **0.6682** | **0.4644** | 当前最高 |
| Route C + MIT PSP SVF | v1 + MIT PSP SVF | 0.6676 | 0.4636 | 同等水平 |
| Route C + both SVF | v1 + 两组 SVF mean/std | 0.6665 | 0.4627 | 同等水平 |
| Route C modality_sep_v1 + indicators | v1 + 17 个 VLM domain indicators | 0.6660 | 0.4605 | 零增益消融 |
| **Route C modality_sep_v1** | 分模态 SBERT + 空间滞后 + ego-gap + SAR lag + POI，RidgeCV | 0.6655 | 0.4600 | 论文 baseline |
| Route C stacking_v1 | SBERT-Ridge OOF + 结构化 HGB，二层 Ridge | 0.6524 | 0.3558 | 退化于 baseline |
| Route C HGB | v1 高维特征 → HGB | 0.6319 | 0.2546 | 小样本高维下退化 |
| Route C HGB+PCA+indicators | PCA 降维 SBERT + indicators → HGB | 0.5905 | 0.1839 | 明显退化 |
| indicators-only | 仅 17 个 domain indicators → RidgeCV | 0.5099 | 0.2866 | 可解释但非新增信息源 |

> 评估口径：所有 OOF rho / R2 均按 **stacked + datazone-aggregated** 计算。先把同一 datazone 的多条 OOF 预测取均值，再把 7 个领域的 `(target, pred)` concat 成一条长向量，计算单一 Spearman / R2。该口径由 `decision/eval/oof.py` 和 `decision/eval/compare.py` 统一实现。

SVF 的提升约在 0.005 rho / R2 量级，适合作为上限或稳健性消融报告，不应被解读为强新信息源。`modality_sep_v1` 是更稳妥的论文 baseline。

---

## 快速复现

### 1. 生成感知层 evidence

```bash
python -m perception.data.build_patch_poi

python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception_v2.jsonl \
  --max-streetviews 20
```

如已有 evidence，只需补齐 `domain_indicators`：

```bash
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception_v2.jsonl \
  --patch-indicators-only
```

### 2. 构建决策层数据集

```bash
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception_v2.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v1.jsonl
```

### 3. 跑论文主表

重跑单个实验：

```bash
python -m decision.experiments.run_manifest \
  --manifest decision/experiments/manifests/paper_v1.yaml \
  --only modality_sep_v1
```

比较已有 OOF 结果：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/paper_v1.yaml
```

结果默认写入：

```
outputs/evaluation/paper_v1_route_c_comparison.csv
```

### 4. 可选：生成 SVF 特征

```bash
python -m perception.segmentation.segformer_infer \
  --metadata dataset/streetview_metadata.csv \
  --raw-dir dataset/streetview_dataset_raw \
  --output outputs/perception/svf/image_level_segformer.parquet \
  --batch-size 16 \
  --device cuda:0

python -m perception.segmentation.aggregate \
  --image-level outputs/perception/svf/image_level_segformer.parquet \
  --output outputs/perception/svf/datazone_svf_segformer.parquet

python -m decision.train.cv_runner_svf \
  --config decision/configs/svf/route_c_modality_sep_svf_segformer_v0.yaml
```

MIT PSP 分支见 [perception/README.md](perception/README.md)。

---

## 项目结构

```
perception/                 # 当前感知层
  data/                     # POI 宽表构建
  prompts/                  # Qwen3-VL evidence + domain_indicators prompt
  infer/                    # 本地 VLM 感知推理
  segmentation/             # SegFormer / MIT PSP SVF 分支

decision/                   # 当前决策层 + 历史路线
  configs/                  # Route C 主线、SVF、诊断、历史 Route A/B 配置
  data/                     # perception JSONL → dataset_v1，POI/spatial/target 工具
  models/route_c/           # 当前 Route C caption/encoder/regressor
  train/                    # Route C CV、SVF、HGB；另含历史 Route A/A'
  eval/                     # 当前 OOF compare + 历史评估封装
  experiments/              # paper_v1 / diagnostics_v1 manifest

data_preparation/           # 数据准备分支：抓取、POI、patch、地图预览

evaluation/                 # 评估与可视化分支：主表消融、诊断、图表

src/glasgow_vlm/
  metrics.py                # 当前仍被 OOF/诊断脚本使用
  splits.py                 # 当前仍被 IZ-grouped CV 使用

legacy/                     # 历史/备用分支：Route A/B、direct VLM scoring、LoRA
```

---

## 代码扫描结果

这次扫描按 `paper_v1.yaml`、`diagnostics_v1.yaml`、当前 README 链路和 Python 引用关系做了分类。下面的“未用”指 **不在当前论文复现链路中调用**，不是说代码一定没有历史价值。

### 当前链路会用

| 模块 | 用途 |
|---|---|
| `perception/data/build_patch_poi.py` | 生成 `patch_poi.csv` |
| `perception/infer/perceive_local.py` | Qwen3-VL 感知层 evidence / indicators |
| `perception/prompts/perception.py` | 当前感知层 prompt |
| `perception/segmentation/*` | SVF 消融特征生成 |
| `decision/data/*` | 解析感知输出、规范化文本、POI/空间/target 工具 |
| `decision/models/route_c/*` | Route C 文本编码与回归器 |
| `decision/train/route_c_train.py` | Route C 单 fold 训练核心 |
| `decision/train/cv_runner_caption.py` | RidgeCV 主线和诊断消融 |
| `decision/train/cv_runner_svf.py` | Route C + SVF |
| `decision/train/cv_runner_caption_lgbm.py` | Route C HGB / PCA-HGB 消融 |
| `decision/models/route_a_lgbm/model.py` | 被 Route C HGB 消融复用 |
| `decision/eval/oof.py`, `decision/eval/compare.py` | 当前唯一对外 OOF 比较口径 |
| `decision/experiments/*` | 论文主表与诊断矩阵复现 |
| `evaluation/indicator_only_cv.py` | indicators-only baseline |
| `evaluation/indicator_diagnostics.py` | indicator 相关性诊断 |
| `evaluation/spatial_residual_diagnostics.py` | SAR-only / residual-after-SAR 诊断 |
| `evaluation/stacking_v1.py` | stacking_v1 消融 |
| `src/glasgow_vlm/metrics.py`, `src/glasgow_vlm/splits.py` | Spearman 与 IZ-grouped CV |

### 数据准备分支

| 模块 | 状态 |
|---|---|
| `data_preparation/*` | 数据抓取、POI、patch、地图可视化的一次性构建脚本；正常复现实验不需要反复运行 |

### 评估与可视化分支

| 模块 | 状态 |
|---|---|
| `evaluation/eval_decision_oof.py` | 旧式详细评估入口，仍可生成图和 bootstrap CI，但论文主表以 `decision.eval.compare` 为准 |
| `evaluation/evaluate_domain_scores.py` | 详细 domain-level 指标工具，非当前主表入口 |
| `evaluation/visualize_domain_scores.py` | 地图/图表生成工具，非当前主表入口 |
| `evaluation/compare_svf_ablation.py` | 早期 SVF 对比脚本，基本被 manifest + `decision.eval.compare` 取代 |

### 当前主线不调用的历史/备用代码

| 模块 | 说明 |
|---|---|
| `legacy/scripts/inference/*` | 早期直接让 VLM 输出 SIMD 分数的 API/LoRA 推理流水线；当前改为“感知证据 → 决策回归” |
| `legacy/scripts/training/*` | 早期 Qwen3-VL LoRA 直打分训练；当前主线不要求训练 LoRA，`perceive_local.py --adapter-path` 仅作为可选能力 |
| `legacy/src/glasgow_vlm/{config.py,data.py,prompts/}` | 主要服务早期直接打分和 LoRA 数据构建 |
| `legacy/decision/models/route_a/*` | Route A MLP 历史基线 |
| `legacy/decision/train/{route_a_train.py,dataset.py,losses.py}` | Route A 训练链路 |
| `legacy/decision/train/{cv_runner_lgbm.py,route_a_lgbm_train.py}` | Route A' LightGBM 历史训练链路；注意 `decision/models/route_a_lgbm/model.py` 仍被 Route C HGB 复用 |
| `legacy/decision/models/route_b/*`, `legacy/decision/infer/route_b_predict.py`, `legacy/decision/utils/remote.py` | Route B LLM-as-regressor / CoT / SFT 历史对照 |
| `legacy/decision/eval/*` | Route A/B/C 早期评估与显著性测试工具，不在 `paper_v1` 主表链路 |
| `legacy/decision/configs/*` | 历史 Route A/B 和 Route C v0/中间消融配置 |
| `legacy/scripts/aggregate_domain_scores.py` | 早期 `outputs/predictions/*.jsonl` 聚合脚本，不处理当前 OOF 主线 |

---

## 数据集

研究区为格拉斯哥市，以 **746 个 datazone** 为基本分析单元，ground truth 来自 SIMD 2020。

| 数据 | 规模 | 默认位置 |
|---|---:|---|
| SIMD 2020 分数 | 746 datazones x 7 域 | `dataset/SIMD/SIMD_score.csv` |
| Datazone 边界 | 746 polygons | `dataset/glasgow_datazone/glasgow_datazone.shp` |
| 街景图像 | 16,075 张，336x336 | `dataset/streetview_dataset/<datazone>/*.jpg` |
| 街景原图 | 16,075 张，640x640 | `dataset/streetview_dataset_raw/<datazone>/*.jpg` |
| 遥感 patch | 856 张，384x384 | `dataset/satellite_dataset/satellite/` |
| 夜光 patch | 856 张，256x256 | `dataset/satellite_dataset/ntl/` |
| POI | 14,558 条 | `dataset/poi_dataset/datazone_poi.csv`, `patch_poi.csv` |

satellite / nightlight patch 数大于 datazone 数，因为部分较大 datazone 包含 extra patches。CV 训练时以 datazone 为聚合单位。

---

## 环境配置

基础环境：

```bash
pip install -r requirements.txt
pip install transformers peft bitsandbytes accelerate sentence-transformers pyarrow pyyaml joblib
```

CUDA 推理/训练环境：

```bash
pip install -r requirements_cuda.txt
```

感知层 Qwen3-VL 推理需要 CUDA GPU：4-bit 量化约 10GB 显存，BF16 约 20GB 显存。Route C RidgeCV 可在 CPU 上运行；SVF 分割建议使用 CUDA。

---

## 详细文档

- 感知层：[perception/README.md](perception/README.md)
- 决策层：[decision/README.md](decision/README.md)
