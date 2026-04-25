# 决策层（Decision Layer）

读取感知层 JSONL（每个 datazone 的多模态文本证据），回归出 SIMD 7 个领域分数：
`Income、Employment、Health、Education、Access、Crime、Housing`（整数 1–10，越高 = 越 deprived）。

---

## 路线总览

| 路线 | 方法 | OOF ρ | OOF R² | 状态 |
|---|---|---|---|---|
| **Route C + SegFormer SVF** | v1 + SegFormer SVF mean/std | **0.6682** | **0.4644** | 当前最优 |
| Route C + MIT PSP SVF | v1 + MIT PSP SVF | 0.6676 | 0.4636 | 同等水平 |
| Route C + both SVF | v1 + 两组 SVF mean/std | 0.6665 | 0.4627 | 同等水平 |
| Route C modality_sep_v1 + indicators | v1 + 17 个 VLM domain indicators | 0.6660 | 0.4605 | 零增益消融 |
| **Route C modality_sep_v1** | 分模态 SBERT + 空间滞后 + ego-gap + SAR lag + POI，RidgeCV | 0.6655 | 0.4600 | 强基线 |
| Route C stacking_v1 | SBERT→Ridge OOF + 结构化 POI/indicator/SVF→HGB，二层 Ridge 融合 | 0.6524 | 0.3558 | 退化于 baseline |
| Route C HGB | v1 原始高维特征 → HGB | 0.6319 | 0.2546 | 小样本高维下退化 |
| Route C HGB+PCA+indicators | PCA 降维 SBERT + indicators → HGB | 0.5905 | 0.1839 | 明显退化 |
| indicators-only | 仅 17 个 domain indicators → RidgeCV | 0.5099 | 0.2866 | 可解释但非新增信息源 |

> **评估口径**：所有 OOF ρ / R² 均按 **stacked + datazone-aggregated** 计算——先把同一 datazone 的多张图 OOF 预测取均值（n_dz = 746），再把 7 个域的 (target, pred) 全部 concat 成一条长向量，计算总 Spearman / R²。该口径与历史 `evaluate_domain_scores.py` 的 `overall` 行一致，是论文唯一对外口径。
> Route A 的 fold-mean 与 OOF 指标的偏差来自 fold 内漂移（各折校准不一致）；Route C 系列 fold-mean ≈ OOF 指标，评估最可靠。
> SVF 系列在 ρ / R² 上均略优于 baseline，但增益在 0.005 量级，属噪声。stacking_v1 在该口径下 R² 与 ρ 双双低于 baseline，说明二层融合损失了校准信息，**不再作为论文最优方案**。

---

## 目录结构

```
decision/
  configs/
    route_c_modality_sep_v0.yaml   # 主线实验配置
    diagnostics/                   # SAR / spatial / text-only 诊断配置
    svf/                           # SegFormer / MIT PSP SVF 消融配置
    route_c_ceiling_v0.yaml        # 单向量版上限配置（对比基准）
    route_c_caption_embed.yaml     # Route C 基础配置（消融用）
    route_a_bge.yaml               # Route A MLP
    route_a_lgbm.yaml              # Route A' LightGBM
    route_b_llm_sft.yaml           # Route B CoT+SFT
  data/
    parse_perception.py            # 解析 perception JSONL → sat/ntl/sv/poi_text + poi_counts
    normalize_evidence.py          # 四段文本规范化，拼接成 [SAT]…[NTL]…[SV]…[POI]
    build_dataset.py               # join SIMD_score.csv → dataset_v0.jsonl
    poi_features.py                # POI 计数 → 固定词表 → log1p + z-score 数值向量
    targets.py                     # logit 空间归一化 / 反归一化
    spatial_neighbors.py           # Queen 邻接关系（10m 缓冲）→ {datazone: [neighbors]}
  models/
    route_c/
      captioner.py                 # 单字符串模式：concat / templated；分模态模式：modality_sep
      encoder.py                   # BERT mean-pool / SBERT；encode_modality_sep()
      regressors.py                # RidgeCV / LassoCV / ElasticNetCV 工厂（含 clip pipeline）
    route_a/ …                     # MLP 编码器、pooling、回归头
    route_a_lgbm/ …                # SegmentPCA + MultiDomainLGBM
    route_b/ …                     # CoT prompt、rationalizer、SFT dataset
  train/
    route_c_train.py               # Route C 单 fold 训练（7 域独立 RidgeCV）
    cv_runner_caption.py           # Route C 5-fold IZ CV 驱动，支持 modality_sep
    cv_runner_svf.py               # Route C + SVF 特征块
    cv_runner_caption_lgbm.py      # Route C + HGB / PCA-HGB 实验
    cv_runner.py                   # Route A CV 驱动
    cv_runner_lgbm.py              # Route A' CV 驱动
  experiments/
    manifests/
      paper_v1.yaml                # 论文主表：baseline / indicators / SVF / HGB / stacking
      diagnostics_v1.yaml          # 诊断矩阵：SAR-only / no-SAR / residual / text-only
    run_manifest.py                # 按 manifest 复现实验或重建比较表
  infer/
    route_b_predict.py             # Route B 推理
  eval/
    oof.py                         # 标准 OOF 读取、聚合、Spearman 计算
    compare.py                     # 任意 OOF run 的统一对比 CLI
    hypothesis_test.py             # 路线间 Wilcoxon + bootstrap Spearman CI
```

输出写入 `outputs/decision/route_c/<实验名>/`。

---

## 可诊断、可消融、可复现的实验框架

当前实验统一围绕 **OOF prediction** 组织：每个 run 必须输出同一套文件，至少包括：

```
outputs/decision/route_c/<run_name>/
  oof_predictions.jsonl
  cv_summary.json
```

论文主表统一使用 **stacked + datazone-aggregated** 口径：先按 datazone 对重复 OOF 行做均值聚合（n_dz = 746），再把 7 个域的 (target, pred) 全部 concat 成一条长向量，计算单一 Spearman / R²。这是项目唯一对外口径，`decision.eval.compare` 与 `decision/eval/oof.py` 已锁死该实现，无需配置项。

可选诊断文件：

```
feature_slices.json        # 特征块列范围，用于系数/消融诊断
fold*/fold_meta.json       # fold 级指标
fold*/reg_<domain>.joblib  # RidgeCV pipeline
```

### 论文主表复现

`decision/experiments/manifests/paper_v1.yaml` 固定了当前论文主表顺序：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/paper_v1.yaml
```

如需重跑某个实验：

```bash
python -m decision.experiments.run_manifest \
  --manifest decision/experiments/manifests/paper_v1.yaml \
  --only stacking_v1
```

只重建比较表：

```bash
python -m decision.experiments.run_manifest \
  --manifest decision/experiments/manifests/paper_v1.yaml \
  --only compare
```

比较结果默认写入：

```
outputs/evaluation/paper_v1_route_c_comparison.csv
```

### 诊断矩阵

`decision/experiments/manifests/diagnostics_v1.yaml` 用于拆分信号来源：

| run | 目的 |
|---|---|
| `full_v1` | 当前 RidgeCV 强基线 |
| `sar_only_v1` | 仅使用 train-neighbour SIMD 均值，估计纯空间目标滞后强度 |
| `no_sar_v1` | 去掉 SAR target lag，估计非目标空间插值贡献 |
| `residual_after_sar_v1` | 先拟合 SAR，再用非 SAR Route C 特征预测剩余残差 |
| `no_spatial_no_sar_v1` | 去掉 spatial lag / ego-gap / SAR，保留 SBERT+POI |
| `text_only_v1` | 仅保留 modality-separated SBERT |
| `indicators_only_v2` | 仅保留 17 个 VLM domain indicators |
| `stacking_v1` | 当前最优融合方案 |

当前诊断结果（stacked + datazone-aggregated）：

| run | ρ | R² | Δρ vs full_v1 |
|---|---:|---:|---:|
| `full_v1` | 0.6655 | 0.4600 | 0.0000 |
| `sar_only_v1` | 0.5528 | 0.3156 | -0.1127 |
| `no_sar_v1` | 0.6403 | 0.4319 | -0.0251 |
| `residual_after_sar_v1` | 0.5970 | 0.3613 | -0.0684 |
| `no_spatial_no_sar_v1` | 0.6456 | 0.4378 | -0.0199 |
| `text_only_v1` | 0.6454 | 0.4378 | -0.0201 |
| `indicators_only_v2` | 0.5099 | 0.2866 | -0.1556 |
| `stacking_v1` | 0.6524 | 0.3558 | -0.0131 |

运行诊断配置：

```bash
python -m decision.experiments.run_manifest \
  --manifest decision/experiments/manifests/diagnostics_v1.yaml
```

比较已有诊断 run：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/diagnostics_v1.yaml
```

### 当前结论定位

`modality_sep_v1` 是论文 baseline：`ρ = 0.6655`，`R² = 0.4600`。SVF 三种变体均带来 ≤ 0.005 的微弱 ρ / R² 增益，落在噪声区间内；最稳定的是 SegFormer SVF（ρ=0.6682, R²=0.4644），可作为论文上限报告，但不应解读为新信息源。

`stacking_v1` 在新口径下 ρ 和 R² 双双低于 baseline（Δρ = -0.0131，ΔR² = -0.1042）：二层 HGB→Ridge 融合在 stacked 评估下损失了数值校准，且没有保留排序优势——不再作为最优方案保留。

SAR 诊断链路解释 full baseline 的来源：`sar_only_v1` 衡量空间标签滞后本身，`no_sar_v1` 衡量视觉/POI/空间嵌入在不看邻居标签时的能力，`residual_after_sar_v1` 衡量视觉特征能否解释 SAR 剩余误差。三者必须和 `full_v1` 一起报告，避免把空间插值能力误写成纯视觉理解能力。

indicators 拼接保留为消融：17 个 VLM domain indicators 的语义已与 SBERT/SAR 主信号高度重叠，直接拼入 RidgeCV 几乎无增益。

---

## Route C modality_sep_v0 完整流程（主线）

### 架构概览

```
perception JSONL
    │
    ▼
[1] dataset_v0.jsonl         4 段文本（sat / ntl / sv / poi_text）+ SIMD targets
    │
    ▼
[2] 分模态 SBERT 编码         4 × 384-dim → 拼接 1536-dim 嵌入向量
    │  ├─ SAT: 全文 encode（~30 tokens，无截断）
    │  ├─ NTL: 全文 encode（~30 tokens，无截断）
    │  ├─ SV:  per-phrase encode → mean-pool（每短语 ≤ 8 词，无截断）
    │  └─ POI: 全文 encode（~15 tokens，无截断）
    │
    ▼
[3] Datazone 级聚合（dz_agg）  多张图 → 每 DZ 唯一嵌入（mean-pool 修复重复 DZ bug）
    │
    ▼
[3.5] SVF 特征注入（可选，仅 cv_runner_svf）
    │  ├─ 语义分割类别比例（SegFormer-B2 / MIT PSP，每 DZ 街景均值/标准差）
    │  ├─ 全局缺失（无街景 DZ）置 NaN
    │  └─ per-fold 用 train 集列均值插补（无 val 泄漏）
    │  → 拼接到 X_text 末尾，spatial lag / ego-gap 自动覆盖 SVF 列
    │
    ▼
[4] 空间特征拼接（per fold）
    │  ├─ 空间滞后：concat([self, neighbor_mean]) → 3072-dim（含 SVF 时维度相应增加）
    │  ├─ Ego-gap：改为 concat([self, self−neighbor_mean])（避免秩亏）
    │  ├─ POI 数值向量（PoiFitter，per fold 拟合）
    │  └─ SAR 目标滞后：mean(y_train_neighbors)，7-dim（无 val 标签泄漏）
    │
    ▼
[5] RidgeCV × 7 域              StandardScaler → clip(±30) → RidgeCV(cv=5, alphas 25档)
    │
    ▼
OOF predictions + cv_summary.json
```

---

### 第一步：构建数据集

```bash
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v0.jsonl
```

**输出每行**（`dataset_v0.jsonl`）：

```json
{
  "datazone": "S01006514",
  "sat": "dense terraced housing; narrow road grid; limited green space",
  "ntl": "dim uniform glow; patchy at edges",
  "sv":  "cracked pavement; closed shopfront; graffiti on walls; ...",
  "poi_text": "42 POIs total: amenity×18, shop×9, ...",
  "poi_counts": {"amenity": 18, "shop": 9, ...},
  "targets":     {"Income": 1.23, ...},
  "targets_raw": {"Income": 7, ...}
}
```

`sat / ntl / sv` 均为以 `"; "` 分隔的短语列表字符串，来自感知层的 evidence 字段。

---

### 第二步：5-fold IZ CV 训练

```bash
python -m decision.train.cv_runner_caption \
    --config decision/configs/route_c_modality_sep_v1.yaml
```

**配置文件（`route_c_modality_sep_v1.yaml`）：**

```yaml
dataset: outputs/decision/dataset_v1.jsonl
out_dir: outputs/decision/route_c/modality_sep_v1

caption:
  mode: modality_sep        # 分模态编码，非 concat

encoder:
  backend: sbert
  name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32
  max_length: 256

cv:
  n_splits: 5

train:
  regressor: ridge_cv
  use_poi_vec: true         # POI 数值向量（log1p + z-score）
  use_spatial_lag: true     # 拼接邻居均值嵌入
  use_ego_gap: true         # 替换邻居均值 → [self, self−nbr]
  use_sar_lag: true         # 拼接 mean(y_train_neighbors)
  use_dz_agg: true          # 多图 → DZ 级 mean-pool
  seed: 42
```

**输出：**
- `caption_cache.pt` — `(datazones, X_1536, meta)`，重跑自动复用
- `fold{k}/reg_{domain}.joblib` — 每折 7 个已拟合 pipeline
- `fold{k}/poi_fitter.json` — per-fold POI 归一化参数
- `fold{k}/fold_meta.json` — 各折 val Spearman
- `oof_predictions.jsonl`、`cv_summary.json`

---

### 各设计决策详解

#### 2a. 分模态独立编码（核心改动）

**问题**：原有 `concat` 模式将 4 段文本拼接成一个字符串再送入 SBERT。  
SBERT（all-MiniLM-L6-v2）硬限 **max_seq_length = 256 tokens**，而实际拼接后：

```
[SAT] ~30 tokens  +  [NTL] ~30 tokens  +  [SV] ~350 tokens  +  [POI] ~15 tokens
                                                   ↑ 超出限制，后半段被静默截断
```

统计 `dataset_v0.jsonl`（856 样本）：
- **830/856（97%）**的样本总词数超过 197 词（约 256 tokens）
- SV 段平均 268 词，SBERT 仅能编码其中约 54%
- **POI 文本在所有样本中全部被截断**（排在 SV 之后，永远不进入编码器）

**解法**：4 个模态各自独立调用一次 SBERT，拼接输出：

```
encode(sat)  →  384-dim   (SAT block)
encode(ntl)  →  384-dim   (NTL block)
per-phrase mean-pool(sv) →  384-dim   (SV block)
encode(poi)  →  384-dim   (POI block)
────────────────────────────────────────
concat → 1536-dim
```

SV 的 per-phrase mean-pool：将 `sv` 按 `"; "` 拆分成单条短语（每条 ≤ 8 词 ≈ 10 tokens），全部短语一次批量编码，再按 datazone 分组均值池化。每条短语都在 SBERT token 限制内，不存在截断。

代码位置：`decision/models/route_c/captioner.py:build_captions_modality_sep()`、`decision/models/route_c/encoder.py:encode_modality_sep()`

#### 2b. Datazone 级聚合（dz_agg）

`dataset_v0.jsonl` 中 746 个 DZ 对应 856 行（27 个 DZ 有多张图，最多重复 18 次）。原始 `dz_index` 构建用 `{dz: i for i, dz in enumerate(...)}` ——**后出现的 DZ 覆盖先出现的**，导致这 27 个 DZ 的早期图像被静默丢弃。

`use_dz_agg=True` 时，在构建 `dz_index` 前先对同一 DZ 的所有行取均值池化，得到每 DZ 唯一的 1536-dim 代表向量：

```python
# cv_runner_caption.py: _aggregate_X_by_dz()
dz_rows: dict[str, list[int]] = defaultdict(list)
for i, s in enumerate(samples):
    dz_rows[s["datazone"]].append(i)
X_agg = np.stack([X[idxs].mean(axis=0) for idxs in dz_rows.values()])
```

#### 2c. 空间滞后（Spatial Lag）

剥夺程度具有强烈的空间自相关——格拉斯哥 East End 等高贫困区在地理上连片分布。引入邻域嵌入均值，赋予线性回归器感知"周边环境"的能力。

**邻接关系构建**（`decision/data/spatial_neighbors.py`）：

1. shapefile 投影至 OSGB36（EPSG:27700，单位：米）
2. 每个多边形向外 buffer 10m（消除浮点裂缝）
3. sjoin(buffered, original, predicate="intersects")，排除自连接
4. 输出 `{datazone: [neighbour_dz, ...]}` — **Queen 邻接**（共享边或顶点均视为毗邻）

**在 fold 内的拼接**：

```python
# route_c_train.py: _neighbor_features()
for i, s in enumerate(samples):
    nbr_dzs = [n for n in neighbors[s["datazone"]] if n in dz_index]
    lag[i] = X_all[nbr_idx].mean(axis=0)  # 若无邻居 → 回退到自身 embedding

X_tr = concat([X_tr, lag_tr], axis=1)    # (n, 3072)
```

> **无泄漏**：`lag` 使用的是全量 `X_all` 的**嵌入向量**，不涉及任何 val 样本的 SIMD 标签。

#### 2d. Ego-gap（对比特征）

普通空间滞后拼接 `[self, neighbor_mean]`。但同时启用 ego-gap 后若改为 `[self, neighbor_mean, self−neighbor_mean]`，会出现**完美共线性**（第 3 项 = 第 1 项 − 第 2 项），使特征矩阵秩亏，引发 RidgeCV 内部 GCV 数值溢出。

`use_ego_gap=True` 时改为：

```python
# 用 [self, self−nbr] 替代 [self, nbr, self−nbr]
# 信息等价（两组向量张成同一列空间），但无共线性
X_tr = concat([X_tr, X_tr - lag_tr], axis=1)   # (n, 3072)
```

`self−neighbor_mean` 捕捉 datazone 相对于周边区域的**对比效应**——即使绝对嵌入相似，若自身与邻居之间有系统性差距，ego-gap 也能放大这一差异信号。

#### 2e. SAR 目标滞后（SAR Lag）

空间自回归（Spatial AutoRegressive）中，目标变量的邻域均值是最强预测因子之一。`use_sar_lag=True` 时额外拼接 7-dim 特征：`mean(y_train_neighbors_per_domain)`。

**无泄漏设计**：
- val 样本的邻居只查 train 集标签（`dz_to_y` 仅从 `train_samples` 构建）
- 若某 val DZ 的所有邻居恰好都在 val fold，则回退到训练集全局均值

```python
# route_c_train.py: _target_lag_features()
dz_to_y = {dz: mean(y_rows) for dz, y_rows in train_accumulator}
for i, s in enumerate(samples):
    train_nbrs = [n for n in neighbors[s["datazone"]] if n in dz_to_y]
    out[i] = mean([dz_to_y[n] for n in train_nbrs])  # or global_mean
```

#### 2f. POI 数值向量

尽管 POI 文本已经由 `encode(poi)` 得到语义嵌入，`use_poi_vec=True` 额外追加数值 POI 向量，两者互补：
- **语义嵌入**（384-dim）：捕捉 POI 类型名称的语义（"amenity"、"healthcare" 等词义）
- **数值向量**（固定词表 log1p + z-score）：捕捉各类 POI 的**数量级**差异，不依赖模型对 POI 计数文本的解析

`PoiFitter` 在每 fold 的 train 集上拟合，避免 val 集 POI 频率分布泄漏到 z-score 计算。

#### 2g. SVF 特征（可选）

街景图像的语义分割可输出每张图的类别像素比例（建筑、植被、道路、天空等），按 datazone 聚合后作为低维结构化特征拼到 X_text。两种来源可单独或合并使用：

| 特征源 | 类数 | parquet 路径 |
|---|---|---|
| SegFormer-B2 (ADE20K) | 15 类 | `outputs/perception/svf/datazone_svf_segformer.parquet` |
| MIT PSPNet (ADE20K) | 15 类 | `outputs/perception/svf/datazone_svf_mitpsp.parquet` |

每行字段：`datazone, <class>_mean, <class>_std`（按街景图取 DZ 内均值/标准差）。配置 `svf.columns: [mean]` 取 15-dim 比例向量；`[mean, std]` 取 30-dim。

**注入位置**：在 `dz_agg` 之后、空间特征拼接之前 concat 到 X_text 末尾。后续 spatial lag / ego-gap / SAR 自动作用于全部列（包括 SVF），即 SVF 也享有空间溢出。

**缺失处理**（无街景 DZ）：
- 全局：缺失 DZ 该列为 NaN
- per-fold：用该 fold 训练集 DZ 的列均值插补，避免用 val 集分布信息

**调用入口**（不走主线 `cv_runner_caption.py`）：

```bash
python -m decision.train.cv_runner_svf \
  --config decision/configs/svf/route_c_modality_sep_svf_segformer_v0.yaml
```

代码位置：[train/cv_runner_svf.py](train/cv_runner_svf.py) `load_svf_matrix()` / `_impute_fold()`。

> 当前结论：三种 SVF 变体相对 baseline 增益 ≤ 0.005（落在噪声区间），保留为可选消融，论文 baseline 仍是 `modality_sep_v1`。

#### 2h. RidgeCV 数值稳定性

特征维度 3072-dim（1536 × 2 + 7 SAR + poi_dim）远超 n_train ≈ 600，高维低方差列（如 ego-gap 中接近零的维度）经 StandardScaler 放大后，RidgeCV 的 GCV / SVD 内部运算会出现数值溢出。

Pipeline 加了 clip 步骤：

```python
# regressors.py
def _clip30(X): return np.clip(X, -30, 30)

Pipeline([
    ("scaler", StandardScaler()),
    ("clip",   FunctionTransformer(_clip30)),   # 防止极端值传入 RidgeCV
    ("reg",    RidgeCV(alphas=np.logspace(-2, 6, 25), cv=5)),
])
```

- `cv=5`：避免 GCV 模式（`cv=None`）在 p≥n 时的数值不稳定
- `alphas=logspace(-2, 6, 25)`：宽范围覆盖高维场景下所需的强正则化
- `_clip30` 为模块级函数（非 lambda）：保证 `joblib.dump` 可序列化

---

### 第三步：评估

```bash
python evaluation/eval_decision_oof.py \
  --oof-jsonl outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl \
  --output-dir outputs/evaluation/decision_modality_sep_v1 \
  --no-spatial --no-pdf
```

**内部步骤：**
1. 按 datazone 均值聚合 OOF 预测 → `pred_aggregated.csv`
2. 调用 `evaluate_domain_scores.py`：RMSE / MAE / Pearson / Spearman / R² / QWK，各带 95% bootstrap CI
3. 调用 `visualize_domain_scores.py`：散点图、误差柱状图、Glasgow 交互地图

---

## 与 ceiling_v0 对比

`sbert_ceiling_v0`（OOF ρ = 0.6458, R² = 0.4338）与 `modality_sep_v0`（OOF ρ = 0.6673, R² = 0.4625）的唯一区别是编码方式：

| 项目 | ceiling_v0 | modality_sep_v0 |
|---|---|---|
| 编码模式 | `concat` | `modality_sep` |
| 嵌入维度 | 384 | 1536 |
| SV 截断 | ~46% 内容被截断 | 无截断（per-phrase） |
| POI 语义编码 | **全部被截断** | 独立 384-dim |
| 其余设置 | spatial + ego + sar + poi_vec + dz_agg | 同左 |

快速对比两次实验：

```bash
python3 -c "
import json
for name, path in [
    ('ceiling_v0',   'outputs/decision/route_c/sbert_ceiling_v0/cv_summary.json'),
    ('modality_sep', 'outputs/decision/route_c/modality_sep_v0/cv_summary.json'),
]:
    s = json.load(open(path))
    per = s['folds'][0]['val_per_domain_spearman']
    print(f'\n{name}  OOF={s[\"oof_mean_spearman\"]:.4f}')
    for d, v in sorted(per.items()):
        print(f'  {d:<12} {v:.4f}')
"
```

理论上 Income 和 Crime 两域涨幅最大（原先 POI 语义被完全丢弃，而 POI 对商业活力 / 服务可及性有直接信号）。

---

## 历史路线

Route C v0 / ceiling、Route A、Route A'、Route B 和早期假设检验代码已移动到 [legacy/](../legacy/readme.md)。当前论文复现不再调用这些入口；如需恢复历史 baseline，请先阅读 legacy README 并检查旧 import 路径。

---

## 数据 schema

- **输入**：`dataset_v0.jsonl`，每行一个 datazone，含 sat / ntl / sv / poi_text / poi_counts / targets_raw
- **训练目标**：`y = logit(clip((score−1)/9, 1e-3, 1−1e-3))`；推理时逆变换回 1–10
- **OOF 输出**：
  ```json
  {"datazone": "S01...", "fold": 0,
   "prediction_json": {"Income": 5.3, "Employment": 4.8, ...},
   "target_raw":      {"Income": 7,   "Employment": 6, ...}}
  ```
- **领域顺序**：`Income, Employment, Health, Education, Access, Crime, Housing`（见 [data/targets.py](data/targets.py)）

## 关键复用

| 需求 | 位置 |
|---|---|
| IZ 分组 CV | [src/glasgow_vlm/splits.py](../src/glasgow_vlm/splits.py) `group_kfold_by_iz` |
| Spearman 验证指标 | [src/glasgow_vlm/metrics.py](../src/glasgow_vlm/metrics.py) |
| 全量评估（含 bootstrap CI） | [evaluation/evaluate_domain_scores.py](../evaluation/evaluate_domain_scores.py) |
| 感知层输出 → 决策层数据集 | [data/build_dataset.py](data/build_dataset.py) |
