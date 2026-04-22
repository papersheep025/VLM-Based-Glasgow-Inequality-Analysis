# 决策层（Decision Layer）

## 远程主机准备（Route B）

### 第零步：下载 Qwen3-8B 模型

在远程主机项目根目录执行（需 ~16GB 磁盘空间）：

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

模型保存到 `models/Qwen3-8B/`，YAML 配置已指向该路径，无需手动修改。

### 安装依赖

```bash
pip install vllm peft datasets accelerate transformers torch
```

vLLM 版本建议 ≥ 0.4，transformers ≥ 4.51（Qwen3 需要）。

---

读取感知层 JSONL（每个 datazone 的多模态文本证据），回归出 SIMD 7 个领域分数：
`Income、Employment、Health、Education、Access、Crime、Housing`（整数 1–10，越高 = 越 deprived）。

## 两条路线

| 路线 | 方法 | 状态 |
|---|---|---|
| **Route A（MLP）** | 冻结文本编码器（BGE-m3）+ POI 结构化特征 → 共享 trunk → 7 个回归头，IZ 分组 5-fold CV | 就绪 |
| **Route A'（LightGBM）** | 冻结 BGE-m3 + per-fold PCA（每段 1024→32） + POI → 7 个独立 LGBMRegressor，IZ 分组 5-fold CV | 就绪 |
| **Route B（对照）** | LLM-as-regressor + CoT，通过教师 rationale 生成 + PEFT LoRA SFT 微调 | 就绪（需远程 GPU） |

## 目录结构

```
decision/
  configs/
    route_a_bge.yaml          # Route A 训练超参
    route_b_llm_sft.yaml      # Route B 三阶段配置
  data/
    parse_perception.py       # 解析 perception JSONL，拆分证据段 + POI 计数
    normalize_evidence.py     # 四段文本规范化 [SAT] [NTL] [SV] [POI_TEXT]
    poi_features.py           # POI 计数 → 固定词表 → log1p + z-score
    build_dataset.py          # join SIMD_score.csv → dataset_v0.jsonl
    targets.py                # logit 空间归一化 / 反归一化
  models/
    route_a/
      encoder.py              # 冻结 FrozenTextEncoder（BGE-m3，CLS池化+L2归一）
      pooling.py              # 街景短语 attention pooling（长度无关）
      heads.py                # PoiMLP + SharedTrunk + 7×DomainHead
    route_b/
      prompt_template.py      # evidence → 3-step CoT 提示（含 cite-only 约束）
      rationalizer.py         # 教师 CoT 反向生成 + 校验（分数匹配 + 引用校验）
      sft_dataset.py          # rationale → SFT 训练 JSONL
      train_sft.py            # PEFT LoRA SFT（远程 GPU，依赖 transformers/peft）
  train/
    losses.py                 # Huber + (1−Pearson) + SoftSpearman，MTL 不确定性加权
    dataset.py                # TensorDataset 构建（4段编码 + POI + logit target）
    route_a_train.py          # 单 fold 训练（早停 + grad clip）
    cv_runner.py              # 5-fold IZ CV 驱动，聚合 OOF 预测
  infer/
    route_b_predict.py        # Route B 推理，JSON 解析失败回退到训练集均值
  eval/
    run_eval.py               # OOF → CSV → 调用 evaluate_domain_scores.py
    ablations.py              # 逐段置零（SAT/NTL/SV/POI_TEXT/v_poi），报告 Spearman 下降
  utils/
    remote.py                 # 统一 LLM 后端（openai_compat / vllm_http / local_hf）
```

输出统一写入 `outputs/decision/{route_a,route_b,compare}/<实验名>/`。

---

## Route A 完整流程

### 第一步：构建数据集

```bash
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v0.jsonl
```

输出每行包含：4 段文本（`sat, ntl, sv, poi_text`）、`poi_counts`、logit 空间 targets、原始 1–10 targets。

### 第二步：5-fold IZ CV 训练

```bash
python -m decision.train.cv_runner \
  --config decision/configs/route_a_bge.yaml
```

CLI 参数覆盖 YAML。输出：
- `segments_cache.pt` — 编码缓存（重跑复用）
- `fold{k}_best.pt` — 每折最优 checkpoint（含 poi_fitter 状态）
- `oof_predictions.jsonl` — 全量 OOF 预测
- `cv_summary.json` — 各折指标汇总

### 第三步：评估

```bash
# 全量指标（RMSE / MAE / Pearson / Spearman / R² / QWK + bootstrap CI）
python -m decision.eval.run_eval \
  --oof outputs/decision/route_a/bge_m3_v0/oof_predictions.jsonl \
  --out-dir outputs/decision/route_a/bge_m3_v0/eval

# 消融（逐段置零，报告每域 Spearman 下降量）
python -m decision.eval.ablations \
  --dataset outputs/decision/dataset_v0.jsonl \
  --run-dir outputs/decision/route_a/bge_m3_v0 \
  --out-csv outputs/decision/route_a/bge_m3_v0/ablation_report.csv
```

`run_eval` 将 OOF 展平为 `evaluate_domain_scores.py` 所需的小写列 CSV（`income, employment, …, housing, overall`，`overall` 取 7 域均值），并调用现有评估脚本。

---

## Route A' 完整流程（LightGBM）

和 Route A 共用 `dataset_v0.jsonl` 与 `segments_cache.pt`。若 `out_dir` 指向已有 MLP 版运行目录，第一次编码的 cache 会自动复用，跳过 BGE 编码。

### 第一步：构建数据集（与 Route A 相同，可跳过）

```bash
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v0.jsonl
```

### 第二步：5-fold IZ CV 训练

```bash
python -m decision.train.cv_runner_lgbm \
  --config decision/configs/route_a_lgbm.yaml
```

输出：
- `fold{k}/lgbm_{domain}.txt` — 7 个 Booster
- `fold{k}/pca.joblib`、`fold{k}/poi_fitter.json`
- `oof_predictions.jsonl`、`cv_summary.json`

### 第三步：评估（复用 Route A 的脚本）

```bash
python -m decision.eval.run_eval \
  --oof outputs/decision/route_a_lgbm/bge_m3_v0/oof_predictions.jsonl \
  --out-dir outputs/decision/route_a_lgbm/bge_m3_v0/eval
```

---

## Route B 完整流程（需远程 GPU）

### 依赖

```bash
pip install peft datasets accelerate  # transformers 已有
```

### 第一步：教师 CoT 生成

先启动远程 LLM 服务（vLLM 示例）：

```bash
vllm serve models/Qwen3-8B --port 8000
```

然后运行 rationaliser：

```bash
python -m decision.models.route_b.rationalizer \
  --config decision/configs/route_b_llm_sft.yaml
```

rationaliser 对每条训练样本：给模型看 evidence + ground truth 分数，要求生成能"引证"evidence 原句的 CoT。每条输出需通过两项校验：
1. STEP3 分数精确等于 ground truth
2. 每处 `support` 字段的短语是 evidence 的逐字子串

打印通过率，低于 80% 时提示调整 prompt 再重跑。

### 第二步：构建 SFT 数据集

```bash
python -m decision.models.route_b.sft_dataset \
  --dataset outputs/decision/dataset_v0.jsonl \
  --rationales outputs/decision/route_b/rationales_v0.jsonl \
  --out outputs/decision/route_b/sft_v0.jsonl
```

### 第三步：PEFT LoRA SFT

```bash
python -m decision.models.route_b.train_sft \
  --config decision/configs/route_b_llm_sft.yaml \
  --sft-data outputs/decision/route_b/sft_v0.jsonl
```

保存 adapter 到 `sft.output_adapter_dir`。

### 第四步：推理

**方式 A（推荐）— vLLM + LoRA serving：**

```bash
vllm serve models/Qwen3-8B \
  --enable-lora \
  --lora-modules sft=outputs/decision/route_b/lora_adapter_v0

# 在 route_b_llm_sft.yaml 中将 remote.model 改为 sft
python -m decision.infer.route_b_predict \
  --config decision/configs/route_b_llm_sft.yaml \
  --dataset outputs/decision/dataset_v0.jsonl
```

**方式 B — local_hf 后端直接加载 adapter：**

```yaml
# route_b_llm_sft.yaml
remote:
  backend: local_hf
  model: Qwen/Qwen3-8B
  adapter_path: outputs/decision/route_b/lora_adapter_v0
```

JSON 解析失败时自动回退到训练集各域均值。

---

## 目标与 schema

- 训练目标：`y = logit(clip((score−1)/9, 1e-3, 1−1e-3))`；推理时逆变换回 1–10。
- OOF / 预测输出均遵循项目约定：
  ```json
  {"datazone": "S01...", "prediction_json": {"Income": 5.3, ...}, "target_raw": {"Income": 6, ...}}
  ```
- 领域顺序：`Income, Employment, Health, Education, Access, Crime, Housing`（见 [data/targets.py](data/targets.py)）。

## 关键复用

| 需求 | 复用位置 |
|---|---|
| IZ 分组 CV | [src/glasgow_vlm/splits.py](../src/glasgow_vlm/splits.py) `group_kfold_by_iz` |
| Spearman 验证指标 | [src/glasgow_vlm/metrics.py](../src/glasgow_vlm/metrics.py) |
| 全量评估（含 bootstrap CI） | [scripts/evaluation/evaluate_domain_scores.py](../scripts/evaluation/evaluate_domain_scores.py) |
