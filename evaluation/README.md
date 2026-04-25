# 评估与可视化分支（Evaluation）

本目录收纳论文主表之外的评估、诊断、消融和可视化脚本。当前论文主表的统一口径仍由 `decision.eval.compare` 实现：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/paper_v1.yaml
```

`paper_v1.yaml` 和 `diagnostics_v1.yaml` 中需要调用脚本文件的实验，已指向本目录。

---

## 输入与输出约定

### OOF 输入

大多数当前决策层实验输出：

```
outputs/decision/route_c/<run_name>/oof_predictions.jsonl
```

每行格式：

```json
{
  "datazone": "S01006514",
  "fold": 0,
  "prediction_json": {"Income": 5.3, "Employment": 4.8},
  "target_raw": {"Income": 6, "Employment": 5}
}
```

### 聚合 CSV 输入

详细评估和可视化脚本读取 datazone 级 CSV，至少需要：

```
datazone,income,employment,health,education,housing,access,crime,overall
```

`eval_decision_oof.py` 会自动把 OOF JSONL 聚合成这个 CSV：

```
<output-dir>/pred_aggregated.csv
```

### Ground truth 与空间文件

默认路径：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--simd-csv` | `dataset/SIMD/SIMD_data.csv` | SIMD ground truth，支持标准列名或 precise 列名 |
| `--rank-csv` | `dataset/SIMD/SIMD_data.csv` | 需要 `SIMD2020v2_Rank`，用于 rank-band 分组 |
| `--shapefile` | `dataset/glasgow_datazone/glasgow_datazone.shp` | 空间 lag baseline、Moran's I、地图 |

---

## 推荐操作：从 OOF 一键评估并生成图

这是最常用路线：输入一个 `oof_predictions.jsonl`，输出详细指标 CSV 和所有可视化。

```bash
python evaluation/eval_decision_oof.py \
  --oof-jsonl outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl \
  --output-dir outputs/evaluation/decision_modality_sep_v1 \
  --simd-csv dataset/SIMD/SIMD_data.csv \
  --shapefile dataset/glasgow_datazone/glasgow_datazone.shp
```

快速模式，不跑空间分析、不生成 PDF：

```bash
python evaluation/eval_decision_oof.py \
  --oof-jsonl outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl \
  --output-dir outputs/evaluation/decision_modality_sep_v1_fast \
  --no-spatial \
  --no-pdf
```

输出目录结构：

```
outputs/evaluation/decision_modality_sep_v1/
  pred_aggregated.csv
  domain_evaluation_report.csv
  figures/
    fig1_domain_scatter.png
    fig2_domain_error_bar.png
    fig3_error_by_rank_band.png
    fig4_correlation_heatmap.png
    fig4b_diagonal_bar.png
    fig5_glasgow_map.html
    fig6_lisa_cluster_map.png
    fig7_r2_bar.png
    *.pdf                 # 未传 --no-pdf 时生成
```

---

## 只生成详细指标

如果已经有 `pred_aggregated.csv`，可只跑指标，不出图：

```bash
python evaluation/evaluate_domain_scores.py \
  --pred-csv outputs/evaluation/decision_modality_sep_v1/pred_aggregated.csv \
  --simd-csv dataset/SIMD/SIMD_data.csv \
  --rank-csv dataset/SIMD/SIMD_data.csv \
  --shapefile dataset/glasgow_datazone/glasgow_datazone.shp \
  --output-csv outputs/evaluation/decision_modality_sep_v1/domain_evaluation_report.csv
```

跳过空间分析：

```bash
python evaluation/evaluate_domain_scores.py \
  --pred-csv outputs/evaluation/decision_modality_sep_v1/pred_aggregated.csv \
  --output-csv outputs/evaluation/decision_modality_sep_v1/domain_evaluation_report.csv \
  --no-spatial
```

主要输出：

| 文件 | 内容 |
|---|---|
| `domain_evaluation_report.csv` | 每个 domain 的 RMSE、MAE、Pearson、Spearman、R2、QWK 和 bootstrap CI |
| 同目录额外 CSV | rank-band breakdown、mean baseline、spatial lag baseline、residual spatial diagnostics（取决于输入和是否启用空间分析） |

---

## 只生成可视化

如果已有 datazone 级预测 CSV，可直接生成图片和地图：

```bash
python evaluation/visualize_domain_scores.py \
  --pred-csv outputs/evaluation/decision_modality_sep_v1/pred_aggregated.csv \
  --simd-csv dataset/SIMD/SIMD_data.csv \
  --rank-csv dataset/SIMD/SIMD_data.csv \
  --shapefile dataset/glasgow_datazone/glasgow_datazone.shp \
  --output-dir outputs/evaluation/decision_modality_sep_v1/figures
```

只生成 PNG 和 HTML，不生成 PDF：

```bash
python evaluation/visualize_domain_scores.py \
  --pred-csv outputs/evaluation/decision_modality_sep_v1/pred_aggregated.csv \
  --output-dir outputs/evaluation/decision_modality_sep_v1/figures_fast \
  --no-pdf
```

可视化输出：

| 文件 | 说明 |
|---|---|
| `fig1_domain_scatter.png/pdf` | 预测 vs 真值散点矩阵，带 Spearman rho 与 RMSE |
| `fig2_domain_error_bar.png/pdf` | 各领域 RMSE / MAE 柱状图 |
| `fig3_error_by_rank_band.png/pdf` | 不同 SIMD rank band 的误差分布 |
| `fig4_correlation_heatmap.png/pdf` | 预测领域 vs 真值领域 Spearman 热图 |
| `fig4b_diagonal_bar.png/pdf` | 同领域相关性条形图 |
| `fig5_glasgow_map.html` | Plotly 交互式 Glasgow 地图，可切换 predicted / true / error |
| `fig6_lisa_cluster_map.png/pdf` | LISA cluster 空间误差图 |
| `fig7_r2_bar.png/pdf` | 各领域 R2 条形图 |

---

## 论文主表与诊断矩阵

比较已有 paper runs：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/paper_v1.yaml
```

通过 manifest 只重建比较表：

```bash
python -m decision.experiments.run_manifest \
  --manifest decision/experiments/manifests/paper_v1.yaml \
  --only compare
```

比较已有 diagnostics runs：

```bash
python -m decision.eval.compare \
  --manifest decision/experiments/manifests/diagnostics_v1.yaml
```

输出：

```
outputs/evaluation/paper_v1_route_c_comparison.csv
outputs/evaluation/diagnostics_v1_comparison.csv
```

---

## 当前会被 manifest 调用的脚本

| 脚本 | 用途 | 常用输出 |
|---|---|---|
| `indicator_only_cv.py` | 仅使用 17 个 VLM domain indicators 的 RidgeCV baseline | `outputs/decision/route_c/indicators_only_v2/` |
| `spatial_residual_diagnostics.py` | `sar_only` 与 `residual_after_sar` 诊断 | `outputs/decision/route_c/diagnostics_*_v1/` |
| `stacking_v1.py` | SBERT-Ridge OOF + 结构化 HGB 的 stacking 消融 | `outputs/decision/route_c/stacking_v1/` |

### indicators-only baseline

```bash
python evaluation/indicator_only_cv.py \
  --perception outputs/perception/qwen3vl_8b_perception_v2.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out-dir outputs/decision/route_c/indicators_only_v2
```

可选：

| 参数 | 说明 |
|---|---|
| `--n-splits` | CV 折数，默认 5 |
| `--seed` | 随机种子，默认 42 |
| `--regressor` | `ridge_cv` / `lasso_cv` / `elasticnet_cv` |
| `--include-missing-flag` | 拼接 indicator missing flag |

### SAR / residual diagnostics

```bash
python evaluation/spatial_residual_diagnostics.py \
  --mode sar_only \
  --config decision/configs/route_c_modality_sep_v1.yaml \
  --out-dir outputs/decision/route_c/diagnostics_sar_only_v1
```

```bash
python evaluation/spatial_residual_diagnostics.py \
  --mode residual_after_sar \
  --config decision/configs/diagnostics/route_c_modality_sep_v1_no_sar.yaml \
  --out-dir outputs/decision/route_c/diagnostics_residual_after_sar_v1 \
  --cache-source outputs/decision/route_c/modality_sep_v1/caption_cache.pt
```

### stacking_v1

```bash
python evaluation/stacking_v1.py \
  --dataset outputs/decision/dataset_v1.jsonl \
  --ridge-oof outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl \
  --svf outputs/perception/svf/datazone_svf_segformer.parquet \
        outputs/perception/svf/datazone_svf_mitpsp.parquet \
  --out-dir outputs/decision/route_c/stacking_v1
```

---

## 辅助诊断

### indicator 相关性表

```bash
python evaluation/indicator_diagnostics.py \
  --perception outputs/perception/qwen3vl_8b_perception_v2.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/evaluation/indicator_rho.csv \
  --top-k 15
```

### 早期 SVF 对比脚本

当前通常用 `paper_v1.yaml` + `decision.eval.compare`。如需单独比较几个 OOF：

```bash
python evaluation/compare_svf_ablation.py \
  --baseline outputs/decision/route_c/modality_sep_v1/oof_predictions.jsonl \
  --runs outputs/decision/route_c/modality_sep_svf_segformer_v1/oof_predictions.jsonl \
         outputs/decision/route_c/modality_sep_svf_mitpsp_v1/oof_predictions.jsonl \
         outputs/decision/route_c/modality_sep_svf_both_v1/oof_predictions.jsonl \
  --output outputs/evaluation/svf_ablation_summary_v1.csv
```

---

## 常见问题

### 只想要 HTML 地图

`visualize_domain_scores.py` 会一次生成所有图，不能只生成单个图。要快速出地图，可加 `--no-pdf` 减少 PDF 写入时间。

### 没有 shapefile 或缺空间依赖

详细评估可使用 `--no-spatial` 跳过 Moran's I 和 spatial lag baseline。可视化中的 `fig5_glasgow_map.html` 和 `fig6_lisa_cluster_map` 需要 shapefile。

### `SIMD2020v2_Rank` 不在 `--simd-csv`

传入单独的 `--rank-csv`，该 CSV 需要包含：

```
datazone,SIMD2020v2_Rank
```

### 主表口径和详细评估口径不同

论文主表使用 `decision/eval/oof.py`：datazone 聚合后 stacked 7-domain Spearman / R2。`evaluate_domain_scores.py` 会输出更多 domain-level 指标和可视化辅助统计，不替代主表口径。
