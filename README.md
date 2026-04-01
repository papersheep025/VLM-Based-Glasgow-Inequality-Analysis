# Glasgow 城市不平等 VLM 项目

这是一个面向 Glasgow 城市不平等分析的多模态视觉语言模型项目。当前主线已经调整为：

- 先做街景、卫星、夜光的空间对齐
- 再生成用于推理的 JSONL
- 使用 `Qwen3-VL-Plus` API 做小批量或全量预测
- 支持断点重跑，适合不能长时间开机的环境

## 当前目录结构

- `dataset/streetview_dataset`
  - 原始街景图像
- `dataset/satellite_dataset`
  - 原始遥感数据
  - `satellite_patches`：卫星 patch
  - `satellite_ntl_patches`：夜光 patch
- `dataset/streetview_satellite_aligned`
  - 街景 + 卫星 对齐结果
- `dataset/streetview_ntl_aligned`
  - 街景 + 夜光 对齐结果
- `dataset/streetview_satellite_ntl_aligned`
  - 街景 + 卫星 + 夜光 三模态对齐结果
- `outputs/predictions`
  - 模型预测输出
- `src/glasgow_vlm`
  - prompt、数据集封装、指标等核心代码

## 环境安装

如果你已经有 `.venv`，直接安装依赖：

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 先做空间对齐

### 1. 三模态空间对齐

```powershell
.\.venv\Scripts\python.exe scripts\build_streetview_satellite_ntl_alignment.py
```

会生成：

- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv`
- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.jsonl`
- `dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_summary.json`

### 2. 生成纯空间对齐 JSONL

当前版本的 `scripts/build_vlm_jsonl.py` **不再合并 SIMD**，只做空间配对和 JSONL 生成。

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_jsonl.py --alignment-csv dataset/streetview_satellite_ntl_aligned/streetview_satellite_ntl_alignment.csv --output-dir dataset/streetview_satellite_ntl_aligned/vlm_data --input-mode triple --task explain
```

会生成：

- `triple_explain_train.jsonl`
- `triple_explain_val.jsonl`
- `triple_explain_test.jsonl`
- `triple_explain_all.jsonl`

如果你想做双模态，把 `--input-mode triple` 改成 `dual`，并把 `--task explain` 保留即可。

### 3. 做 smoke test

`smoke_test_vlm_pipeline.py` 现在适配纯空间对齐版，不再强制要求 `answer_json`。

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_vlm_pipeline.py --jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_train.jsonl --num-samples 5
```

### 4. 小批量 API 预览

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_preview.jsonl --input-mode triple --task explain --max-samples 5
```

## 全量运行

如果小批量结果正常，就去掉 `--max-samples 5`：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --input-mode triple --task explain
```

### 断点重跑

当前 `predict_qwen3_vl_plus_api.py` 支持断点重跑：

- 直接重新运行同一条命令即可
- 脚本会自动读取已有输出文件中的 `id`
- 已完成的样本会自动跳过
- 追加写入未完成部分

如果你想强制从头开始，加入 `--overwrite`：

```powershell
.\.venv\Scripts\python.exe scripts\predict_qwen3_vl_plus_api.py --input-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --input-mode triple --task explain --overwrite
```

### 评估

如果输出文件里包含预测字段，可以继续用评估脚本检查结果：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_predictions.py --gold-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --pred-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl
```

### 可选聚合

如果你希望把样本级结果汇总到 `datazone` 层，可以继续使用：

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_datazone_predictions.py --pred-jsonl outputs/predictions/qwen3_vl_plus_triple_test.jsonl --gold-jsonl dataset/streetview_satellite_ntl_aligned/vlm_data/triple_explain_test.jsonl --output-csv outputs/datazone_predictions.csv
```

## 推荐的当前流程

1. 先做三模态空间对齐
2. 生成 `triple_explain_*.jsonl`
3. 跑 smoke test
4. 先做 `--max-samples 5` 的小批量 API 预览
5. 如果没问题，再做全量预测
6. 需要中断后恢复时，直接重跑同一命令即可

## 提示词设计

当前 `src/glasgow_vlm/prompts.py` 采用的是结构化输出风格：

- 模型内部只做隐式推理
- 输出只保留 JSON
- `evidence` 是结构化对象
- `visual_indicators` 需要模型根据图像推理
- `explain` 任务会带有 `above_median_deprivation` 和 `predicted_rank_band`

## 说明

当前 `build_vlm_jsonl.py` 是**纯空间对齐版**，不再合并 SIMD 标签。也就是说：

- 它适合做数据管线、prompt 和 API 推理测试
- 如果你后面要重新做 `rank` 或 `ordinal` 的监督训练，需要再单独把 SIMD 标签接回去

## 结果预期

跑通后你会得到：

- 空间对齐结果
- 纯空间对齐 JSONL
- smoke test 通过的样本
- 可断点重跑的 API 预测结果
- 样本级与 `datazone` 级分析表


