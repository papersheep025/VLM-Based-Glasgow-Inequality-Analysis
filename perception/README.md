# 感知层（Perception Layer）

读取多模态数据集（遥感 patch、夜光 patch、街景图像、POI），调用本地 Qwen3-VL-8B 模型，为每个 datazone 生成多模态文本证据，输出 `outputs/perception/qwen3vl_8b_perception.jsonl`。

感知层输出作为[决策层](../decision/README.md)的输入，不做任何剥夺程度判断，仅描述可见视觉特征。

---

## 目录结构

```
perception/
  data/
    build_patch_poi.py     # 从 datazone_poi.csv 聚合为 patch 级 POI 宽表
  prompts/
    perception.py          # 系统提示词、单模态提示词、多模态合并提示词
  infer/
    perceive_local.py      # 主推理脚本（逐 datazone 调用 VLM，支持续跑）
  README.md
```

---

## 完整流程

### 第一步：构建 patch_poi.csv

从 `dataset/poi_dataset/datazone_poi.csv` 按 patch 汇总 OSM POI 类型计数：

```bash
python -m perception.data.build_patch_poi
```

输出：`dataset/poi_dataset/patch_poi.csv`
列：`patch_id, total, amenity, craft, emergency, healthcare, historic, leisure, office, public_transport, railway, shop, sport, tourism`

### 第二步：运行感知推理

```bash
# 预览（5 个 datazone）
python perception/infer/perceive_local.py \
    --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
    --max-streetviews 20 --max-samples 5

# 完整运行（自动续跑，--overwrite 从头开始）
python perception/infer/perceive_local.py \
    --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
    --max-streetviews 20

# 调试单个 datazone
python perception/infer/perceive_local.py \
    --output-jsonl outputs/perception/debug.jsonl \
    --only-datazone S01006514

# 加载 LoRA adapter
python perception/infer/perceive_local.py \
    --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
    --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
    --max-streetviews 20
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-jsonl` | 必填 | 输出路径 |
| `--base-model-id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace 模型 ID |
| `--adapter-path` | 无 | LoRA adapter 目录（可选） |
| `--max-streetviews` | 20 | 每个 datazone 最多处理的街景数 |
| `--max-new-tokens` | 6144 | 生成 token 上限 |
| `--temperature` | 0.0 | 采样温度（0 = greedy） |
| `--max-samples` | 0（全量） | 限制处理的 datazone 数 |
| `--only-datazone` | 无 | 只处理指定 datazone（调试用） |
| `--overwrite` | False | 覆盖已有输出，否则自动续跑 |
| `--no-quantization` | False | 用 BF16 加载，不做 4-bit 量化 |
| `--merge-adapter` | False | 推理前将 adapter 合并进基模型 |

---

## 输出格式

`outputs/perception/qwen3vl_8b_perception.jsonl`，每行一个 datazone：

```json
{
  "patch_id": "S01006514",
  "datazone": "S01006514",
  "streetview_indices": [0, 1, 2],
  "reasoning_json": {
    "evidence": {
      "satellite": ["dense terraced housing", "narrow road grid", "limited green space"],
      "nightlight": ["dim uniform glow", "patchy at edges"],
      "streetview_00": ["cracked pavement", "closed shopfront"],
      "streetview_01": ["open grocery store", "well-kept facade"],
      "POI": ["mostly convenience amenities", "few healthcare facilities"],
      "general": ["streetview consistent with dense satellite layout"]
    }
  }
}
```

每个模态的 `evidence` 字段为 3–6 个描述短语（每条 ≤8 词）。推理脚本对每张图片单独调用 VLM（satellite、nightlight 各 1 次，每张街景 1 次），然后合并 evidence，不做剥夺程度推断。

---

## 依赖

```bash
pip install torch transformers peft pillow accelerate bitsandbytes
```

需要 CUDA GPU（4-bit 量化模式约需 ~10GB 显存；BF16 模式约需 ~20GB）。

---

## 关键复用

| 需求 | 位置 |
|---|---|
| 感知输出 → 决策层输入 | [decision/data/parse_perception.py](../decision/data/parse_perception.py) |
| 感知输出 → 决策层数据集 | [decision/data/build_dataset.py](../decision/data/build_dataset.py) |
