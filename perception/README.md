# 感知层（Perception Layer）

感知层负责把多模态原始数据转换成决策层可消费的中间表示：

1. **VLM 文本证据**：调用本地 `Qwen/Qwen3-VL-8B-Instruct`，逐个 datazone 读取遥感 patch、夜光 patch、街景图像和 POI 汇总，输出描述性 evidence 短语与 17 个领域 indicator。
2. **街景语义分割特征（SVF）**：使用 ADE20K 语义分割模型统计街景中的 building、sky、road、sidewalk、tree 等像素占比，聚合到 datazone 级 parquet。

感知层只描述可见证据，不直接判断贫困或剥夺程度。主要输出被[决策层](../decision/README.md)读取，用于 SIMD 7 域分数回归。

---

## 目录结构

```
perception/
  data/
    build_patch_poi.py        # datazone_poi.csv → patch_poi.csv
  prompts/
    perception.py             # VLM evidence prompt + domain_indicators prompt
  infer/
    perceive_local.py         # Qwen3-VL-8B 本地推理，支持续跑与补 indicator
  segmentation/
    categories.py             # ADE20K → SVF 列映射
    segformer_infer.py        # SegFormer-B2 街景语义分割
    mitpsp_infer.py           # MIT ResNet18-dilated + PPM 语义分割
    aggregate.py              # image-level SVF → datazone-level SVF
    download_mit_weights.py   # MIT PSP 权重下载
  README.md
```

---

## 输入数据

| 输入 | 默认路径 | 说明 |
|---|---|---|
| satellite metadata | `dataset/satellite_dataset/satellite_metadata.csv` | 每个 patch 的 satellite / nightlight 文件路径 |
| streetview images | `dataset/streetview_dataset/<patch_id>/*.jpg` | VLM 使用的街景图像 |
| raw streetview images | `dataset/streetview_dataset_raw/<patch_id>/*.jpg` | SVF 分割使用的 640×640 原图 |
| POI 原始表 | `dataset/poi_dataset/datazone_poi.csv` | OSM POI 长表 |
| patch POI 宽表 | `dataset/poi_dataset/patch_poi.csv` | VLM 和决策层使用的 POI 类型计数 |

---

## 流程 A：VLM 文本证据

### 第一步：构建 `patch_poi.csv`

从 `dataset/poi_dataset/datazone_poi.csv` 按 patch 汇总 OSM POI 类型计数：

```bash
python -m perception.data.build_patch_poi
```

输出：

```
dataset/poi_dataset/patch_poi.csv
```

列：

```
patch_id, total, amenity, craft, emergency, healthcare, historic,
leisure, office, public_transport, railway, shop, sport, tourism
```

### 第二步：运行 Qwen3-VL 感知推理

```bash
# 预览 5 个 datazone
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
  --max-streetviews 20 \
  --max-samples 5

# 完整运行；默认自动续跑，已有有效行会跳过
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
  --max-streetviews 20

# 调试单个 datazone
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/debug.jsonl \
  --only-datazone S01006514 \
  --max-streetviews 20

# 使用 LoRA adapter
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
  --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
  --max-streetviews 20
```

默认流程会先分别为 satellite、nightlight、每张 streetview 提取短语，再用文本二次 pass 生成 `domain_indicators`。这些 indicators 是诊断/消融特征，不替代 evidence 主线。

### 只补齐缺失的 indicators

如果已有 JSONL 只包含 evidence，可原地补 `domain_indicators`：

```bash
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
  --patch-indicators-only
```

续跑时如需把“缺少 indicators”的行视为未完成：

```bash
python perception/infer/perceive_local.py \
  --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
  --require-indicators \
  --max-streetviews 20
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--output-jsonl` | 必填 | VLM 感知输出路径 |
| `--base-model-id` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace 模型 ID |
| `--adapter-path` | 无 | LoRA adapter 目录 |
| `--max-streetviews` | `20` | 每个 patch 最多处理的街景数 |
| `--max-new-tokens` | `6144` | evidence 生成 token 上限 |
| `--indicator-max-new-tokens` | `2048` | indicator 二次 pass token 上限 |
| `--temperature` | `0.0` | 采样温度，0 表示 greedy |
| `--max-samples` | `0` | 限制处理 patch 数，0 表示全量 |
| `--only-datazone` | 无 | 只处理指定 datazone |
| `--overwrite` | `False` | 覆盖输出，从头运行 |
| `--no-quantization` | `False` | 使用 BF16 加载，不做 4-bit 量化 |
| `--merge-adapter` | `False` | 推理前将 adapter 合并进基模型 |
| `--skip-indicators` | `False` | 跳过 indicator 二次 pass，常用于调试 |
| `--require-indicators` | `False` | 续跑时要求 evidence 和 indicators 都存在 |
| `--patch-indicators-only` | `False` | 只给已有 JSONL 补 indicators |

### VLM 输出格式

`outputs/perception/qwen3vl_8b_perception.jsonl` 每行一个 patch/datazone：

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
      "POI": ["42 POIs total: amenity x18, shop x9"]
    },
    "domain_indicators": {
      "physical_disorder": {"score": 2, "cue": "minor graffiti visible"},
      "streetlight_presence": {"score": 3, "cue": "regular lamp posts"}
    }
  }
}
```

`evidence` 是下游主输入：每个模态通常为 3-6 个描述短语，每条短语尽量保持在 8 词以内。`domain_indicators` 包含 17 个 0-4 分的可解释视觉/POI proxy，缺失时决策层会使用全零向量并设置 missing flag。

---

## 流程 B：街景语义分割 SVF

SVF 分支不调用 VLM。它对街景原图做语义分割，输出每张图中 15 类比例：

```
building, sky, tree, road, grass, sidewalk, person, earth, plant,
car, fence, signboard, streetlight, pole, other
```

### SegFormer-B2

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
```

### MIT PSP

```bash
python -m perception.segmentation.mitpsp_infer \
  --metadata dataset/streetview_metadata.csv \
  --raw-dir dataset/streetview_dataset_raw \
  --output outputs/perception/svf/image_level_mitpsp.parquet \
  --input-size 512 \
  --batch-size 16 \
  --device cuda:0

python -m perception.segmentation.aggregate \
  --image-level outputs/perception/svf/image_level_mitpsp.parquet \
  --output outputs/perception/svf/datazone_svf_mitpsp.parquet
```

`mitpsp_infer.py` 会通过 `download_mit_weights.py` 自动下载官方 MIT ADE20K 权重；如已离线准备好权重，可传入 `--weights-dir`。

### SVF 输出格式

image-level parquet：

```
image_id, patch_id, datazone, pano_index,
building, sky, tree, road, grass, sidewalk, person, earth, plant,
car, fence, signboard, streetlight, pole, other
```

datazone-level parquet：

```
datazone, n_patches, n_images, n_images_filtered,
building_mean, sky_mean, ..., other_mean,
building_std, sky_std, ..., other_std
```

聚合前会过滤疑似异常街景：

| 过滤规则 | 解释 |
|---|---|
| `sky < 0.01 and building > 0.6` | 可能为室内图或近距离墙面 |
| `road > 0.5 and sidewalk < 0.01 and building < 0.1` | 可能为高速/非居住街景 |
| `other > 0.8` | 模糊或模型无法识别 |

---

## 下游衔接

### 构建决策层数据集

```bash
python -m decision.data.build_dataset \
  --perception outputs/perception/qwen3vl_8b_perception.jsonl \
  --simd dataset/SIMD/SIMD_score.csv \
  --out outputs/decision/dataset_v1.jsonl
```

### 运行 SVF 消融

```bash
python -m decision.train.cv_runner_svf \
  --config decision/configs/svf/route_c_modality_sep_svf_segformer_v0.yaml

python -m decision.train.cv_runner_svf \
  --config decision/configs/svf/route_c_modality_sep_svf_mitpsp_v0.yaml
```

---

## 依赖与资源

VLM 推理：

```bash
pip install torch transformers peft pillow accelerate bitsandbytes
```

SVF 分割：

```bash
pip install torch torchvision transformers pillow pandas pyarrow numpy
```

资源需求：

| 流程 | 需求 |
|---|---|
| Qwen3-VL 4-bit 推理 | CUDA GPU，约 10GB 显存 |
| Qwen3-VL BF16 推理 | CUDA GPU，约 20GB 显存 |
| SegFormer-B2 / MIT PSP | CUDA GPU 推荐；CPU 可跑但很慢 |

---

## 关键复用入口

| 需求 | 位置 |
|---|---|
| 解析 VLM JSONL | [decision/data/parse_perception.py](../decision/data/parse_perception.py) |
| 构建决策层 dataset | [decision/data/build_dataset.py](../decision/data/build_dataset.py) |
| Route C + SVF CV | [decision/train/cv_runner_svf.py](../decision/train/cv_runner_svf.py) |
| SVF 配置 | [decision/configs/svf/](../decision/configs/svf/) |
