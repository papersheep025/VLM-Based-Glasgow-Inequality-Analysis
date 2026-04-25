# 数据准备分支（Data Preparation）

本目录收纳一次性 / 低频运行的数据准备脚本：从原始矢量与栅格数据出发，构建感知层与决策层所需的所有中间产物。

> 当前论文主线复现通常**直接从已经准备好的 `dataset/` 与 `outputs/perception/` 开始**，不需要重跑本目录的脚本。本 README 描述的是「全链路从零构建」时的架构。

---

## 1. 总体架构

数据准备分为三大子系统，共同写入项目根目录的 `dataset/`：

```
                ┌─────────────────────────────────────────────┐
                │   原始数据源（External / Raw Inputs）        │
                │   ─ Glasgow datazone shapefile (746 zones)  │
                │   ─ 卫星 GeoTIFF（glasgow.tif）              │
                │   ─ 夜光 GeoTIFF（glasgow_ntl.tif）          │
                │   ─ Mapillary / Google Street View API       │
                │   ─ OSM Overpass / Google Places API         │
                │   ─ SIMD 2020 score CSV                     │
                └─────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  A. 影像 Patch   │      │  B. 街景抓取与    │      │  C. POI 抓取与    │
│     裁剪          │      │     对齐          │      │     匹配          │
│ (TIFF → PNG)     │      │ (API → JPG)      │      │ (API → CSV)      │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
   datazone_patches/         streetview_dataset/         poi_dataset/
   satellite_dataset/                                    osm_poi/
                                    │
                                    ▼
                   ┌──────────────────────────────────┐
                   │  D. 验证 / 可视化（QA）           │
                   │  Folium 交互地图 + 覆盖统计       │
                   └──────────────────────────────────┘
                                    │
                                    ▼
              下游：perception/ → decision/
```

---

## 2. 子系统 A：影像 Patch 裁剪

将原始 GeoTIFF（卫星 + 夜光）按 datazone 切成统一规格的 PNG patch。

| 脚本 | 职责 | 输入 | 输出 |
|---|---|---|---|
| [build_datazone_centroid_patches.py](build_datazone_centroid_patches.py) | 按 datazone **质心**裁剪 333m × 333m 卫星 / NTL patch（卫星 384², NTL 256²）；附带街景对齐 | `glasgow_datazone.shp`, `glasgow.tif`, `glasgow_ntl.tif`, `streetview_dataset/metadata.csv` | `datazone_patches/satellite/`, `datazone_patches/ntl/`, `datazone_patch_metadata.csv`, `datazone_streetview_alignment.csv`, `summary.json` |
| [build_datazone_extra_patches.py](build_datazone_extra_patches.py) | 对**面积大于阈值**（area_ratio ≥ 3.5）的 datazone 额外采样 1–8 个 patch，避免大区域只用一个质心代表 | 同上 + centroid metadata | `datazone_patches/satellite/{dz}_extra_*.png`、扩展后的 metadata |
| [resample_patches_from_tiff.py](resample_patches_from_tiff.py) | 从原 TIFF 重新采样所有 patch（统一 EPSG:32630，NTL 用 WarpedVRT 即时重投影），用于覆盖已有 patch | `satellite_dataset/satellite_metadata.csv`, 两个 TIFF | 覆盖 `satellite_dataset/{dz}_satellite.png` 与 NTL patch |

**关键约定：**
- 物理边长固定 `PATCH_SIDE_M = 333.0` 米（≈ datazone 平均面积一半的边长）。
- 投影统一在 EPSG:27700（British National Grid）下做几何运算，再转回 WGS84 与栅格 CRS。
- `extract_patch()` 使用 `rasterio.windows.Window` + `boundless=True` 容许 patch 越界（用 0 填充）。

---

## 3. 子系统 B：街景抓取与对齐

在每个 datazone 内部采样若干点，请求 Mapillary / Google Street View API，下载并存储 336×336 街景图。

| 脚本 | 职责 |
|---|---|
| [scraper/datazone_to_dataset_streetview.py](scraper/datazone_to_dataset_streetview.py) | 主流：Mapillary，每 datazone 采样 30 点，断点续传（每 5 点写 checkpoint） |
| [scraper/datazone_to_dataset_streetview_mapillary.py](scraper/datazone_to_dataset_streetview_mapillary.py) | Mapillary 备用版本 |
| [scraper/datazone_to_dataset_streetview_gsv.py](scraper/datazone_to_dataset_streetview_gsv.py) | 按 bbox 从 Google Street View 抓取（需 API key） |
| [scraper/download_gsv_images.py](scraper/download_gsv_images.py) | 根据 GSV metadata 批量下载图像 |
| [scraper/datazone_to_dataset_satellite.py](scraper/datazone_to_dataset_satellite.py) | 备用：通过 API 抓取卫星瓦片（默认走本地 TIFF，本脚本仅在没有 TIFF 时用） |
| [scraper/align_streetview_to_satellite.py](scraper/align_streetview_to_satellite.py) | 把街景点对齐到卫星 patch（落点匹配 + 距离裁剪），生成对齐 CSV |

**输出契约：** `dataset/streetview_dataset/metadata.csv`（含 `datazone, image, lat, lon`），`dataset/streetview_dataset/*.jpg`。子系统 A 的 `build_datazone_centroid_patches.py` 在最后一步会读这个 metadata 以统计 `sv_count`、生成 alignment。

---

## 4. 子系统 C：POI 抓取与匹配

收集兴趣点（OSM + Google Places），并做 point-in-polygon 匹配到 datazone。

| 脚本 | 职责 | 输出 |
|---|---|---|
| [scraper/osm_poi.py](scraper/osm_poi.py) | Overpass API 抓取 Glasgow 范围内 OSM POI | `dataset/osm_poi/osm_poi.csv` |
| [scraper/google_places_poi.py](scraper/google_places_poi.py) | Google Places Nearby Search 抓取（覆盖 OSM 缺失类别） | `dataset/poi_dataset/google_places_poi.csv` |
| [create_datazone_poi.py](create_datazone_poi.py) | **零依赖**（纯 stdlib）实现 shapefile + DBF 读取 + 射线法 PIP，把每条 POI 落到 datazone | `dataset/osm_poi/datazone_poi.csv` |

`datazone_poi.csv` 是决策层 [perception/data/build_patch_poi.py](../perception/data/build_patch_poi.py) 的直接上游，会被进一步聚合成 `patch_poi.csv` 宽表。

---

## 5. 子系统 D：验证 / 可视化

| 脚本 | 职责 |
|---|---|
| [visualize_datazone_patches_map.py](visualize_datazone_patches_map.py) | Folium 交互地图：栅格底图 + datazone 边界 + patch 中心 + 街景点，用于人工审查覆盖 / 漏点 |

---

## 6. 推荐执行顺序

从零构建（**首次** / 数据源更新时）：

```bash
# Step 1 — 抓取街景与 POI（耗时，依赖 API 配额）
python data_preparation/scraper/datazone_to_dataset_streetview.py
python data_preparation/scraper/osm_poi.py

# Step 2 — POI → datazone 匹配
python data_preparation/create_datazone_poi.py

# Step 3 — 卫星 / NTL patch 裁剪 + 街景对齐
python data_preparation/build_datazone_centroid_patches.py
python data_preparation/build_datazone_extra_patches.py        # 可选，覆盖大 datazone

# Step 4 — QA
python data_preparation/visualize_datazone_patches_map.py
```

**重新采样**（仅 TIFF 更换、其它不变）：

```bash
python data_preparation/resample_patches_from_tiff.py
```

---

## 7. 与主线的接口

下游消费者明确读取以下产物：

| 产物 | 下游 |
|---|---|
| `dataset/satellite_dataset/satellite_metadata.csv` | [perception/infer/perceive_local.py](../perception/infer/perceive_local.py) |
| `dataset/datazone_patches/satellite/`, `datazone_patches/ntl/` | 同上（dz 级 patch） |
| `dataset/streetview_dataset/` + `metadata.csv` | 同上（街景子集） |
| `dataset/streetview_dataset_raw/` | `perception/segmentation/*_infer.py` |
| `dataset/osm_poi/datazone_poi.csv` | [perception/data/build_patch_poi.py](../perception/data/build_patch_poi.py) |
| `dataset/glasgow_datazone/glasgow_datazone.shp` | [src/glasgow_vlm/splits.py](../src/glasgow_vlm/splits.py)、[decision/data/spatial_neighbors.py](../decision/data/spatial_neighbors.py) |

---

## 8. 设计约定

- **路径根**：所有脚本以 `ROOT = Path(__file__).resolve().parents[1]` 锚定项目根目录；CLI 参数可覆盖默认输入 / 输出路径。
- **CRS 约定**：几何运算用 EPSG:27700；栅格采样按各自 TIFF 的原生 CRS；最终输出统一回 WGS84 (EPSG:4326)。
- **Patch 物理尺寸**：333 m × 333 m（约 datazone 一半面积），SAT 384², NTL 256²。
- **续跑**：抓取类脚本（街景）每 N 点写一次 checkpoint，重启时跳过已完成项；patch 裁剪脚本默认跳过已存在文件，`--overwrite` 可强制覆盖。
- **依赖最小化原则**：`create_datazone_poi.py` 故意只用标准库（无 geopandas / shapely / fiona），便于在最简环境运行。
