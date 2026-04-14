import geopandas as gpd
import pandas as pd
import numpy as np
import os
import requests
from shapely.geometry import Point
from tqdm import tqdm
import random
import json
import csv
from datetime import datetime, timezone

# -------------------------------
# 参数配置
# -------------------------------
DATAZONE_SHP = "glasgow_datazone/glasgow_datazone.shp"
SIMD_SHP = "SIMD/simd2020_withgeog/sc_dz_11.shp"
SIMD_CSV = "SIMD/simd2020_withgeog/simd2020_withinds.csv"
OUTPUT_DIR = "streetview_dataset"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
STREETVIEW_JSON = os.path.join(OUTPUT_DIR, "glasgow_streetview.json")
STREETVIEW_JSONL = os.path.join(OUTPUT_DIR, "glasgow_streetview.jsonl")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_streetview.json")
ACCESS_TOKEN = "MLYATgedGw5TDFZCzgbA4jIsR0BoZCCirUUIX11LwhEPI1nTjg3xdnsqcfVSdWfKdK5jjZAxUrTH15ylblYcJLWwvCdreIymoXED7wAJ3a6xlcDoXfOpuOZAY79n0X14wZDZD"  # User Access Token

POINTS_PER_ZONE = 30  # 每个 DataZone 采样点数量
SEARCH_RADIUS = 50   # 米，不再使用，但保留
CHECKPOINT_EVERY_POINTS = 5  # 每处理多少个采样点写一次断点

# 创建输出目录
os.makedirs(IMAGE_DIR, exist_ok=True)


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load checkpoint {path}: {e}")
        return None


def load_seen_ids_from_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("mapillary_id")
                if mid:
                    seen.add(str(mid))
    except Exception as e:
        print(f"Warning: failed to read existing metadata CSV {path}: {e}")
    return seen


def load_metadata_from_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _to_int(x):
        try:
            if x is None or x == "":
                return None
            return int(float(x))
        except Exception:
            return None

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image": row.get("image"),
                "datazone": row.get("datazone"),
                "lat": _to_float(row.get("lat")),
                "lon": _to_float(row.get("lon")),
                "mapillary_id": str(row.get("mapillary_id")) if row.get("mapillary_id") else None,
                "deprivation_rank": _to_int(row.get("deprivation_rank")),
                "deprivation_quintile": _to_int(row.get("deprivation_quintile")),
                "deprivation_level": row.get("deprivation_level"),
            })
    return rows


def load_streetview_data_from_jsonl(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    items = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj.get("mapillary_id") or obj.get("id")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                items.append(obj)
            except Exception:
                continue
    return items

# -------------------------------
# 加载 shapefile
# -------------------------------
gdf = gpd.read_file(DATAZONE_SHP)
print(f"Loaded {len(gdf)} datazones from shapefile")

# -------------------------------
# 加载 SIMD 数据
# -------------------------------
simd_gdf = gpd.read_file(SIMD_SHP)
simd_df = pd.read_csv(SIMD_CSV)
print(f"Loaded SIMD data: {len(simd_gdf)} zones, {len(simd_df)} records")

# 合并 SIMD 数据到 datazone
gdf = gdf.merge(simd_df[['Data_Zone', 'SIMD2020v2_Rank', 'SIMD2020v2_Quintile']], left_on='DataZone', right_on='Data_Zone', how='left')
print(f"Merged SIMD data, deprivation ranks available for {gdf['SIMD2020v2_Rank'].notna().sum()} zones")

# 预构建从 DataZone 代码到 SIMD 信息的字典，用于快速查找
zone_lookup = {}
for _, row in gdf.iterrows():
    dz = row['DataZone']
    zone_lookup[dz] = (
        row.get('SIMD2020v2_Rank'),
        row.get('SIMD2020v2_Quintile')
    )


# -------------------------------
# 生成随机采样点
# -------------------------------
def generate_random_points(polygon, num_points):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
    return points

checkpoint = load_checkpoint(CHECKPOINT_FILE)
if checkpoint and isinstance(checkpoint.get("all_points"), list) and checkpoint["all_points"]:
    all_points = checkpoint["all_points"]
    start_index = int(checkpoint.get("next_point_index", 0) or 0)
    start_index = max(0, min(start_index, len(all_points)))
    checkpoint_created_at = checkpoint.get("created_at") or _utc_now_iso()
    print(f"Resuming from checkpoint: {CHECKPOINT_FILE}, next_point_index={start_index}/{len(all_points)}")
else:
    checkpoint_created_at = _utc_now_iso()
    all_points = []
    for idx, row in gdf.iterrows():
        polygon = row.geometry
        dz_code = row["DataZone"]  # 假设 shapefile 有 DataZone 字段
        points = generate_random_points(polygon, POINTS_PER_ZONE)
        for pt in points:
            all_points.append({
                "datazone": dz_code,
                "lat": pt.y,
                "lon": pt.x
            })

    start_index = 0
    atomic_write_json(CHECKPOINT_FILE, {
        "version": 1,
        "created_at": checkpoint_created_at,
        "updated_at": _utc_now_iso(),
        "total_points": len(all_points),
        "next_point_index": start_index,
        "all_points": all_points,
        "completed": False,
    })
    print(f"Generated {len(all_points)} sampling points and created checkpoint: {CHECKPOINT_FILE}")

# -------------------------------
# Mapillary API 搜索街景图片
# -------------------------------
def get_mapillary_images(lat, lon, radius=SEARCH_RADIUS, limit=5):
    # 使用 bbox 而不是 closeto，bbox 大小约为 1km x 1km
    bbox_size = 0.002  # 约 200m
    left = lon - bbox_size
    bottom = lat - bbox_size
    right = lon + bbox_size
    top = lat + bbox_size
    url = f"https://graph.mapillary.com/images?access_token={ACCESS_TOKEN}&fields=id,thumb_1024_url&bbox={left},{bottom},{right},{top}&limit={limit}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            print(f"API response for {lat},{lon}: {len(data)} images found")
            return data
        else:
            print(f"API error: status {resp.status_code}, response: {resp.text}")
            return []
    except Exception as e:
        print(f"Mapillary API 请求失败: {e}")
        return []

# -------------------------------
# 下载图片并保存 metadata
# -------------------------------
csv_header = ["image", "datazone", "lat", "lon", "mapillary_id", "deprivation_rank", "deprivation_quintile", "deprivation_level"]
seen_ids = load_seen_ids_from_csv(METADATA_CSV)
print(f"Loaded {len(seen_ids)} existing Mapillary IDs from {METADATA_CSV}")

csv_needs_header = (not os.path.exists(METADATA_CSV)) or (os.path.getsize(METADATA_CSV) == 0)
jsonl_needs_file = (not os.path.exists(STREETVIEW_JSONL)) or (os.path.getsize(STREETVIEW_JSONL) == 0)
if jsonl_needs_file:
    os.makedirs(os.path.dirname(STREETVIEW_JSONL), exist_ok=True)

current_next_index = start_index
points_since_checkpoint = 0

try:
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as csvfile, open(STREETVIEW_JSONL, "a", encoding="utf-8") as jsonlfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        if csv_needs_header:
            writer.writeheader()
            csvfile.flush()

        for i in tqdm(range(start_index, len(all_points)), initial=start_index, total=len(all_points)):
            pt = all_points[i]
            lat, lon, dz_code = pt["lat"], pt["lon"], pt["datazone"]

            # 通过预构建的字典快速获取 SIMD 数据
            rank, quintile = zone_lookup.get(dz_code, (None, None))
            deprivation_rank = rank
            deprivation_quintile = quintile
            if quintile is not None:
                deprivation_level = "high deprivation" if quintile <= 2 else "low deprivation"
            else:
                deprivation_level = "unknown"

            images = get_mapillary_images(lat, lon)
            for j, img in enumerate(images):
                map_id = img.get("id")
                if not map_id:
                    continue
                map_id = str(map_id)

                # 跳过已经处理过的图像
                if map_id in seen_ids:
                    continue

                img_url = img.get("thumb_1024_url")
                if not img_url:
                    continue

                try:
                    img_resp = requests.get(img_url, timeout=30)
                    if img_resp.status_code != 200:
                        continue

                    image_name = f"{dz_code}_{i}_{j}.jpg"
                    image_path = os.path.join(IMAGE_DIR, image_name)
                    with open(image_path, "wb") as f:
                        f.write(img_resp.content)

                    row = {
                        "image": image_name,
                        "datazone": dz_code,
                        "lat": lat,
                        "lon": lon,
                        "mapillary_id": map_id,
                        "deprivation_rank": deprivation_rank,
                        "deprivation_quintile": deprivation_quintile,
                        "deprivation_level": deprivation_level
                    }

                    writer.writerow(row)
                    csvfile.flush()
                    seen_ids.add(map_id)

                    # 生成 UrbanLLaVA 对话（写入 JSONL，便于断点续跑）
                    conversation = [
                        {
                            "from": "human",
                            "value": "<image>\nDescribe this urban street scene and analyze the socioeconomic indicators."
                        },
                        {
                            "from": "gpt",
                            "value": f"This is a street view image from Glasgow datazone {dz_code}, which shows {deprivation_level} with SIMD rank {deprivation_rank}. The scene depicts typical urban infrastructure and surroundings that may reflect socioeconomic conditions."
                        }
                    ]

                    streetview_item = {
                        "id": f"glasgow_{dz_code}_{i}_{j}",
                        "image": image_name,
                        "mapillary_id": map_id,
                        "conversations": conversation
                    }
                    jsonlfile.write(json.dumps(streetview_item, ensure_ascii=False) + "\n")
                    jsonlfile.flush()

                except Exception as e:
                    print(f"下载图片失败: {e}")

            current_next_index = i + 1
            points_since_checkpoint += 1
            if points_since_checkpoint >= CHECKPOINT_EVERY_POINTS:
                atomic_write_json(CHECKPOINT_FILE, {
                    "version": 1,
                    "created_at": checkpoint_created_at,
                    "updated_at": _utc_now_iso(),
                    "total_points": len(all_points),
                    "next_point_index": current_next_index,
                    "all_points": all_points,
                    "completed": False,
                })
                points_since_checkpoint = 0
finally:
    # 无论中断/异常，都尽量落盘断点，保证可续跑
    atomic_write_json(CHECKPOINT_FILE, {
        "version": 1,
        "created_at": checkpoint_created_at,
        "updated_at": _utc_now_iso(),
        "total_points": len(all_points),
        "next_point_index": current_next_index,
        "all_points": all_points,
        "completed": current_next_index >= len(all_points),
    })

# -------------------------------
# 保存 metadata 和 UrbanLLaVA 数据
# -------------------------------
metadata = load_metadata_from_csv(METADATA_CSV)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

streetview_data = load_streetview_data_from_jsonl(STREETVIEW_JSONL)
with open(STREETVIEW_JSON, "w", encoding="utf-8") as f:
    json.dump(streetview_data, f, ensure_ascii=False, indent=2)

print(f"下载完成！总共 {len(metadata)} 张街景图，保存在 {IMAGE_DIR}")
print(f"生成了 StreetView 数据集: {STREETVIEW_JSON}")
print(f"实时更新了元数据 CSV: {METADATA_CSV}")
print(f"断点文件: {CHECKPOINT_FILE} (next_point_index={current_next_index}/{len(all_points)})")
