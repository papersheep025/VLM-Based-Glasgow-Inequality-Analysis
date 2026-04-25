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
from PIL import Image
from io import BytesIO

# -------------------------------
# 参数配置
# -------------------------------
MISSING_SHP = "dataset/missing/missing_datazones.shp"
OUTPUT_DIR = "dataset/streetview_dataset_mapillary_missing"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
STREETVIEW_JSON = os.path.join(OUTPUT_DIR, "glasgow_streetview.json")
STREETVIEW_JSONL = os.path.join(OUTPUT_DIR, "glasgow_streetview.jsonl")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_streetview.json")
ACCESS_TOKEN = "MLY|26936598692612655|7d5ba6e8dd0a2d86c57c977fc0d3d3cd"

TARGET_IMAGES_PER_ZONE = 30
CANDIDATE_POINTS_PER_ZONE = 60
IMAGE_SIZE = 336
BBOX_SIZE = 0.002  # ~200m
API_LIMIT = 20
CHECKPOINT_EVERY_ZONES = 1

os.makedirs(IMAGE_DIR, exist_ok=True)


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def generate_random_points(polygon, num_points, seed=None):
    if seed is not None:
        random.seed(seed)
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0
    max_attempts = num_points * 100
    while len(points) < num_points and attempts < max_attempts:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
        attempts += 1
    return points


def get_mapillary_images(lat, lon, bbox_size=BBOX_SIZE, limit=API_LIMIT):
    left = lon - bbox_size
    bottom = lat - bbox_size
    right = lon + bbox_size
    top = lat + bbox_size
    url = (
        f"https://graph.mapillary.com/images"
        f"?access_token={ACCESS_TOKEN}"
        f"&fields=id,geometry,thumb_1024_url,captured_at,compass_angle,sequence"
        f"&bbox={left},{bottom},{right},{top}"
        f"&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            return data
        else:
            print(f"API error: status {resp.status_code}, response: {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"Mapillary API request failed: {e}")
        return []


def download_and_resize_image(url, target_size=IMAGE_SIZE):
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")
        img = img.resize((target_size, target_size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        print(f"  Warning: failed to process image: {e}")
        return None


# -------------------------------
# 从 missing_datazones.shp 加载
# -------------------------------
print(f"Loading missing datazones from {MISSING_SHP}")
gdf = gpd.read_file(MISSING_SHP)
print(f"Loaded {len(gdf)} missing datazones")

rank_col = None
quintile_col = None
for col in gdf.columns:
    if col == "SIMD2020v2" or col.startswith("SIMD2020v2"):
        if "Rank" in col or (rank_col is None and "Quintile" not in col and "Decile" not in col
                             and "Vigintile" not in col and "Percentile" not in col):
            rank_col = col
        if "Quintile" in col or col == "SIMD2020_3":
            quintile_col = col

if rank_col is None:
    rank_col = "SIMD2020v2"
if quintile_col is None:
    quintile_col = "SIMD2020_3"

print(f"Using rank column: {rank_col}, quintile column: {quintile_col}")

missing_datazones = []
dz_polygons = {}
zone_lookup = {}

for _, row in gdf.iterrows():
    dz = row["DataZone"]
    missing_datazones.append(dz)
    dz_polygons[dz] = row.geometry

    rank = row.get(rank_col)
    quintile = row.get(quintile_col)
    try:
        rank = int(float(rank)) if rank is not None else None
    except (ValueError, TypeError):
        rank = None
    try:
        quintile = int(float(quintile)) if quintile is not None else None
    except (ValueError, TypeError):
        quintile = None

    zone_lookup[dz] = (rank, quintile)

print(f"Loaded {len(missing_datazones)} datazones with geometry and SIMD data")

# -------------------------------
# Checkpoint & resume
# -------------------------------
checkpoint = load_checkpoint(CHECKPOINT_FILE)
completed_datazones = set()
resume_dz = None
resume_point_idx = 0
resume_image_count = 0

if checkpoint:
    completed_datazones = set(checkpoint.get("completed_datazones", []))
    resume_dz = checkpoint.get("current_datazone")
    resume_point_idx = checkpoint.get("current_datazone_point_index", 0)
    resume_image_count = checkpoint.get("current_datazone_image_count", 0)
    print(f"Resuming: {len(completed_datazones)} datazones completed, current={resume_dz} at point {resume_point_idx}")

seen_ids = load_seen_ids_from_csv(METADATA_CSV)
print(f"Loaded {len(seen_ids)} existing Mapillary IDs from {METADATA_CSV}")

csv_header = [
    "image", "datazone", "lat", "lon", "mapillary_id",
    "deprivation_rank", "deprivation_quintile", "deprivation_level",
]
csv_needs_header = not os.path.exists(METADATA_CSV) or os.path.getsize(METADATA_CSV) == 0

datazones_to_process = [dz for dz in missing_datazones if dz not in completed_datazones]
zones_completed_this_run = 0
total_api_requests = 0
total_image_downloads = 0

print(f"\nProcessing {len(datazones_to_process)} datazones, target {TARGET_IMAGES_PER_ZONE} images each")
print(f"Candidate points per zone: {CANDIDATE_POINTS_PER_ZONE}")
print(f"Bbox size: ±{BBOX_SIZE}° (~{BBOX_SIZE * 111000:.0f}m)")
print(f"Output image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print()

# -------------------------------
# 主循环：按 datazone 逐个处理
# -------------------------------
try:
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as csvfile, \
         open(STREETVIEW_JSONL, "a", encoding="utf-8") as jsonlfile:

        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        if csv_needs_header:
            writer.writeheader()
            csvfile.flush()

        for dz_code in tqdm(datazones_to_process, desc="Datazones"):
            if dz_code not in dz_polygons:
                print(f"  Warning: no polygon for {dz_code}, skipping")
                continue

            polygon = dz_polygons[dz_code]
            seed = hash(dz_code) & 0xFFFFFFFF
            points = generate_random_points(polygon, CANDIDATE_POINTS_PER_ZONE, seed=seed)

            if len(points) < CANDIDATE_POINTS_PER_ZONE:
                print(f"  Warning: only generated {len(points)}/{CANDIDATE_POINTS_PER_ZONE} points for {dz_code}")

            rank, quintile = zone_lookup.get(dz_code, (None, None))
            if quintile is not None:
                deprivation_level = "high deprivation" if quintile <= 2 else "low deprivation"
            else:
                deprivation_level = "unknown"

            start_idx = 0
            image_count = 0
            if dz_code == resume_dz:
                start_idx = resume_point_idx
                image_count = resume_image_count

            for pt_idx in range(start_idx, len(points)):
                if image_count >= TARGET_IMAGES_PER_ZONE:
                    break

                pt = points[pt_idx]
                lat, lon = pt.y, pt.x

                images = get_mapillary_images(lat, lon)
                total_api_requests += 1

                if not images:
                    continue

                for img in images:
                    if image_count >= TARGET_IMAGES_PER_ZONE:
                        break

                    map_id = img.get("id")
                    if not map_id:
                        continue
                    map_id = str(map_id)

                    if map_id in seen_ids:
                        continue

                    img_url = img.get("thumb_1024_url")
                    if not img_url:
                        continue

                    geometry = img.get("geometry", {})
                    coords = geometry.get("coordinates", [])
                    photo_lon = coords[0] if len(coords) > 0 else lon
                    photo_lat = coords[1] if len(coords) > 1 else lat

                    photo_point = Point(photo_lon, photo_lat)
                    if not polygon.contains(photo_point):
                        continue

                    img_data = download_and_resize_image(img_url)
                    total_image_downloads += 1

                    if img_data is None:
                        continue

                    image_name = f"{dz_code}_{map_id}.jpg"
                    image_path = os.path.join(IMAGE_DIR, image_name)
                    with open(image_path, "wb") as f:
                        f.write(img_data)

                    row = {
                        "image": image_name,
                        "datazone": dz_code,
                        "lat": photo_lat,
                        "lon": photo_lon,
                        "mapillary_id": map_id,
                        "deprivation_rank": rank,
                        "deprivation_quintile": quintile,
                        "deprivation_level": deprivation_level,
                    }
                    writer.writerow(row)
                    csvfile.flush()
                    seen_ids.add(map_id)

                    streetview_item = {
                        "id": f"glasgow_{dz_code}_{map_id}",
                        "image": image_name,
                        "mapillary_id": map_id,
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\nDescribe this urban street scene and analyze the socioeconomic indicators.",
                            },
                            {
                                "from": "gpt",
                                "value": f"This is a street view image from Glasgow datazone {dz_code}, which shows {deprivation_level} with SIMD rank {rank}. The scene depicts typical urban infrastructure and surroundings that may reflect socioeconomic conditions.",
                            },
                        ],
                    }
                    jsonlfile.write(json.dumps(streetview_item, ensure_ascii=False) + "\n")
                    jsonlfile.flush()

                    image_count += 1

                atomic_write_json(CHECKPOINT_FILE, {
                    "version": 2,
                    "created_at": checkpoint.get("created_at", _utc_now_iso()) if checkpoint else _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                    "total_datazones": len(missing_datazones),
                    "completed_datazones": list(completed_datazones),
                    "current_datazone": dz_code,
                    "current_datazone_point_index": pt_idx + 1,
                    "current_datazone_image_count": image_count,
                    "completed": False,
                })

            if image_count < TARGET_IMAGES_PER_ZONE:
                print(f"  Warning: {dz_code} only got {image_count}/{TARGET_IMAGES_PER_ZONE} images")

            completed_datazones.add(dz_code)
            zones_completed_this_run += 1

            if zones_completed_this_run % CHECKPOINT_EVERY_ZONES == 0:
                atomic_write_json(CHECKPOINT_FILE, {
                    "version": 2,
                    "created_at": checkpoint.get("created_at", _utc_now_iso()) if checkpoint else _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                    "total_datazones": len(missing_datazones),
                    "completed_datazones": list(completed_datazones),
                    "current_datazone": None,
                    "current_datazone_point_index": 0,
                    "current_datazone_image_count": 0,
                    "completed": False,
                })

            resume_dz = None

finally:
    atomic_write_json(CHECKPOINT_FILE, {
        "version": 2,
        "created_at": checkpoint.get("created_at", _utc_now_iso()) if checkpoint else _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "total_datazones": len(missing_datazones),
        "completed_datazones": list(completed_datazones),
        "current_datazone": None,
        "current_datazone_point_index": 0,
        "current_datazone_image_count": 0,
        "completed": len(completed_datazones) >= len(missing_datazones),
    })

# -------------------------------
# 保存汇总文件
# -------------------------------
metadata = load_metadata_from_csv(METADATA_CSV)
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

streetview_data = load_streetview_data_from_jsonl(STREETVIEW_JSONL)
with open(STREETVIEW_JSON, "w", encoding="utf-8") as f:
    json.dump(streetview_data, f, ensure_ascii=False, indent=2)

print(f"\nDone! Completed {zones_completed_this_run} datazones this run")
print(f"Total completed: {len(completed_datazones)}/{len(missing_datazones)}")
print(f"API requests: {total_api_requests} searches, {total_image_downloads} image downloads")
print(f"Total images: {len(metadata)}")
print(f"Output: {OUTPUT_DIR}")
