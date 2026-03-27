# -*- coding: utf-8 -*-
import argparse
import csv
import json
import os
from datetime import datetime, timezone

import ee
import geemap
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point

import random


# 记录当前时间
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 原子性写入 JSON（先写 tmp 文件，写完后再替换）
def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# 从 JSON 文件加载（如果文件不存在或无效，则返回 None）
def _load_json(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# 初始化 Earth Engine
def _init_ee(project: str | None, authenticate: bool, auth_code: str | None, auth_mode: str | None) -> None:
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
        return
    except Exception:
        pass

    if authenticate or auth_code:
        ee.Authenticate(authorization_code=auth_code, auth_mode=auth_mode)
        ee.Initialize(project=project) if project else ee.Initialize()
        return

    raise RuntimeError(
        "Earth Engine 初始化失败：需要先授权。\n"
        "运行本脚本加 `--authenticate`（推荐），或用 `--auth-code <code>`。"
    )


# 从 GeoDataFrame 中选择合适的 ID 字段
def _pick_id_field(gdf: gpd.GeoDataFrame, id_field: str | None):
    if id_field and id_field in gdf.columns:
        return id_field
    for candidate in ("DataZone", "Data_Zone", "DZ_CODE", "id", "ID"):
        if candidate in gdf.columns:
            return candidate
    return None


# 将值转换为整数
def _to_int(x):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


# 根据 SIMD quintile 计算 deprivation level
def _deprivation_level_from_quintile(quintile: int | None) -> str:
    if quintile is None:
        return "unknown"
    return "high deprivation" if quintile <= 2 else "low deprivation"


# 构建 SIMD 查找表
def _build_simd_lookup(simd_shp_path: str, simd_csv_path: str) -> dict[str, dict]:
    # 按你的要求：两个文件都读取（shp 用于存在性/字段检查，csv 提供 rank/quintile）
    simd_gdf = gpd.read_file(simd_shp_path)
    if simd_gdf.empty:
        raise ValueError(f"SIMD shapefile is empty: {simd_shp_path}")

    simd_df = pd.read_csv(simd_csv_path)
    if simd_df.empty:
        raise ValueError(f"SIMD csv is empty: {simd_csv_path}")

    if "Data_Zone" not in simd_df.columns:
        raise ValueError("SIMD_CSV missing column: Data_Zone")

    lookup: dict[str, dict] = {}
    for _, row in simd_df.iterrows():
        dz = row.get("Data_Zone")
        if pd.isna(dz):
            continue
        dz = str(dz)
        rank = _to_int(row.get("SIMD2020v2_Rank"))
        quintile = _to_int(row.get("SIMD2020v2_Quintile"))
        lookup[dz] = {
            "deprivation_rank": rank,
            "deprivation_quintile": quintile,
            "deprivation_level": _deprivation_level_from_quintile(quintile),
        }

    return lookup


# 从 CSV 读取已完成的 zone 集合
def _get_datazone_from_row(row: dict) -> str | None:
    # Backward compatible with older CSVs that used 'zone' or 'zone_id'
    for key in ("datazone", "zone", "zone_id", "id", "DataZone", "Data_Zone"):
        v = row.get(key)
        if v is None:
            continue
        v = str(v).strip()
        if v != "":
            return v
    return None


# 从 CSV 读取已完成的 zone 集合
def _load_completed_from_csv(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    completed = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            datazone = _get_datazone_from_row(row)
            if datazone:
                completed.add(str(datazone))
    return completed


# 确保 CSV 文件有正确的表头
def _ensure_csv_header(csv_path: str, fieldnames: list[str], simd_lookup: dict[str, dict] | None = None) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        # header migration if needed
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                existing = next(reader)
            except StopIteration:
                existing = []
        if existing == fieldnames:
            return

        tmp_path = f"{csv_path}.tmp"
        with open(csv_path, "r", encoding="utf-8", newline="") as src, open(
            tmp_path, "w", encoding="utf-8", newline=""
        ) as dst:
            old_reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in old_reader:
                datazone = _get_datazone_from_row(row)
                image = row.get("image") or row.get("file_path")
                if datazone is None:
                    continue
                simd = (simd_lookup or {}).get(datazone, {})
                writer.writerow(
                    {
                        "datazone": datazone,
                        "sample": row.get("sample"),
                        "lat": row.get("lat"),
                        "lon": row.get("lon"),
                        "image": image,
                        "deprivation_rank": row.get("deprivation_rank") or simd.get("deprivation_rank"),
                        "deprivation_quintile": row.get("deprivation_quintile") or simd.get("deprivation_quintile"),
                        "deprivation_level": row.get("deprivation_level") or simd.get("deprivation_level"),
                    }
                )
        os.replace(tmp_path, csv_path)
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _parse_completed_samples_from_csv(csv_path: str) -> set[str]:
    """
    Completed key format: "{zone}__{sample}"
    Only rows with non-empty 'sample' count as completed samples.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    completed = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zone = (_get_datazone_from_row(row) or "").strip()
            sample = (row.get("sample") or "").strip()
            if not zone or sample == "":
                continue
            completed.add(f"{zone}__{sample}")
    return completed


def _pick_sampling_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    Choose a projected CRS for point sampling so buffers are in meters.
    Prefer the shapefile CRS if it's projected; otherwise fall back to EPSG:3857.
    """
    try:
        if gdf.crs is not None and getattr(gdf.crs, "is_projected", False):
            return gdf.crs.to_string()
    except Exception:
        pass
    return "EPSG:3857"


def _generate_random_points_within(geom, n: int, rng: random.Random, max_tries: int = 20000) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    minx, miny, maxx, maxy = geom.bounds
    points: list[Point] = []
    tries = 0
    while len(points) < n and tries < max_tries:
        tries += 1
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        try:
            if geom.contains(p):
                points.append(p)
        except Exception:
            continue
    return points


def main():
    """
    主流程：
    1. 参数解析
    2. 初始化 EE
    3. 读取 shapefile
    4. 构建影像集合
    5. 遍历导出影像（支持断点续跑）
    6. 生成 CSV + JSON
    """
    parser = argparse.ArgumentParser(description="Extract Sentinel-2 RGB chips by shapefile polygons (resumeable).")
    
    # ===== 数据参数 =====
    parser.add_argument("--shp", default="glasgow_datazone/glasgow_datazone.shp", help="Input shapefile path")
    parser.add_argument("--id-field", default=None, help="Polygon id field name (default: auto-detect)")
    
    # ===== Sentinel-2 参数 =====
    parser.add_argument("--dataset-dir", default="satellite_dataset", help="Dataset output directory")
    parser.add_argument("--collection", default="COPERNICUS/S2_SR_HARMONIZED", help="Sentinel-2 collection id")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--cloud", type=float, default=10.0, help="Max CLOUDY_PIXEL_PERCENTAGE")
    
    # ===== 导出参数 =====
    parser.add_argument("--samples-per-zone", type=int, default=10, help="Sample points per DataZone")
    parser.add_argument("--buffer-m", type=float, default=150.0, help="Each image covers ~buffer meters around point")
    parser.add_argument("--scale", type=float, default=10.0, help="Export scale (meters). Sentinel-2 native is 10m")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between downloads (avoid throttling)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit number of zones (debug)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for point sampling")

    # ===== SIMD =====
    parser.add_argument("--simd-shp", default="SIMD/simd2020_withgeog/sc_dz_11.shp", help="SIMD shapefile path")
    parser.add_argument("--simd-csv", default="SIMD/simd2020_withgeog/simd2020_withinds.csv", help="SIMD csv path")


    # ===== Earth Engine =====
    parser.add_argument(
        "--project",
        default="project-5b6009de-697e-410b-b24",
        help="Optional EE project id for ee.Initialize(project=...)",
    )
    parser.add_argument("--authenticate", action="store_true", help="Run ee.Authenticate() if needed")
    parser.add_argument(
        "--auth-code",
        default=None,
        help="Optional EE OAuth authorization code (do not share). If omitted, reads env var EE_AUTH_CODE.",
    )
    parser.add_argument("--auth-mode", default="notebook", help="ee.Authenticate auth_mode (e.g. notebook/colab)")
    args = parser.parse_args()

    # ===== 路径 =====
    images_dir = os.path.join(args.dataset_dir, "images")
    csv_path = os.path.join(args.dataset_dir, "glasgow_satellite.csv")
    json_path = os.path.join(args.dataset_dir, "glasgow_satellite.json")
    checkpoint_path = os.path.join(args.dataset_dir, "checkpoint_satellite.json")

    os.makedirs(images_dir, exist_ok=True)
    
    # ===== SIMD lookup =====
    simd_lookup = _build_simd_lookup(args.simd_shp, args.simd_csv)
    
    # ===== CSV 初始化 =====
    csv_fields = [
        "datazone",
        "sample",
        "lat",
        "lon",
        "image",
        "deprivation_rank",
        "deprivation_quintile",
        "deprivation_level",
    ]
    _ensure_csv_header(csv_path, csv_fields, simd_lookup=simd_lookup)

    # ===== 断点续跑 =====
    checkpoint = _load_json(checkpoint_path) or {}
    completed = set(map(str, checkpoint.get("completed_samples", [])))
    failed = checkpoint.get("failed", {}) if isinstance(checkpoint.get("failed"), dict) else {}

    completed |= _parse_completed_samples_from_csv(csv_path)

    # 读取 shapefile，构建 zone_id 列表
    print("Loading shapefile...")
    gdf = gpd.read_file(args.shp)
    if gdf.empty:
        raise ValueError(f"Shapefile is empty: {args.shp}")

    gdf_wgs84 = gdf.to_crs(epsg=4326)
    if args.limit is not None:
        gdf_wgs84 = gdf_wgs84.iloc[: args.limit].copy()

    id_field = _pick_id_field(gdf_wgs84, args.id_field)
    zone_ids = []
    for i in range(len(gdf_wgs84)):
        zone_ids.append(str(gdf_wgs84.iloc[i][id_field]) if id_field else str(i))

    print(f"Loaded {len(gdf_wgs84)} polygons. id_field={id_field!r}")

    # ===== Earth Engine 初始化 =====
    auth_code = args.auth_code or os.environ.get("EE_AUTH_CODE")
    _init_ee(project=args.project, authenticate=args.authenticate, auth_code=auth_code, auth_mode=args.auth_mode)
    print("Earth Engine initialized.")

    # 构建 Sentinel-2 影像（一次构建，多次按 region 导出）
    fc = geemap.geopandas_to_ee(gdf_wgs84)
    image = (
        ee.ImageCollection(args.collection)
        .filterBounds(fc)
        .filterDate(args.start, args.end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", args.cloud))
        .median()
        .select(["B4", "B3", "B2"])
    )

    # ===== 每个 DataZone 采样 points =====
    sampling_crs = _pick_sampling_crs(gdf)
    gdf_sampling = gdf.to_crs(sampling_crs)
    if args.limit is not None:
        gdf_sampling = gdf_sampling.iloc[: args.limit].copy()

    rng = random.Random(args.seed)
    samples: list[dict] = []
    for i in range(len(gdf_sampling)):
        zone_id = zone_ids[i]
        poly = gdf_sampling.iloc[i].geometry
        pts = _generate_random_points_within(poly, args.samples_per_zone, rng=rng)
        if not pts:
            continue
        pts_wgs84 = gpd.GeoSeries(pts, crs=sampling_crs).to_crs(epsg=4326)
        for j, p in enumerate(pts_wgs84):
            samples.append({"zone": zone_id, "sample": j, "lon": float(p.x), "lat": float(p.y)})

    total = len(samples)
    if total == 0:
        raise RuntimeError("采样点数量为 0：请检查 shapefile 几何是否有效。")

    remaining: list[dict] = []
    for s in samples:
        key = f"{s['zone']}__{s['sample']}"
        out_path = os.path.join(images_dir, f"{s['zone']}_{int(s['sample']):02d}.tif")
        if key in completed and os.path.exists(out_path):
            continue
        remaining.append(s)

    print(f"Resume: completed={len(completed)}/{total}, remaining={len(remaining)}, failed={len(failed)}")

    created_at = checkpoint.get("created_at") or _utc_now_iso()

    with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)

        pbar = tqdm(remaining, total=total, initial=min(len(completed), total))
        for s in pbar:
            zone_id = s["zone"]
            sample_idx = int(s["sample"])
            lat = float(s["lat"])
            lon = float(s["lon"])

            key = f"{zone_id}__{sample_idx}"
            out_path = os.path.join(images_dir, f"{zone_id}_{sample_idx:02d}.tif")

            if key in completed and os.path.exists(out_path):
                continue

            # 以采样点为中心，buffer_m 米，导出其 bounds（方形）
            region = ee.Geometry.Point([lon, lat]).buffer(args.buffer_m).bounds()
            export_img = image.clip(region)

            try:
                geemap.ee_export_image(
                    export_img,
                    filename=out_path,
                    scale=args.scale,
                    region=region,
                    file_per_band=False,
                    verbose=False,
                )
            except Exception as e:
                failed[key] = str(e)
                last_status = "FAILED"
            else:
                completed.add(key)
                failed.pop(key, None)
                last_status = "OK"

                simd = simd_lookup.get(zone_id, {})
                writer.writerow(
                    {
                        "datazone": zone_id,
                        "sample": sample_idx,
                        "lat": lat,
                        "lon": lon,
                        "image": f"images/{zone_id}_{sample_idx:02d}.tif",
                        "deprivation_rank": simd.get("deprivation_rank"),
                        "deprivation_quintile": simd.get("deprivation_quintile"),
                        "deprivation_level": simd.get("deprivation_level"),
                    }
                )
                csvfile.flush()

            pbar.set_postfix(ok=len(completed), failed=len(failed), status=last_status, zone=zone_id, sample=sample_idx)

            _atomic_write_json(
                checkpoint_path,
                {
                    "version": 2,
                    "created_at": created_at,
                    "updated_at": _utc_now_iso(),
                    "shp": args.shp,
                    "id_field": id_field,
                    "simd_shp": args.simd_shp,
                    "simd_csv": args.simd_csv,
                    "samples_per_zone": args.samples_per_zone,
                    "buffer_m": args.buffer_m,
                    "scale": args.scale,
                    "seed": args.seed,
                    "total": total,
                    "completed_samples": sorted(completed),
                    "failed": failed,
                },
            )

            if args.sleep and args.sleep > 0:
                import time

                time.sleep(args.sleep)

    # json 以 csv 为准（支持断点续跑时去重/汇总）
    records = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            zone = _get_datazone_from_row(row)
            sample = row.get("sample")
            image_path = row.get("image")
            if not zone or sample is None or str(sample).strip() == "" or not image_path:
                continue
            key = f"{zone}__{str(sample).strip()}"
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": key,
                    "datazone": zone,
                    "sample": _to_int(sample),
                    "lat": float(row.get("lat")) if row.get("lat") not in (None, "") else None,
                    "lon": float(row.get("lon")) if row.get("lon") not in (None, "") else None,
                    "image": image_path,
                    "deprivation_rank": _to_int(row.get("deprivation_rank")),
                    "deprivation_quintile": _to_int(row.get("deprivation_quintile")),
                    "deprivation_level": row.get("deprivation_level") or _deprivation_level_from_quintile(None),
                }
            )

    _atomic_write_json(json_path, records)

    print(f"Done. dataset_dir={args.dataset_dir}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Checkpoint: {checkpoint_path}")
    if failed:
        print(f"Warning: {len(failed)} zones failed. See checkpoint for details.")


if __name__ == "__main__":
    main()
