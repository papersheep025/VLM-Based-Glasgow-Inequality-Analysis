# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SHAPEFILE = ROOT / "dataset" / "glasgow_datazone" / "glasgow_datazone.shp"
DEFAULT_SAT_TIFF = ROOT / "dataset" / "satellite_dataset" / "TIFF" / "glasgow" / "glasgow.tif"
DEFAULT_NTL_TIFF = ROOT / "dataset" / "satellite_dataset" / "TIFF" / "glasgow_ntl" / "glasgow_ntl.tif"
DEFAULT_SV_METADATA = ROOT / "dataset" / "streetview_dataset" / "metadata.csv"
DEFAULT_PATCH_DIR = ROOT / "dataset" / "datazone_patches"

SAT_RESIZE = (384, 384)
NTL_RESIZE = (256, 256)
PATCH_SIDE_M = 333.0
AREA_RATIO_THRESHOLD = 3.5
AREA_RATIO_UNLIMITED = 6.0
MIN_EXTRA = 1
MAX_EXTRA = 8
MIN_BOUNDARY_DIST_M = 100.0
MIN_BOUNDARY_DIST_M_LARGE = 166.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample extra satellite & NTL patches for large datazones."
    )
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--sat-tiff", type=Path, default=DEFAULT_SAT_TIFF)
    parser.add_argument("--ntl-tiff", type=Path, default=DEFAULT_NTL_TIFF)
    parser.add_argument("--sv-metadata", type=Path, default=DEFAULT_SV_METADATA)
    parser.add_argument("--patch-dir", type=Path, default=DEFAULT_PATCH_DIR)
    parser.add_argument("--patch-side-m", type=float, default=PATCH_SIDE_M)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts, do not write patches.")
    return parser.parse_args()


def meters_per_degree_lat(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return (
        111132.92
        - 559.82 * math.cos(2 * lat)
        + 1.175 * math.cos(4 * lat)
        - 0.0023 * math.cos(6 * lat)
    )


def meters_per_degree_lon(lat_deg: float) -> float:
    lat = math.radians(lat_deg)
    return (
        111412.84 * math.cos(lat)
        - 93.5 * math.cos(3 * lat)
        + 0.118 * math.cos(5 * lat)
    )


def square_bbox_wgs84(lat: float, lon: float, side_m: float) -> tuple[float, float, float, float]:
    half_side = side_m / 2.0
    dlat = half_side / meters_per_degree_lat(lat)
    dlon = half_side / meters_per_degree_lon(lat)
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def bbox_to_shapely(lat: float, lon: float, side_m: float):
    min_lon, min_lat, max_lon, max_lat = square_bbox_wgs84(lat, lon, side_m)
    return box(min_lon, min_lat, max_lon, max_lat)


def ensure_uint8_rgb(array):
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")
    bands = array.shape[0]
    if bands == 1:
        array = array.repeat(3, axis=0)
    elif bands >= 3:
        array = array[:3]
    else:
        raise ValueError(f"Unsupported band count: {bands}")
    if array.dtype != "uint8":
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 0:
            array = array.astype("uint8")
        else:
            scaled = (array.astype("float32") / max_value) * 255.0
            array = scaled.clip(0, 255).astype("uint8")
    return array.transpose(1, 2, 0)


def extract_patch(dataset, transformer, lat: float, lon: float, target_size: tuple[int, int]) -> Image.Image | None:
    x, y = transformer.transform(lon, lat)
    row, col = dataset.index(x, y)
    if not (0 <= row < dataset.height and 0 <= col < dataset.width):
        return None

    pixel_width = abs(dataset.transform.a)
    pixel_height = abs(dataset.transform.e)
    half_w_px = int(round((PATCH_SIDE_M / 2.0) / pixel_width))
    half_h_px = int(round((PATCH_SIDE_M / 2.0) / pixel_height))
    patch_w = half_w_px * 2
    patch_h = half_h_px * 2

    window = Window(col - half_w_px, row - half_h_px, patch_w, patch_h)
    patch = dataset.read(window=window, boundless=True, fill_value=0)
    patch_rgb = ensure_uint8_rgb(patch)
    img = Image.fromarray(patch_rgb)
    if img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


def detect_datazone_column(gdf: gpd.GeoDataFrame) -> str:
    for col in ("DataZone", "datazone", "DZ_CODE", "DZ_code", "dz_code", "DATAZONE"):
        if col in gdf.columns:
            return col
    raise ValueError(f"Cannot detect datazone column. Available: {list(gdf.columns)}")


def find_extra_positions(dz_geom, centroid_lat, centroid_lon, side_m, max_extra, all_bboxes_union, min_boundary_dist_m=MIN_BOUNDARY_DIST_M):
    dlat = side_m / meters_per_degree_lat(centroid_lat)
    dlon = side_m / meters_per_degree_lon(centroid_lat)

    min_bdist_lat = min_boundary_dist_m / meters_per_degree_lat(centroid_lat)
    min_bdist_lon = min_boundary_dist_m / meters_per_degree_lon(centroid_lat)
    min_bdist_deg = (min_bdist_lat + min_bdist_lon) / 2.0

    dz_boundary = dz_geom.boundary

    minx, miny, maxx, maxy = dz_geom.bounds
    step_lat = dlat * 0.5
    step_lon = dlon * 0.5

    candidates = []
    y = miny + dlat / 2
    while y <= maxy - dlat / 2:
        x = minx + dlon / 2
        while x <= maxx - dlon / 2:
            pt = Point(x, y)
            if dz_geom.contains(pt) and dz_boundary.distance(pt) > min_bdist_deg:
                dist = math.hypot(x - centroid_lon, y - centroid_lat)
                candidates.append((x, y, dist))
            x += step_lon
        y += step_lat

    candidates.sort(key=lambda c: c[2])

    placed = []
    placed_bboxes = []
    for cx, cy, _ in candidates:
        if max_extra is not None and len(placed) >= max_extra:
            break
        cbbox = bbox_to_shapely(cy, cx, side_m)
        if cbbox.intersects(all_bboxes_union):
            continue
        intersects_new = any(cbbox.intersects(pb) for pb in placed_bboxes)
        if intersects_new:
            continue
        placed.append((cy, cx))  # (lat, lon)
        placed_bboxes.append(cbbox)

    return placed, placed_bboxes


def main() -> None:
    args = parse_args()
    patch_dir = args.patch_dir
    sat_dir = patch_dir / "satellite"
    ntl_dir = patch_dir / "ntl"

    meta = pd.read_csv(patch_dir / "datazone_patch_metadata.csv")

    gdf = gpd.read_file(args.shapefile)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    dz_col = detect_datazone_column(gdf)

    gdf_proj = gdf.to_crs(epsg=27700)
    gdf_proj["area_m2"] = gdf_proj.geometry.area
    area_map = dict(zip(gdf_proj[dz_col], gdf_proj["area_m2"]))
    geom_map = dict(zip(gdf[dz_col], gdf.geometry))

    bbox_area = args.patch_side_m ** 2

    from shapely.ops import unary_union
    existing_bboxes = []
    for _, r in meta.iterrows():
        existing_bboxes.append(
            box(r["bbox_min_lon"], r["bbox_min_lat"], r["bbox_max_lon"], r["bbox_max_lat"])
        )
    all_bboxes_union = unary_union(existing_bboxes)

    qualifying = []
    for _, r in meta.iterrows():
        dz = r["datazone"]
        area = area_map.get(dz, 0)
        ratio = area / bbox_area
        if ratio >= AREA_RATIO_UNLIMITED:
            qualifying.append((dz, r, ratio, None, MIN_BOUNDARY_DIST_M_LARGE))
        elif ratio >= AREA_RATIO_THRESHOLD:
            qualifying.append((dz, r, ratio, MAX_EXTRA, MIN_BOUNDARY_DIST_M))

    n_unlimited = sum(1 for q in qualifying if q[3] is None)
    n_capped = len(qualifying) - n_unlimited
    print(f"Datazones with area >= {AREA_RATIO_THRESHOLD}x bbox: {len(qualifying)} "
          f"(capped: {n_capped}, unlimited: {n_unlimited})")

    placement_plan = []
    total_new = 0
    for dz, row, ratio, max_extra, bdist in qualifying:
        dz_geom = geom_map[dz]
        positions, new_bboxes = find_extra_positions(
            dz_geom, row["centroid_lat"], row["centroid_lon"],
            args.patch_side_m, max_extra, all_bboxes_union,
            min_boundary_dist_m=bdist,
        )
        if len(positions) >= MIN_EXTRA:
            placement_plan.append((dz, positions, new_bboxes))
            total_new += len(positions)
            for nb in new_bboxes:
                all_bboxes_union = all_bboxes_union.union(nb)

    print(f"Datazones receiving extra patches: {len(placement_plan)}")
    print(f"Total new patches to create: {total_new}")

    if args.dry_run:
        from collections import Counter
        dist = Counter(len(p[1]) for p in placement_plan)
        for k in sorted(dist):
            print(f"  {k} extra patches: {dist[k]} datazones")
        return

    sat_ds = rasterio.open(args.sat_tiff)
    ntl_ds = rasterio.open(args.ntl_tiff)
    sat_transformer = Transformer.from_crs("EPSG:4326", sat_ds.crs, always_xy=True)
    ntl_transformer = Transformer.from_crs("EPSG:4326", ntl_ds.crs, always_xy=True)

    new_rows = []
    created = 0

    sv_meta = pd.read_csv(args.sv_metadata)

    for dz, positions, new_bboxes in placement_plan:
        for idx, ((lat, lon), nbbox) in enumerate(zip(positions, new_bboxes), start=1):
            suffix = f"_extra{idx}"
            sat_name = f"{dz}{suffix}_satellite.png"
            ntl_name = f"{dz}{suffix}_ntl.png"
            sat_path = sat_dir / sat_name
            ntl_path = ntl_dir / ntl_name

            sat_ok = False
            ntl_ok = False

            if args.overwrite or not sat_path.exists():
                sat_img = extract_patch(sat_ds, sat_transformer, lat, lon, SAT_RESIZE)
                if sat_img is not None:
                    sat_img.save(sat_path)
                    sat_ok = True
            else:
                sat_ok = True

            if args.overwrite or not ntl_path.exists():
                ntl_img = extract_patch(ntl_ds, ntl_transformer, lat, lon, NTL_RESIZE)
                if ntl_img is not None:
                    ntl_img.save(ntl_path)
                    ntl_ok = True
            else:
                ntl_ok = True

            if not sat_ok and not ntl_ok:
                continue

            min_lon, min_lat, max_lon, max_lat = square_bbox_wgs84(lat, lon, args.patch_side_m)

            sv_in_bbox = sv_meta[
                (sv_meta["datazone"] == dz)
                & (sv_meta["lon"] >= min_lon)
                & (sv_meta["lon"] <= max_lon)
                & (sv_meta["lat"] >= min_lat)
                & (sv_meta["lat"] <= max_lat)
            ]

            new_rows.append({
                "datazone": dz,
                "patch_id": f"{dz}{suffix}",
                "centroid_lat": lat,
                "centroid_lon": lon,
                "satellite_patch": str(sat_path.resolve()) if sat_ok else "",
                "ntl_patch": str(ntl_path.resolve()) if ntl_ok else "",
                "bbox_min_lon": min_lon,
                "bbox_min_lat": min_lat,
                "bbox_max_lon": max_lon,
                "bbox_max_lat": max_lat,
                "sv_count": len(sv_in_bbox),
                "has_streetview": len(sv_in_bbox) > 0,
            })
            created += 1

    sat_ds.close()
    ntl_ds.close()

    extra_df = pd.DataFrame(new_rows)
    extra_csv = patch_dir / "datazone_extra_patch_metadata.csv"
    extra_df.to_csv(extra_csv, index=False)

    extra_alignment_rows = []
    for _, er in extra_df[extra_df["has_streetview"]].iterrows():
        matched_sv = sv_meta[
            (sv_meta["datazone"] == er["datazone"])
            & (sv_meta["lon"] >= er["bbox_min_lon"])
            & (sv_meta["lon"] <= er["bbox_max_lon"])
            & (sv_meta["lat"] >= er["bbox_min_lat"])
            & (sv_meta["lat"] <= er["bbox_max_lat"])
        ]
        for _, sv_row in matched_sv.iterrows():
            extra_alignment_rows.append({
                "datazone": er["datazone"],
                "patch_id": er["patch_id"],
                "satellite_patch": er["satellite_patch"],
                "ntl_patch": er["ntl_patch"],
                "sv_image": sv_row["image"],
                "sv_lat": sv_row["lat"],
                "sv_lon": sv_row["lon"],
            })

    extra_alignment_df = pd.DataFrame(extra_alignment_rows)
    extra_alignment_csv = patch_dir / "datazone_extra_streetview_alignment.csv"
    extra_alignment_df.to_csv(extra_alignment_csv, index=False)

    summary = {
        "qualifying_datazones": len(qualifying),
        "datazones_with_extra_patches": len(placement_plan),
        "total_extra_patches_created": created,
        "extra_patches_with_streetview": int(extra_df["has_streetview"].sum()),
        "extra_sv_matched": len(extra_alignment_df),
    }
    summary_path = patch_dir / "extra_patches_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Extra patches created: {created}")
    print(f"  With street-view: {summary['extra_patches_with_streetview']}")
    print(f"  SV images matched: {summary['extra_sv_matched']}")
    print(f"\nOutputs:")
    print(f"  {extra_csv}")
    print(f"  {extra_alignment_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
