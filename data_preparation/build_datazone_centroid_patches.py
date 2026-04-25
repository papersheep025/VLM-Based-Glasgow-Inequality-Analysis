# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SHAPEFILE = ROOT / "dataset" / "glasgow_datazone" / "glasgow_datazone.shp"
DEFAULT_SAT_TIFF = ROOT / "dataset" / "satellite_dataset" / "TIFF" / "glasgow" / "glasgow.tif"
DEFAULT_NTL_TIFF = ROOT / "dataset" / "satellite_dataset" / "TIFF" / "glasgow_ntl" / "glasgow_ntl.tif"
DEFAULT_SV_METADATA = ROOT / "dataset" / "streetview_dataset" / "metadata.csv"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "datazone_patches"

SAT_RESIZE = (384, 384)
NTL_RESIZE = (256, 256)
PATCH_SIDE_M = 333.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract satellite & NTL patches per datazone centroid, then match street-view images."
    )
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE)
    parser.add_argument("--sat-tiff", type=Path, default=DEFAULT_SAT_TIFF)
    parser.add_argument("--ntl-tiff", type=Path, default=DEFAULT_NTL_TIFF)
    parser.add_argument("--sv-metadata", type=Path, default=DEFAULT_SV_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patch-side-m", type=float, default=PATCH_SIDE_M)
    parser.add_argument("--overwrite", action="store_true")
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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    sat_dir = output_dir / "satellite"
    ntl_dir = output_dir / "ntl"
    sat_dir.mkdir(parents=True, exist_ok=True)
    ntl_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(args.shapefile)
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    dz_col = detect_datazone_column(gdf)
    gdf_proj = gdf.to_crs(epsg=27700)
    centroids_proj = gdf_proj.geometry.centroid
    centroids_wgs84 = centroids_proj.to_crs(epsg=4326)
    gdf["centroid_lat"] = centroids_wgs84.y
    gdf["centroid_lon"] = centroids_wgs84.x

    patch_rows = []
    outside_count = 0

    sat_ds = rasterio.open(args.sat_tiff)
    ntl_ds = rasterio.open(args.ntl_tiff)
    sat_transformer = Transformer.from_crs("EPSG:4326", sat_ds.crs, always_xy=True)
    ntl_transformer = Transformer.from_crs("EPSG:4326", ntl_ds.crs, always_xy=True)

    print(f"Processing {len(gdf)} datazones...")

    for _, row in gdf.iterrows():
        datazone = str(row[dz_col])
        lat = float(row["centroid_lat"])
        lon = float(row["centroid_lon"])

        sat_name = f"{datazone}_satellite.png"
        ntl_name = f"{datazone}_ntl.png"
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
            outside_count += 1
            continue

        min_lon, min_lat, max_lon, max_lat = square_bbox_wgs84(lat, lon, args.patch_side_m)
        patch_rows.append({
            "datazone": datazone,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "satellite_patch": str(sat_path.resolve()) if sat_ok else "",
            "ntl_patch": str(ntl_path.resolve()) if ntl_ok else "",
            "bbox_min_lon": min_lon,
            "bbox_min_lat": min_lat,
            "bbox_max_lon": max_lon,
            "bbox_max_lat": max_lat,
        })

    sat_ds.close()
    ntl_ds.close()

    patch_df = pd.DataFrame(patch_rows)
    patch_csv = output_dir / "datazone_patch_metadata.csv"
    patch_df.to_csv(patch_csv, index=False)
    print(f"Saved {len(patch_df)} datazone patches to {patch_csv}")
    print(f"Skipped {outside_count} datazones (centroid outside TIFF).")

    sv_meta = pd.read_csv(args.sv_metadata)
    patch_dz_set = set(patch_df["datazone"].values)
    sv_with_patch = sv_meta[sv_meta["datazone"].isin(patch_dz_set)].copy()
    unmatched_sv = len(sv_meta) - len(sv_with_patch)

    patch_lookup = patch_df.set_index("datazone")
    alignment_rows = []

    print(f"Matching {len(sv_meta)} street-view images to {len(patch_df)} patches by datazone field...")
    for _, sv_row in sv_with_patch.iterrows():
        dz = sv_row["datazone"]
        p = patch_lookup.loc[dz]
        alignment_rows.append({
            "datazone": dz,
            "satellite_patch": p["satellite_patch"],
            "ntl_patch": p["ntl_patch"],
            "sv_image": sv_row["image"],
            "sv_lat": sv_row["lat"],
            "sv_lon": sv_row["lon"],
        })

    alignment_df = pd.DataFrame(alignment_rows)

    sv_counts = alignment_df.groupby("datazone").size().reset_index(name="sv_count")
    patch_df = patch_df.merge(sv_counts, on="datazone", how="left")
    patch_df["sv_count"] = patch_df["sv_count"].fillna(0).astype(int)
    patch_df["has_streetview"] = patch_df["sv_count"] > 0
    patch_df.to_csv(patch_csv, index=False)

    alignment_csv = output_dir / "datazone_streetview_alignment.csv"
    alignment_df.to_csv(alignment_csv, index=False)

    summary = {
        "total_datazones": len(gdf),
        "patches_created": len(patch_df),
        "outside_tiff": outside_count,
        "datazones_with_streetview": int(patch_df["has_streetview"].sum()),
        "datazones_without_streetview": int((~patch_df["has_streetview"]).sum()),
        "total_sv_matched": len(alignment_df),
        "sv_unmatched_no_patch": unmatched_sv,
        "sv_count_stats": {
            "mean": float(patch_df["sv_count"].mean()),
            "median": float(patch_df["sv_count"].median()),
            "min": int(patch_df["sv_count"].min()),
            "max": int(patch_df["sv_count"].max()),
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSummary:")
    print(f"  Patches created: {summary['patches_created']}")
    print(f"  With street-view: {summary['datazones_with_streetview']}")
    print(f"  Without street-view: {summary['datazones_without_streetview']}")
    print(f"  SV images matched: {summary['total_sv_matched']}")
    print(f"  SV images unmatched (no patch): {summary['sv_unmatched_no_patch']}")
    print(f"  SV per datazone: mean={summary['sv_count_stats']['mean']:.1f}, "
          f"median={summary['sv_count_stats']['median']:.0f}, "
          f"min={summary['sv_count_stats']['min']}, max={summary['sv_count_stats']['max']}")
    print(f"\nOutputs:")
    print(f"  {patch_csv}")
    print(f"  {alignment_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
