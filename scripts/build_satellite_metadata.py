# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = ROOT / "dataset" / "streetview_dataset" / "metadata.csv"
DEFAULT_PATCHES_DIR = ROOT / "dataset" / "satellite_dataset" / "satellite_patches"
DEFAULT_OUTPUT_CSV = ROOT / "dataset" / "satellite_dataset" / "satellite_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build satellite_metadata.csv from street-view metadata and satellite patch files."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Street-view metadata CSV containing image, datazone, lat, lon columns.",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=DEFAULT_PATCHES_DIR,
        help="Directory containing satellite patch images named like <streetview_stem>_satellite_384.png.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the generated satellite metadata CSV.",
    )
    parser.add_argument(
        "--patch-side-m",
        type=float,
        default=333.0,
        help="Ground width/height in meters of each square satellite patch.",
    )
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
    min_lat = lat - dlat
    max_lat = lat + dlat
    min_lon = lon - dlon
    max_lon = lon + dlon
    return min_lon, min_lat, max_lon, max_lat


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_path}")
    if not args.patches_dir.exists():
        raise FileNotFoundError(f"Satellite patches directory not found: {args.patches_dir}")

    rows: list[dict[str, object]] = []
    missing_patches: list[str] = []
    with open(args.metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "datazone", "lat", "lon"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Metadata missing required columns: {sorted(missing_columns)}")

        for row in reader:
            image = str(row["image"]).strip()
            datazone = str(row["datazone"]).strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            image_stem = Path(image).stem
            patch_name = f"{image_stem}_satellite_384.png"
            patch_path = args.patches_dir / patch_name
            if not patch_path.exists():
                missing_patches.append(image)
                continue

            min_lon, min_lat, max_lon, max_lat = square_bbox_wgs84(lat, lon, args.patch_side_m)
            rows.append(
                {
                    "id": image_stem,
                    "image": image,
                    "datazone": datazone,
                    "lat": lat,
                    "lon": lon,
                    "satellite_patch": str(patch_path.resolve()),
                    "patch_name": patch_name,
                    "patch_size_px": 384,
                    "patch_side_m": float(args.patch_side_m),
                    "bbox_min_lon": min_lon,
                    "bbox_min_lat": min_lat,
                    "bbox_max_lon": max_lon,
                    "bbox_max_lat": max_lat,
                }
            )

    fieldnames = [
        "id",
        "image",
        "datazone",
        "lat",
        "lon",
        "satellite_patch",
        "patch_name",
        "patch_size_px",
        "patch_side_m",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
    ]
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_csv}")
    if missing_patches:
        print(f"Warning: {len(missing_patches)} satellite patches were missing and skipped.")


if __name__ == "__main__":
    main()
