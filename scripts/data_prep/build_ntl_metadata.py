# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SATELLITE_METADATA_PATH = ROOT / "dataset" / "satellite_dataset" / "satellite_metadata.csv"
DEFAULT_PATCHES_DIR = ROOT / "dataset" / "satellite_dataset" / "satellite_ntl_patches"
DEFAULT_OUTPUT_CSV = ROOT / "dataset" / "satellite_dataset" / "ntl_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ntl_metadata.csv from satellite metadata and nightlight patch files."
    )
    parser.add_argument(
        "--satellite-metadata-path",
        type=Path,
        default=DEFAULT_SATELLITE_METADATA_PATH,
        help="Satellite metadata CSV whose bbox can be copied to the nightlight metadata.",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=DEFAULT_PATCHES_DIR,
        help="Directory containing nightlight patch images named like <streetview_stem>_ntl_256.png.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the generated nightlight metadata CSV.",
    )
    parser.add_argument(
        "--patch-size-px",
        type=int,
        default=256,
        help="Pixel size of each nightlight patch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.satellite_metadata_path.exists():
        raise FileNotFoundError(f"Satellite metadata CSV not found: {args.satellite_metadata_path}")
    if not args.patches_dir.exists():
        raise FileNotFoundError(f"Nightlight patches directory not found: {args.patches_dir}")

    rows: list[dict[str, object]] = []
    missing_patches: list[str] = []

    with open(args.satellite_metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "image", "datazone", "lat", "lon", "bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Satellite metadata missing required columns: {sorted(missing_columns)}")

        for row in reader:
            image = str(row["image"]).strip()
            image_stem = Path(image).stem
            patch_name = f"{image_stem}_ntl_{args.patch_size_px}.png"
            patch_path = args.patches_dir / patch_name
            if not patch_path.exists():
                missing_patches.append(image)
                continue

            rows.append(
                {
                    "id": row["id"],
                    "image": image,
                    "datazone": row["datazone"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "ntl_patch": str(patch_path.resolve()),
                    "patch_name": patch_name,
                    "patch_size_px": args.patch_size_px,
                    "patch_side_m": float(row.get("patch_side_m", 333.0)),
                    "bbox_min_lon": float(row["bbox_min_lon"]),
                    "bbox_min_lat": float(row["bbox_min_lat"]),
                    "bbox_max_lon": float(row["bbox_max_lon"]),
                    "bbox_max_lat": float(row["bbox_max_lat"]),
                }
            )

    fieldnames = [
        "id",
        "image",
        "datazone",
        "lat",
        "lon",
        "ntl_patch",
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
        print(f"Warning: {len(missing_patches)} nightlight patches were missing and skipped.")


if __name__ == "__main__":
    main()
