# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.windows import Window


DEFAULT_NTL_PATH = Path("dataset") / "satellite_dataset" / "glasgow_ntl" / "glasgow_ntl.tif"
DEFAULT_METADATA_PATH = Path("dataset") / "streetview_dataset" / "metadata.csv"
DEFAULT_IMAGE_DIR = Path("dataset") / "streetview_dataset" / "images"
DEFAULT_OUTPUT_DIR = Path("dataset") / "streetview_ntl_aligned"
DEFAULT_PATCHES_DIR = Path("dataset") / "satellite_dataset" / "satellite_ntl_patches"


def parse_args():
    parser = argparse.ArgumentParser(description="Align Glasgow street-view images to the Glasgow nightlight raster.")
    parser.add_argument(
        "--ntl-path",
        type=Path,
        default=DEFAULT_NTL_PATH,
        help="Path to the georeferenced nightlight TIFF.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Street-view metadata CSV or JSON file with lat/lon/image columns.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing street-view image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store alignment outputs.",
    )
    parser.add_argument(
        "--patches-dir",
        type=Path,
        default=DEFAULT_PATCHES_DIR,
        help="Directory to store saved nightlight patches.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Nightlight patch size in pixels. Must be a positive even number.",
    )
    parser.add_argument(
        "--patch-format",
        choices=("png", "jpg"),
        default="png",
        help="Image format for saved nightlight patches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N rows for testing.",
    )
    parser.add_argument(
        "--skip-patches",
        action="store_true",
        help="Only compute the alignment table, without saving nightlight patches.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing patch files if they already exist.",
    )
    return parser.parse_args()


def load_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported metadata format: {path}")

    required = {"image", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["image", "lat", "lon"]).reset_index(drop=True)
    return df


def ensure_even_positive(value: int) -> int:
    if value <= 0 or value % 2 != 0:
        raise ValueError("--patch-size must be a positive even number.")
    return value


def ensure_uint8_rgb(array):
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")

    bands, _, _ = array.shape
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


def save_patch(dataset, center_row: int, center_col: int, patch_size: int, patch_path: Path):
    half = patch_size // 2
    window = Window(center_col - half, center_row - half, patch_size, patch_size)
    patch = dataset.read(window=window, boundless=True, fill_value=0)
    patch_rgb = ensure_uint8_rgb(patch)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch_rgb).save(patch_path)


def main():
    args = parse_args()
    patch_size = ensure_even_positive(args.patch_size)

    output_dir = args.output_dir
    patches_dir = args.patches_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()

    aligned_rows = []
    missing_image_count = 0
    outside_raster_count = 0

    with rasterio.open(args.ntl_path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Nightlight raster has no CRS: {args.ntl_path}")

        transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        pixel_width = abs(dataset.transform.a)
        pixel_height = abs(dataset.transform.e)

        for _, row in metadata.iterrows():
            image_name = str(row["image"])
            streetview_path = args.image_dir / image_name
            image_exists = streetview_path.exists()
            if not image_exists:
                missing_image_count += 1

            lon = float(row["lon"])
            lat = float(row["lat"])
            x, y = transformer.transform(lon, lat)
            pixel_row, pixel_col = dataset.index(x, y)
            inside_raster = 0 <= pixel_row < dataset.height and 0 <= pixel_col < dataset.width

            patch_relpath = None
            if inside_raster and not args.skip_patches:
                patch_name = f"{Path(image_name).stem}_ntl_{patch_size}.{args.patch_format}"
                patch_path = patches_dir / patch_name
                patch_relpath = str(patch_path.resolve())
                if args.overwrite or not patch_path.exists():
                    save_patch(dataset, pixel_row, pixel_col, patch_size, patch_path)
            elif not inside_raster:
                outside_raster_count += 1

            aligned_rows.append(
                {
                    "image": image_name,
                    "streetview_path": str(streetview_path),
                    "streetview_exists": image_exists,
                    "datazone": row.get("datazone"),
                    "mapillary_id": row.get("mapillary_id"),
                    "lat": lat,
                    "lon": lon,
                    "ntl_crs": str(dataset.crs),
                    "ntl_x": x,
                    "ntl_y": y,
                    "pixel_row": int(pixel_row),
                    "pixel_col": int(pixel_col),
                    "inside_raster": inside_raster,
                    "pixel_size_x_m": pixel_width,
                    "pixel_size_y_m": pixel_height,
                    "patch_size_px": patch_size,
                    "patch_ground_width_m": patch_size * pixel_width,
                    "patch_ground_height_m": patch_size * pixel_height,
                    "satellite_patch": patch_relpath,
                    "secondary_modality": "ntl",
                }
            )

    aligned_df = pd.DataFrame(aligned_rows)
    aligned_csv_path = output_dir / "streetview_ntl_alignment.csv"
    aligned_json_path = output_dir / "streetview_ntl_alignment.json"
    summary_path = output_dir / "alignment_summary.json"

    aligned_df.to_csv(aligned_csv_path, index=False, encoding="utf-8")
    aligned_df.to_json(aligned_json_path, orient="records", force_ascii=False, indent=2)

    summary = {
        "ntl_path": str(args.ntl_path),
        "metadata_path": str(args.metadata_path),
        "image_dir": str(args.image_dir),
        "output_dir": str(output_dir),
        "processed_rows": int(len(aligned_df)),
        "aligned_rows": int(aligned_df["inside_raster"].sum()) if not aligned_df.empty else 0,
        "outside_raster_rows": outside_raster_count,
        "missing_streetview_images": missing_image_count,
        "patch_size_px": patch_size,
        "patch_ground_width_m": patch_size * pixel_width if len(aligned_df) else None,
        "patch_ground_height_m": patch_size * pixel_height if len(aligned_df) else None,
        "skip_patches": bool(args.skip_patches),
        "secondary_modality": "ntl",
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Processed {summary['processed_rows']} rows.")
    print(f"Aligned rows inside raster: {summary['aligned_rows']}")
    print(f"Rows outside raster: {summary['outside_raster_rows']}")
    print(f"Missing street-view images: {summary['missing_streetview_images']}")
    print(f"Alignment table saved to: {aligned_csv_path}")
    print(f"Summary saved to: {summary_path}")
    if not args.skip_patches:
        print(f"Nightlight patches saved under: {patches_dir}")


if __name__ == "__main__":
    main()
