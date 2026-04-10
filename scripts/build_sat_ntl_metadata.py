# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SATELLITE_METADATA_PATH = ROOT / "dataset" / "satellite_dataset" / "satellite_metadata.csv"
DEFAULT_NTL_METADATA_PATH = ROOT / "dataset" / "satellite_dataset" / "ntl_metadata.csv"
DEFAULT_OUTPUT_CSV = ROOT / "dataset" / "satellite_dataset" / "sat_ntl_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge satellite and nightlight metadata into sat_ntl_metadata.csv."
    )
    parser.add_argument(
        "--satellite-metadata-path",
        type=Path,
        default=DEFAULT_SATELLITE_METADATA_PATH,
        help="Path to satellite_metadata.csv.",
    )
    parser.add_argument(
        "--ntl-metadata-path",
        type=Path,
        default=DEFAULT_NTL_METADATA_PATH,
        help="Path to ntl_metadata.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to the merged sat_ntl_metadata.csv.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    satellite_rows = load_csv(args.satellite_metadata_path)
    ntl_rows = load_csv(args.ntl_metadata_path)

    sat_by_id = {row["id"]: row for row in satellite_rows}
    ntl_by_id = {row["id"]: row for row in ntl_rows}

    sat_ids = set(sat_by_id)
    ntl_ids = set(ntl_by_id)
    missing_in_ntl = sorted(sat_ids - ntl_ids)
    missing_in_sat = sorted(ntl_ids - sat_ids)
    if missing_in_ntl or missing_in_sat:
        raise RuntimeError(
            "Satellite and nightlight metadata are not one-to-one.\n"
            f"Missing in ntl: {len(missing_in_ntl)}\n"
            f"Missing in satellite: {len(missing_in_sat)}"
        )

    fieldnames = [
        "id",
        "image",
        "datazone",
        "lat",
        "lon",
        "satellite_patch",
        "ntl_patch",
        "satellite_patch_name",
        "ntl_patch_name",
        "satellite_patch_size_px",
        "ntl_patch_size_px",
        "patch_side_m",
        "bbox_min_lon",
        "bbox_min_lat",
        "bbox_max_lon",
        "bbox_max_lat",
    ]

    merged_rows: list[dict[str, object]] = []
    for row in satellite_rows:
        row_id = row["id"]
        ntl_row = ntl_by_id[row_id]
        core_keys = ["image", "datazone", "lat", "lon", "bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
        mismatched = [key for key in core_keys if row.get(key) != ntl_row.get(key)]
        if mismatched:
            raise RuntimeError(f"Metadata mismatch for id={row_id}: {mismatched}")

        merged_rows.append(
            {
                "id": row_id,
                "image": row["image"],
                "datazone": row["datazone"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "satellite_patch": row["satellite_patch"],
                "ntl_patch": ntl_row["ntl_patch"],
                "satellite_patch_name": row["patch_name"],
                "ntl_patch_name": ntl_row["patch_name"],
                "satellite_patch_size_px": int(row["patch_size_px"]),
                "ntl_patch_size_px": int(ntl_row["patch_size_px"]),
                "patch_side_m": float(row.get("patch_side_m", ntl_row.get("patch_side_m", 333.0))),
                "bbox_min_lon": float(row["bbox_min_lon"]),
                "bbox_min_lat": float(row["bbox_min_lat"]),
                "bbox_max_lon": float(row["bbox_max_lon"]),
                "bbox_max_lat": float(row["bbox_max_lat"]),
            }
        )

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved {len(merged_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
