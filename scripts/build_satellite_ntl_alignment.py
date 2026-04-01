# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SATELLITE_DIR = ROOT / "dataset" / "satellite_dataset" / "satellite_patches"
NTL_DIR = ROOT / "dataset" / "satellite_dataset" / "satellite_ntl_patches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge satellite and nightlight alignments into a paired satellite+nightlight dataset."
    )
    parser.add_argument(
        "--satellite-alignment-csv",
        type=Path,
        default=Path("dataset") / "streetview_satellite_aligned" / "streetview_prefix_satellite_alignment.csv",
        help="Existing street-view + satellite alignment table.",
    )
    parser.add_argument(
        "--ntl-alignment-csv",
        type=Path,
        default=Path("dataset") / "streetview_ntl_aligned" / "streetview_ntl_alignment.csv",
        help="Existing street-view + nightlight alignment table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset") / "satellite_ntl_aligned",
        help="Directory to write the paired satellite+nightlight alignment outputs.",
    )
    return parser.parse_args()



def _resolve(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate.resolve())
    parts = candidate.parts
    if "dataset" in parts:
        suffix = Path(*parts[parts.index("dataset"):])
        mapped = (ROOT / suffix).resolve()
        if mapped.exists():
            return str(mapped)
    return str(candidate.resolve())



def load_satellite_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "satellite_path" not in df.columns and "satellite_patch" in df.columns:
        df = df.rename(columns={"satellite_patch": "satellite_path"})
    df["streetview_path"] = df["streetview_path"].apply(_resolve)
    df["satellite_path"] = df["satellite_path"].apply(lambda p: str((SATELLITE_DIR / Path(p).name).resolve()))
    return df



def load_ntl_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ntl_path" not in df.columns and "satellite_patch" in df.columns:
        df = df.rename(columns={"satellite_patch": "ntl_path"})
    if "streetview_path" in df.columns:
        df["streetview_path"] = df["streetview_path"].apply(_resolve)
    df["ntl_path"] = df["ntl_path"].apply(lambda p: str((NTL_DIR / Path(p).name).resolve()))
    return df



def main() -> None:
    args = parse_args()
    sat_df = load_satellite_df(args.satellite_alignment_csv)
    ntl_df = load_ntl_df(args.ntl_alignment_csv)

    merge_keys = ["image", "datazone", "mapillary_id", "lat", "lon"]
    merged = sat_df.merge(ntl_df, on=merge_keys, how="inner", suffixes=("_satellite", "_ntl"))

    if len(merged) != len(sat_df):
        raise RuntimeError(
            f"Merged rows ({len(merged)}) do not match satellite rows ({len(sat_df)}). "
            "Check whether the satellite and nightlight alignment tables are consistent."
        )

    output = pd.DataFrame(
        {
            "image": merged["image"],
            "streetview_path": merged["streetview_path_satellite"],
            "streetview_exists": merged.get("streetview_exists_satellite", merged.get("streetview_exists", True)),
            "datazone": merged["datazone"],
            "mapillary_id": merged["mapillary_id"],
            "lat": merged["lat"],
            "lon": merged["lon"],
            "satellite_path": merged["satellite_path"],
            "ntl_path": merged["ntl_path"],
            "satellite_crs": merged.get("satellite_crs"),
            "satellite_x": merged.get("satellite_x"),
            "satellite_y": merged.get("satellite_y"),
            "satellite_pixel_row": merged.get("pixel_row_satellite"),
            "satellite_pixel_col": merged.get("pixel_col_satellite"),
            "satellite_inside_raster": merged.get("inside_raster_satellite"),
            "satellite_patch_size_px": merged.get("patch_size_px_satellite"),
            "satellite_patch_ground_width_m": merged.get("patch_ground_width_m_satellite"),
            "satellite_patch_ground_height_m": merged.get("patch_ground_height_m_satellite"),
            "satellite_prefix": merged.get("prefix"),
            "satellite_exists": merged.get("satellite_exists"),
            "ntl_crs": merged.get("ntl_crs"),
            "ntl_x": merged.get("ntl_x"),
            "ntl_y": merged.get("ntl_y"),
            "ntl_pixel_row": merged.get("pixel_row_ntl"),
            "ntl_pixel_col": merged.get("pixel_col_ntl"),
            "ntl_inside_raster": merged.get("inside_raster_ntl"),
            "ntl_patch_size_px": merged.get("patch_size_px_ntl"),
            "ntl_patch_ground_width_m": merged.get("patch_ground_width_m_ntl"),
            "ntl_patch_ground_height_m": merged.get("patch_ground_height_m_ntl"),
            "secondary_modality": "satellite",
            "tertiary_modality": "ntl",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "satellite_ntl_alignment.csv"
    jsonl_path = args.output_dir / "satellite_ntl_alignment.jsonl"
    summary_path = args.output_dir / "satellite_ntl_summary.json"

    output.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in output.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "satellite_alignment_csv": str(args.satellite_alignment_csv),
        "ntl_alignment_csv": str(args.ntl_alignment_csv),
        "output_dir": str(args.output_dir),
        "rows": int(len(output)),
        "streetview_exists_rows": int(output["streetview_exists"].fillna(False).sum()) if "streetview_exists" in output else None,
        "satellite_exists_rows": int(output["satellite_exists"].fillna(False).sum()) if "satellite_exists" in output else None,
        "ntl_inside_raster_rows": int(output["ntl_inside_raster"].fillna(False).sum()) if "ntl_inside_raster" in output else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
