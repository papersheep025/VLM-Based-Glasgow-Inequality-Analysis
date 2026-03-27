# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align original Glasgow street-view images with prefix-deduplicated satellite patches."
    )
    parser.add_argument(
        "--streetview-alignment-csv",
        type=Path,
        default=Path("dataset") / "satellite_dataset" / "streetview_satellite_alignment.csv",
        help="Original street-view/satellite alignment table.",
    )
    parser.add_argument(
        "--prefix-dedup-dir",
        type=Path,
        default=Path("dataset") / "satellite_dataset",
        help="Directory containing prefix-deduplicated satellite patches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset") / "streetview_satellite_aligned",
        help="Directory to write the new alignment table.",
    )
    return parser.parse_args()


def prefix_from_image(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return stem


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = (repo_root / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def main():
    args = parse_args()
    alignment = pd.read_csv(args.streetview_alignment_csv)
    required = {"image", "streetview_path", "datazone", "lat", "lon"}
    missing = required - set(alignment.columns)
    if missing:
        raise ValueError(f"Alignment CSV missing required columns: {sorted(missing)}")

    alignment = alignment.copy()
    alignment["prefix"] = alignment["image"].astype(str).apply(prefix_from_image)
    alignment["streetview_path"] = alignment["streetview_path"].apply(lambda p: str(resolve_path(p, ROOT)))
    alignment["satellite_patch"] = alignment["prefix"].apply(
        lambda p: str((args.prefix_dedup_dir / "satellite_patches" / f"{p}.png").resolve())
    )
    alignment["satellite_exists"] = alignment["satellite_patch"].apply(lambda p: Path(p).exists())
    alignment["streetview_exists"] = alignment["streetview_path"].apply(lambda p: Path(p).exists())

    missing_sat = int((~alignment["satellite_exists"]).sum())
    missing_sv = int((~alignment["streetview_exists"]).sum())
    if missing_sat or missing_sv:
        print(f"Warning: missing streetview={missing_sv}, missing satellite={missing_sat}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "streetview_prefix_satellite_alignment.csv"
    jsonl_path = out_dir / "streetview_prefix_satellite_alignment.jsonl"
    summary_path = out_dir / "streetview_prefix_satellite_summary.json"

    alignment.to_csv(csv_path, index=False, encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in alignment.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "streetview_alignment_csv": str(args.streetview_alignment_csv),
        "prefix_dedup_dir": str(args.prefix_dedup_dir),
        "output_dir": str(out_dir),
        "rows": int(len(alignment)),
        "unique_prefixes": int(alignment["prefix"].nunique()),
        "missing_streetview_images": missing_sv,
        "missing_satellite_patches": missing_sat,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved alignment CSV to: {csv_path}")
    print(f"Saved alignment JSONL to: {jsonl_path}")


if __name__ == "__main__":
    main()
