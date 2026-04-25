"""Aggregate image-level SVF parquet → datazone-level parquet.

Filters applied per image (before aggregation):
    sky < 0.01 and building > 0.6             → drop (likely indoor / facing wall)
    road > 0.5 and sidewalk < 0.01 and building < 0.1
                                                → drop (likely highway)
    other > 0.8                                → drop (blurry / unrecognised)

Aggregation:
    For each datazone, take the **pixel-weighted mean** of all retained images
    (each image contributes the same weight since resolutions match). This
    avoids the ``_extra`` patch over-weighting issue: aggregation happens
    directly at image level, not patch-level first.

Output columns:
    datazone, n_patches, n_images, n_images_filtered,
    <cls>_mean, <cls>_std  for each cls in SVF_COLUMNS
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from perception.segmentation.categories import SVF_COLUMNS


def _filter_images(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cond_indoor = (df["sky"] < 0.01) & (df["building"] > 0.6)
    cond_highway = (df["road"] > 0.5) & (df["sidewalk"] < 0.01) & (df["building"] < 0.1)
    cond_other = df["other"] > 0.8
    bad = cond_indoor | cond_highway | cond_other
    return df[~bad].reset_index(drop=True), int(bad.sum())


def aggregate(image_level: pd.DataFrame) -> pd.DataFrame:
    clean, n_filtered_total = _filter_images(image_level)

    # Per-datazone image count BEFORE filtering (for diagnostic)
    pre_counts = image_level.groupby("datazone").size().rename("n_images_total")
    post_counts = clean.groupby("datazone").size().rename("n_images")
    n_patches = clean.groupby("datazone")["patch_id"].nunique().rename("n_patches")

    grouped = clean.groupby("datazone")[SVF_COLUMNS]
    mean_df = grouped.mean().rename(columns={c: f"{c}_mean" for c in SVF_COLUMNS})
    std_df = grouped.std(ddof=0).fillna(0.0).rename(
        columns={c: f"{c}_std" for c in SVF_COLUMNS}
    )

    out = (
        pd.concat([n_patches, post_counts, pre_counts, mean_df, std_df], axis=1)
        .reset_index()
    )
    out["n_images_filtered"] = (out["n_images_total"] - out["n_images"]).astype(int)
    out = out.drop(columns=["n_images_total"])

    col_order = (
        ["datazone", "n_patches", "n_images", "n_images_filtered"]
        + [f"{c}_mean" for c in SVF_COLUMNS]
        + [f"{c}_std" for c in SVF_COLUMNS]
    )
    print(f"[aggregate] total_images={len(image_level)}  "
          f"filtered={n_filtered_total}  "
          f"datazones_kept={len(out)}")
    return out[col_order]


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-level", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.image_level)
    out = aggregate(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"[aggregate] wrote {len(out)} datazones → {args.output}")


if __name__ == "__main__":
    _main()
