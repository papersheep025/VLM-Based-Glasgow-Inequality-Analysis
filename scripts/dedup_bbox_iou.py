"""
Deduplicate satellite/NTL metadata by removing bboxes with IoU >= threshold.

Uses greedy NMS over unique GPS points (each point defines one bbox).
Points with more street-view images are preferred (kept first).

Outputs:
  - dataset/satellite_dataset/satellite_metadata_iou{thresh}.csv
  - dataset/satellite_dataset/sat_ntl_metadata_iou{thresh}.csv
  - dataset/satellite_dataset/satellite_iou{thresh}/   (copies of kept satellite patches)
  - dataset/satellite_dataset/satellite_ntl_iou{thresh}/  (copies of kept NTL patches)

Usage:
    python scripts/dedup_bbox_iou.py --iou-threshold 0.5 [--dry-run]
"""

import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


def compute_iou(b1, b2):
    x_left = max(b1[0], b2[0])
    y_bottom = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_top = min(b1[3], b2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    inter = (x_right - x_left) * (y_top - y_bottom)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def greedy_nms(points_df, iou_threshold):
    order = points_df["count"].argsort()[::-1].values
    bboxes = points_df[["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]].values

    kept = np.zeros(len(points_df), dtype=bool)
    kept_indices = []

    for idx in order:
        b = bboxes[idx]
        suppress = any(compute_iou(b, bboxes[ki]) >= iou_threshold for ki in kept_indices)
        if not suppress:
            kept[idx] = True
            kept_indices.append(idx)

    return kept


def copy_images(patch_names, src_dir, dst_dir, dry_run):
    dst_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for name in patch_names:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            missing += 1
            continue
        if not dry_run:
            shutil.copy2(src, dst)
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    thresh_str = str(args.iou_threshold).replace(".", "")  # "05"
    sat_dir = Path("dataset/satellite_dataset")

    df_sat = pd.read_csv(sat_dir / "satellite_metadata.csv")
    df_ntl = pd.read_csv(sat_dir / "sat_ntl_metadata.csv")

    bbox_cols = ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
    points = (
        df_sat.groupby(["lat", "lon"] + bbox_cols)
        .size()
        .reset_index(name="count")
    )
    print(f"唯一GPS点: {len(points)}")

    kept_mask = greedy_nms(points, args.iou_threshold)
    kept_points = points[kept_mask][["lat", "lon"]].copy()

    print(f"保留: {kept_mask.sum()}  移除: {(~kept_mask).sum()}  (IoU >= {args.iou_threshold})")

    df_sat_out = df_sat.merge(kept_points, on=["lat", "lon"])
    df_ntl_out = df_ntl.merge(kept_points, on=["lat", "lon"])

    print(f"satellite_metadata: {len(df_sat)} → {len(df_sat_out)} 行")
    print(f"sat_ntl_metadata:   {len(df_ntl)} → {len(df_ntl_out)} 行")

    # Output paths
    out_sat_csv = sat_dir / f"satellite_metadata_iou{thresh_str}.csv"
    out_ntl_csv = sat_dir / f"sat_ntl_metadata_iou{thresh_str}.csv"
    out_sat_dir = sat_dir / f"satellite_iou{thresh_str}"
    out_ntl_dir = sat_dir / f"satellite_ntl_iou{thresh_str}"

    if args.dry_run:
        print(f"\n[dry-run] 将写入:\n  {out_sat_csv}\n  {out_ntl_csv}")
        print(f"  图片目录: {out_sat_dir}  ({len(df_sat_out)} 张)")
        print(f"  图片目录: {out_ntl_dir}  ({len(df_ntl_out)} 张)")
        return

    df_sat_out.to_csv(out_sat_csv, index=False)
    df_ntl_out.to_csv(out_ntl_csv, index=False)
    print(f"\n已写入 CSV:\n  {out_sat_csv}\n  {out_ntl_csv}")

    src_sat_dir = sat_dir / "satellite_patches"
    src_ntl_dir = sat_dir / "satellite_ntl_patches"

    print(f"\n复制卫星图像 → {out_sat_dir} ...")
    missing_sat = copy_images(df_sat_out["patch_name"], src_sat_dir, out_sat_dir, dry_run=False)

    print(f"复制夜光图像 → {out_ntl_dir} ...")
    missing_ntl = copy_images(df_ntl_out["ntl_patch_name"], src_ntl_dir, out_ntl_dir, dry_run=False)

    print(f"\n完成。缺失文件: 卫星 {missing_sat} 张，夜光 {missing_ntl} 张")


if __name__ == "__main__":
    main()
