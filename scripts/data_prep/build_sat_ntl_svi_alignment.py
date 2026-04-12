"""
三模态对齐：遥感 + 夜光 + 街景

对齐逻辑：
  - 遥感与夜光一一对应（sat_id 相同）
  - 每个遥感 bbox 与其空间范围内的所有街景图像对应（1对多）

输出：dataset/sat_ntl_svi_aligned/alignment.csv

Usage:
    python scripts/data_prep/build_sat_ntl_svi_alignment.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def main():
    sv = pd.read_csv(BASE / "dataset/streetview_dataset/metadata.csv")
    sat = pd.read_csv(BASE / "dataset/satellite_dataset/sat_dataset/sat_metadata.csv")
    ntl = pd.read_csv(BASE / "dataset/satellite_dataset/ntl_dataset/ntl_metadata.csv")

    sv_img_dir = BASE / "dataset/streetview_dataset/images"

    # sat 和 ntl 按 id 合并（一一对应）
    sat_ntl = sat.merge(
        ntl[["id", "ntl_patch", "patch_name", "patch_size_px"]].rename(
            columns={"patch_name": "ntl_patch_name", "patch_size_px": "ntl_patch_size_px"}
        ),
        on="id",
    )

    records = []
    for _, row in sat_ntl.iterrows():
        mask = (
            (sv["lon"] >= row["bbox_min_lon"]) & (sv["lon"] <= row["bbox_max_lon"]) &
            (sv["lat"] >= row["bbox_min_lat"]) & (sv["lat"] <= row["bbox_max_lat"])
        )
        sv_in_bbox = sv[mask]
        for _, sv_row in sv_in_bbox.iterrows():
            records.append({
                "id": f"{row['id']}__{sv_row['image']}",
                "sat_id": row["id"],
                "datazone": sv_row["datazone"],
                "sv_image": sv_row["image"],
                "sv_path": str(sv_img_dir / sv_row["image"]),
                "sv_lat": sv_row["lat"],
                "sv_lon": sv_row["lon"],
                "mapillary_id": sv_row["mapillary_id"],
                "satellite_patch": row["satellite_patch"],
                "satellite_patch_name": row["patch_name"],
                "satellite_patch_size_px": row["patch_size_px"],
                "ntl_patch": row["ntl_patch"],
                "ntl_patch_name": row["ntl_patch_name"],
                "ntl_patch_size_px": row["ntl_patch_size_px"],
                "patch_side_m": row["patch_side_m"],
                "bbox_min_lon": row["bbox_min_lon"],
                "bbox_min_lat": row["bbox_min_lat"],
                "bbox_max_lon": row["bbox_max_lon"],
                "bbox_max_lat": row["bbox_max_lat"],
                "deprivation_rank": sv_row["deprivation_rank"],
                "deprivation_quintile": sv_row["deprivation_quintile"],
                "deprivation_level": sv_row["deprivation_level"],
            })

    df = pd.DataFrame(records)

    out_dir = BASE / "dataset/sat_ntl_svi_aligned"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "alignment.csv"
    df.to_csv(out_path, index=False)

    print(f"对齐完成：{len(df)} 行")
    print(f"覆盖街景图像：{df['sv_image'].nunique()} 张（共 {len(sv)} 张）")
    print(f"涉及遥感/夜光 patch：{df['sat_id'].nunique()} 个")
    print(f"输出：{out_path}")


if __name__ == "__main__":
    main()
