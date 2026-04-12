# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from glasgow_vlm.prompts import build_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Build VLM JSONL files for Glasgow.")
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=Path("dataset") / "satellite_dataset" / "streetview_satellite_alignment.csv",
    )
    parser.add_argument(
        "--simd-csv",
        type=Path,
        default=Path("SIMD") / "simd2020_withgeog" / "simd2020_withinds.csv",
        help="Deprecated and ignored. Spatial alignment no longer merges SIMD labels.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset") / "vlm_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=("streetview", "satellite", "dual", "satellite_ntl", "triple"), default="dual")
    parser.add_argument("--task", choices=("ordinal", "explain"), default="ordinal")
    parser.add_argument(
        "--secondary-modality",
        choices=("satellite", "ntl"),
        default="satellite",
        help="Name used in the prompt for the second image modality.",
    )
    parser.add_argument(
        "--dedupe-satellite",
        action="store_true",
        help="Keep only one sample per unique satellite patch position to reduce redundancy.",
    )
    parser.add_argument(
        "--dedupe-by",
        choices=("pixel", "patch"),
        default="pixel",
        help="How to identify duplicate satellite samples.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def to_abs(path: str | Path) -> str:
    return str(Path(path).resolve())


def resolve_path_like(alignment_csv: Path, value: str | Path) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return str(candidate.resolve())
        parts = candidate.parts
        if "dataset" in parts:
            suffix = Path(*parts[parts.index("dataset"):])
            mapped = (ROOT / suffix).resolve()
            if mapped.exists():
                return str(mapped)
        return str(candidate.resolve())

    base_candidate = (alignment_csv.parent / candidate).resolve()
    if base_candidate.exists():
        return str(base_candidate)

    return str(candidate.resolve())


def first_present(row: pd.Series, *keys: str, default=None):
    for key in keys:
        if key in row.index and pd.notna(row[key]):
            return row[key]
    return default


def prompt_config_for_mode(input_mode: str, secondary_modality: str) -> tuple[str, str, str | None, tuple[str, ...]]:
    if input_mode == "satellite_ntl":
        return "satellite", "ntl", None, ("ntl",)
    if input_mode == "triple":
        return "streetview", secondary_modality, "ntl", ("satellite", "ntl")
    if input_mode == "satellite":
        return "satellite", secondary_modality, None, ()
    if input_mode == "streetview":
        return "streetview", secondary_modality, None, ()
    return "streetview", secondary_modality, None, (secondary_modality,)


def read_alignment(alignment_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(alignment_csv)
    col_renames: dict[str, str] = {}
    if "sv_path" in df.columns and "streetview_path" not in df.columns:
        col_renames["sv_path"] = "streetview_path"
    if "sv_lat" in df.columns and "lat" not in df.columns:
        col_renames["sv_lat"] = "lat"
    if "sv_lon" in df.columns and "lon" not in df.columns:
        col_renames["sv_lon"] = "lon"
    if "ntl_patch" in df.columns and "ntl_path" not in df.columns:
        col_renames["ntl_patch"] = "ntl_path"
    if "sv_image" in df.columns and "image" not in df.columns:
        col_renames["sv_image"] = "image"
    if col_renames:
        df = df.rename(columns=col_renames)
    if "satellite_patch" not in df.columns and "satellite_path" in df.columns:
        df = df.rename(columns={"satellite_path": "satellite_patch"})
    if "streetview_path" not in df.columns:
        raise ValueError("Alignment CSV missing streetview_path column")
    if "satellite_patch" not in df.columns:
        raise ValueError("Alignment CSV missing satellite_patch or satellite_path column")
    df = df.dropna(subset=["datazone", "streetview_path", "satellite_patch"])
    df["streetview_path"] = df["streetview_path"].apply(lambda p: resolve_path_like(alignment_csv, p))
    df["satellite_path"] = df["satellite_patch"].apply(lambda p: resolve_path_like(alignment_csv, p))
    if "ntl_path" in df.columns:
        df["ntl_path"] = df["ntl_path"].apply(lambda p: resolve_path_like(alignment_csv, p))
    return df


def canonical_satellite_path(output_dir: Path, row: pd.Series, patch_size: int) -> Path:
    dedup_dir = output_dir / "satellite_patches_dedup"
    dedup_dir.mkdir(parents=True, exist_ok=True)
    name = f"px_{int(row['pixel_row'])}_py_{int(row['pixel_col'])}_satellite_{patch_size}.png"
    return dedup_dir / name


def dedupe_alignment_df(df: pd.DataFrame, dedupe_by: str) -> pd.DataFrame:
    if dedupe_by == "pixel":
        key_cols = ["pixel_row", "pixel_col"]
    else:
        key_cols = ["satellite_patch"]
    return df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def make_records(df: pd.DataFrame, task: str, input_mode: str, secondary_modality: str) -> list[dict]:
    records: list[dict] = []
    for _, row in df.iterrows():
        sample_id = row.get("image") if "image" in df.columns else row.get("prefix") or row.get("streetview_source_image")
        pixel_row = first_present(row, "pixel_row", "satellite_pixel_row", "pixel_row_satellite", default=None)
        pixel_col = first_present(row, "pixel_col", "satellite_pixel_col", "pixel_col_satellite", default=None)
        primary_modality, prompt_secondary_modality, tertiary_modality, prompt_modalities = prompt_config_for_mode(input_mode, secondary_modality)
        record = {
            "id": f"{row['datazone']}__{sample_id}",
            "datazone": row["datazone"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "pixel_row": int(pixel_row) if pixel_row is not None else None,
            "pixel_col": int(pixel_col) if pixel_col is not None else None,
            "streetview_path": row["streetview_path"],
            "satellite_path": row["satellite_path"],
            "ntl_path": row["ntl_path"] if "ntl_path" in df.columns and pd.notna(row.get("ntl_path")) else None,
            "secondary_modality": prompt_secondary_modality,
            "input_mode": input_mode,
            "prompt": build_prompt(
                {
                    "datazone": row["datazone"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                },
                task,
                secondary_modality=prompt_secondary_modality,
                tertiary_modality=tertiary_modality,
                modalities=prompt_modalities,
                primary_modality=primary_modality,
            ),
            "task": task,
        }
        if tertiary_modality:
            record["tertiary_modality"] = tertiary_modality
        records.append(record)
    return records


def group_split(
    records: list[dict],
    group_key: str = "datazone",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups: dict[object, list[dict]] = {}
    for record in records:
        group = record.get(group_key)
        groups.setdefault(group, []).append(record)

    group_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_train = min(n_train, n_groups)
    n_val = min(n_val, max(0, n_groups - n_train))

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train : n_train + n_val])
    test_groups = set(group_ids[n_train + n_val :])

    def collect(selected_groups: set[object]) -> list[dict]:
        output: list[dict] = []
        for group in selected_groups:
            output.extend(groups[group])
        return output

    return collect(train_groups), collect(val_groups), collect(test_groups)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    df = read_alignment(args.alignment_csv)
    if args.dedupe_satellite:
        df = dedupe_alignment_df(df, args.dedupe_by)
        patch_size = int(df["patch_size_px"].iloc[0]) if "patch_size_px" in df.columns and len(df) else 256
        for _, row in df.iterrows():
            src = Path(row["satellite_path"])
            dst = canonical_satellite_path(args.output_dir, row, patch_size)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            df.at[row.name, "satellite_path"] = str(dst.resolve())

    records = make_records(df, args.task, args.input_mode, args.secondary_modality)

    train, val, test = group_split(
        records,
        group_key="datazone",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / f"{args.input_mode}_{args.task}_train.jsonl", train)
    write_jsonl(out_dir / f"{args.input_mode}_{args.task}_val.jsonl", val)
    write_jsonl(out_dir / f"{args.input_mode}_{args.task}_test.jsonl", test)
    write_jsonl(out_dir / f"{args.input_mode}_{args.task}_all.jsonl", records)

    summary = {
        "alignment_csv": str(args.alignment_csv),
        "simd_csv": str(args.simd_csv),
        "simd_merged": False,
        "input_mode": args.input_mode,
        "task": args.task,
        "dedupe_satellite": bool(args.dedupe_satellite),
        "dedupe_by": args.dedupe_by if args.dedupe_satellite else None,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "all": len(records),
    }
    with open(out_dir / f"{args.input_mode}_{args.task}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

