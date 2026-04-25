"""Run SegFormer-B2 (ADE20K) over street-view images and dump per-image SVF.

Reads ``dataset/streetview_metadata.csv``, finds the raw 640x640 copy under
``--raw-dir``, runs HF SegFormer in batches, and emits a parquet with columns:

    image_id, patch_id, datazone, pano_index,
    building, sky, tree, road, grass, sidewalk, person, earth, plant, car,
    fence, signboard, streetlight, pole, other

Supports resume: on rerun, rows whose ``image_id`` already exists in the output
parquet are skipped.

CLI:
    python -m perception.segmentation.segformer_infer \
        --metadata dataset/streetview_metadata.csv \
        --raw-dir dataset/streetview_dataset_raw \
        --output outputs/perception/svf/image_level_segformer.parquet \
        --batch-size 16 --device cuda:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from perception.segmentation.categories import ADE20K_KEEP, SVF_COLUMNS


def _image_id(patch_id: str, pano_index: int) -> str:
    return f"{patch_id}_{int(pano_index):04d}"


def _resolve_raw_path(image_path: str, raw_dir: Path) -> Path:
    """Map ``dataset/streetview_dataset/<patch>/<name>.jpg`` → raw_dir variant."""
    p = Path(image_path)
    parts = p.parts
    # Replace the 'streetview_dataset' segment with raw_dir's basename.
    if "streetview_dataset" in parts:
        idx = parts.index("streetview_dataset")
        tail = Path(*parts[idx + 1:])
        return raw_dir / tail
    return raw_dir / p.name


def _load_done_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    try:
        df = pd.read_parquet(output_path, columns=["image_id"])
        return set(df["image_id"].tolist())
    except Exception:
        return set()


def _compute_svf_from_logits(
    pred_ids: torch.Tensor,
    ade_to_col: dict[int, int],
    n_cols: int,
) -> np.ndarray:
    """
    pred_ids: (B, H, W) int64 — class index in [0, 149].
              Both HF SegFormer and mit-semseg PPM output 0-indexed channels
              that align directly with ADE20K_KEEP keys (HF convention).
    Returns (B, n_cols) float32 in [0, 1], columns ordered by SVF_COLUMNS.
    """
    B, H, W = pred_ids.shape
    n_pixels = float(H * W)
    out = np.zeros((B, n_cols), dtype=np.float32)
    flat = pred_ids.view(B, -1)
    for b in range(B):
        counts = torch.bincount(flat[b], minlength=150)  # 0..149 valid
        counts_np = counts.cpu().numpy()
        total_kept = 0.0
        for ade_id, col in ade_to_col.items():
            c = float(counts_np[ade_id]) if ade_id < counts_np.shape[0] else 0.0
            out[b, col] = c / n_pixels
            total_kept += c
        out[b, -1] = max(0.0, 1.0 - total_kept / n_pixels)  # other
    return out


def run(
    metadata_csv: Path,
    raw_dir: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    device: str,
    flush_every: int,
) -> None:
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    meta["image_id"] = [
        _image_id(p, i) for p, i in zip(meta["patch_id"], meta["pano_index"])
    ]
    done = _load_done_ids(output_path)
    todo = meta[~meta["image_id"].isin(done)].reset_index(drop=True)
    print(
        f"[segformer] total={len(meta)} done={len(done)} todo={len(todo)}"
    )
    if len(todo) == 0:
        print("[segformer] nothing to do.")
        return

    print(f"[segformer] loading {model_name} on {device}")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
    ).to(device).eval()

    ade_to_col = {
        ade_id: SVF_COLUMNS.index(name) for ade_id, name in ADE20K_KEEP.items()
    }
    n_cols = len(SVF_COLUMNS)

    buffer_rows: list[dict] = []
    n_missing = 0

    with torch.no_grad():
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start:start + batch_size]
            imgs, keep_rows = [], []
            for _, row in batch.iterrows():
                img_path = _resolve_raw_path(row["image_path"], raw_dir)
                if not img_path.exists():
                    n_missing += 1
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    n_missing += 1
                    continue
                imgs.append(img)
                keep_rows.append(row)
            if not imgs:
                continue

            inputs = processor(images=imgs, return_tensors="pt").to(device)
            if model.dtype == torch.float16:
                inputs = {k: (v.half() if v.dtype == torch.float32 else v)
                          for k, v in inputs.items()}
            outputs = model(**inputs)
            # Upsample logits to input size
            logits = torch.nn.functional.interpolate(
                outputs.logits.float(),
                size=imgs[0].size[::-1],  # (H, W)
                mode="bilinear", align_corners=False,
            )
            pred_ids = logits.argmax(dim=1)   # (B, H, W)
            svf = _compute_svf_from_logits(pred_ids, ade_to_col, n_cols)

            for row, svf_row in zip(keep_rows, svf):
                rec = {
                    "image_id": row["image_id"],
                    "patch_id": row["patch_id"],
                    "datazone": row["datazone"],
                    "pano_index": int(row["pano_index"]),
                }
                for col_name, v in zip(SVF_COLUMNS, svf_row):
                    rec[col_name] = float(v)
                buffer_rows.append(rec)

            if (start // batch_size) % 20 == 0:
                print(f"[segformer] {start + len(imgs)}/{len(todo)} "
                      f"buffered={len(buffer_rows)} missing={n_missing}")

            if len(buffer_rows) >= flush_every:
                _append_parquet(output_path, buffer_rows)
                buffer_rows = []

    if buffer_rows:
        _append_parquet(output_path, buffer_rows)
    print(f"[segformer] done. missing images: {n_missing}")


def _append_parquet(path: Path, rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_parquet(path, index=False)
    print(f"[flush] wrote {len(df)} rows → {path}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="Directory containing the 640x640 originals.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model",
                        default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flush-every", type=int, default=1024)
    args = parser.parse_args()
    run(
        metadata_csv=args.metadata,
        raw_dir=args.raw_dir,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    _main()
