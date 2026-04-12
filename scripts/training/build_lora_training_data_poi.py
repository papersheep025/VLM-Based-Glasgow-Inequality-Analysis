# -*- coding: utf-8 -*-
"""Build HuggingFace Dataset for mlx-vlm LoRA fine-tuning WITH POI context.

Based on build_lora_training_data.py, but injects per-datazone POI context
into the user prompt (via structured_fewshot_poi.format_poi_context).

Usage:
    python scripts/training/build_lora_training_data_poi.py
    python scripts/training/build_lora_training_data_poi.py --max-samples 100
    python scripts/training/build_lora_training_data_poi.py --poi-csv dataset/osm_poi/datazone_poi.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glasgow_vlm.prompts.structured_plus_fewshot import derive_intermediate_scores
from glasgow_vlm.prompts.structured_fewshot_poi import (
    SYSTEM_PROMPT,
    build_prompt,
    load_poi_lookup,
    format_poi_context,
    DEFAULT_POI_CSV,
)

DEFAULT_TRAIN_JSONL = ROOT / "dataset" / "sat_ntl_svi_aligned" / "vlm_data" / "triple_explain_train.jsonl"
DEFAULT_VAL_JSONL = ROOT / "dataset" / "sat_ntl_svi_aligned" / "vlm_data" / "triple_explain_val.jsonl"
DEFAULT_SIMD_CSV = ROOT / "dataset" / "SIMD" / "SIMD_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "lora_training_data_poi"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    p.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    p.add_argument("--simd-csv", type=Path, default=DEFAULT_SIMD_CSV)
    p.add_argument("--poi-csv", type=Path, default=DEFAULT_POI_CSV)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples per split (0=all).")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_simd_scores(csv_path: Path) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    simd = {}
    for _, row in df.iterrows():
        dz = row["datazone"]
        scores = {
            "income": int(row["income_score"]),
            "employment": int(row["employment_score"]),
            "health": int(row["health_score"]),
            "education": int(row["education_score"]),
            "access": int(row["access_score"]),
            "crime": int(row["crime_score"]),
            "housing": int(row["housing_score"]),
            "overall": float(row["overall_score"]),
        }
        simd[dz] = scores
    return simd


def build_target_response(scores: dict) -> str:
    result = {
        "income": scores["income"],
        "employment": scores["employment"],
        "health": scores["health"],
        "education": scores["education"],
        "housing": scores["housing"],
        "access": scores["access"],
        "crime": scores["crime"],
        "overall": scores["overall"],
    }
    result.update(derive_intermediate_scores(scores))
    return json.dumps(result, ensure_ascii=False)


def build_dataset_records(
    records: list[dict],
    simd: dict[str, dict],
    poi_lookup: dict,
    max_samples: int = 0,
) -> list[dict]:
    samples = []
    skipped = 0

    for rec in records:
        dz = rec["datazone"]
        if dz not in simd:
            skipped += 1
            continue

        scores = simd[dz]
        target = build_target_response(scores)

        sv_path = rec["streetview_path"]
        sat_path = rec["satellite_path"]
        ntl_path = rec["ntl_path"]

        for p in (sv_path, sat_path, ntl_path):
            if not Path(p).exists():
                skipped += 1
                break
        else:
            modalities = ("satellite", "ntl")
            poi_ctx = format_poi_context(poi_lookup.get(dz))
            prompt_text = build_prompt(
                rec, "structured_plus",
                modalities=modalities,
                primary_modality="streetview",
                poi_context=poi_ctx,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target},
                    ],
                },
            ]

            samples.append({
                "id": rec["id"],
                "datazone": dz,
                "images": [sv_path, sat_path, ntl_path],
                "messages": messages,
            })

        if max_samples and len(samples) >= max_samples:
            break

    if skipped:
        print(f"  Skipped {skipped} records (missing datazone or image files)")
    return samples


def main():
    args = parse_args()

    print("Loading SIMD scores...")
    simd = load_simd_scores(args.simd_csv)
    print(f"  {len(simd)} datazones loaded")

    print(f"Loading POI lookup from {args.poi_csv}...")
    poi_lookup = load_poi_lookup(args.poi_csv)
    print(f"  {len(poi_lookup)} datazones with POI data")

    print(f"Loading train JSONL: {args.train_jsonl}")
    train_records = load_jsonl(args.train_jsonl)
    print(f"  {len(train_records)} records")

    print(f"Loading val JSONL: {args.val_jsonl}")
    val_records = load_jsonl(args.val_jsonl)
    print(f"  {len(val_records)} records")

    print("Building train samples...")
    train_samples = build_dataset_records(train_records, simd, poi_lookup, args.max_samples)
    print(f"  {len(train_samples)} train samples")

    print("Building val samples...")
    val_samples = build_dataset_records(val_records, simd, poi_lookup, args.max_samples)
    print(f"  {len(val_samples)} val samples")

    poi_hit = sum(1 for s in train_samples + val_samples if poi_lookup.get(s["datazone"]))
    total = len(train_samples) + len(val_samples)
    print(f"  POI coverage: {poi_hit}/{total} samples ({poi_hit/total*100:.1f}%)" if total else "")

    def to_hf_dataset(samples: list[dict]) -> Dataset:
        return Dataset.from_dict({
            "id": [s["id"] for s in samples],
            "datazone": [s["datazone"] for s in samples],
            "images": [s["images"] for s in samples],
            "messages": [json.dumps(s["messages"], ensure_ascii=False) for s in samples],
        })

    print("Creating HuggingFace DatasetDict...")
    ds = DatasetDict({
        "train": to_hf_dataset(train_samples),
        "validation": to_hf_dataset(val_samples),
    })
    print(ds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.output_dir))
    print(f"Saved to {args.output_dir}")

    print("\nSample train record (first):")
    sample = train_samples[0]
    print(f"  id: {sample['id']}")
    print(f"  datazone: {sample['datazone']}")
    print(f"  images: {[Path(p).name for p in sample['images']]}")
    msgs = sample["messages"]
    user_text = ""
    for part in msgs[1]["content"]:
        if isinstance(part, dict) and part.get("type") == "text":
            user_text = part["text"]
    has_poi = "POI context" in user_text
    print(f"  has POI context: {has_poi}")
    print(f"  assistant response: {msgs[-1]['content'][0]['text'][:200]}")


if __name__ == "__main__":
    main()
