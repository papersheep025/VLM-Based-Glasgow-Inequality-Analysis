# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DOMAIN_FIELDS = ["income", "employment", "health", "education", "housing", "access", "crime", "overall"]
ENV_FIELDS = [
    "built_environment_quality_score",
    "infrastructure_adequacy_score",
    "residential_environment_quality_score",
    "connectivity_accessibility_score",
    "commercial_economic_activity_score",
    "night_time_activity_coverage_score",
    "open_space_greenery_score",
    "spatial_marginality_score",
]
ALL_FIELDS = DOMAIN_FIELDS + ENV_FIELDS


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate sample-level predictions to datazone level.")
    parser.add_argument("--pred-jsonl", type=Path, required=True, help="Path to prediction JSONL file")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/datazone_domain_scores.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rows = []
    skipped = 0
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pred = record.get("prediction_json")
            if not isinstance(pred, dict):
                skipped += 1
                continue
            row = {"datazone": record["datazone"]}
            for field in ALL_FIELDS:
                val = pred.get(field)
                row[field] = float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else None
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid predictions found in the input file.")

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples across {df['datazone'].nunique()} datazones (skipped {skipped} invalid rows)")

    counts = df.groupby("datazone").size().reset_index(name="n_samples")

    agg = df.groupby("datazone")[ALL_FIELDS].agg(["mean", "median", "std"]).reset_index()
    agg.columns = ["datazone"] + [
        f"{field}__{stat}" for field in ALL_FIELDS for stat in ("mean", "median", "std")
    ]

    summary = counts.merge(agg, on="datazone")

    col_order = ["datazone", "n_samples"]
    for field in ALL_FIELDS:
        mean_col = f"{field}__mean"
        med_col = f"{field}__median"
        std_col = f"{field}__std"
        summary[field] = summary[mean_col]
        summary[f"{field}_median"] = summary[med_col]
        summary[f"{field}_std"] = summary[std_col]
        summary[f"{field}_sem"] = summary[std_col] / np.sqrt(summary["n_samples"])
        col_order += [field, f"{field}_median", f"{field}_std", f"{field}_sem"]

    summary = summary[col_order]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved {len(summary)} datazone-level rows → {args.output_csv}")
    print(summary[["datazone", "n_samples"] + DOMAIN_FIELDS].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
