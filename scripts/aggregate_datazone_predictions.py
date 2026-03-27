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
    parser = argparse.ArgumentParser(description="Aggregate sample predictions to datazone level.")
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--simd-csv", type=Path, default=Path("SIMD") / "simd2020_withgeog" / "simd2020_withinds.csv")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs") / "datazone_predictions.csv")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()
    gold = {row["id"]: row for row in load_jsonl(args.gold_jsonl)}
    preds = load_jsonl(args.pred_jsonl)

    rows = []
    for row in preds:
        gold_row = gold.get(row["id"])
        if gold_row is None:
            continue
        pred = row.get("prediction_json", {})
        rows.append(
            {
                "id": row["id"],
                "datazone": gold_row["datazone"],
                "predicted_quintile": pred.get("predicted_quintile"),
                "predicted_rank_band": pred.get("predicted_rank_band"),
                "prediction_text": row.get("prediction_text"),
                "lat": gold_row.get("lat"),
                "lon": gold_row.get("lon"),
                "deprivation_quintile": gold_row.get("deprivation_quintile"),
                "deprivation_rank": gold_row.get("deprivation_rank"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No predictions were matched to gold samples.")

    summary = (
        df.groupby("datazone")
        .agg(
            n_samples=("id", "count"),
            mean_predicted_quintile=("predicted_quintile", "mean"),
            mean_gold_quintile=("deprivation_quintile", "mean"),
            mean_gold_rank=("deprivation_rank", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
    )

    simd = pd.read_csv(args.simd_csv)[["Data_Zone", "SIMD2020v2_Rank", "SIMD2020v2_Quintile"]]
    summary = summary.merge(simd, left_on="datazone", right_on="Data_Zone", how="left")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"Saved aggregated predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

