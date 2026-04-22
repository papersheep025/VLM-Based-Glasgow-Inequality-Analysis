"""
Convert Route A OOF predictions to the CSV schema expected by
``scripts/evaluation/evaluate_domain_scores.py`` and invoke it.

OOF JSONL schema (from cv_runner):
    {
      "datazone": "S01...",
      "fold": 0,
      "prediction_json": {"Income": 5.3, ..., "Housing": 4.1},
      "target_raw":      {"Income": 6.0, ..., "Housing": 5.0}
    }

Output CSV columns (lowercase, matches DOMAIN_MAP_STANDARD):
    datazone, income, employment, health, education, housing, access, crime, overall

`overall` = unweighted mean of the 7 domain predictions (proxy for SIMD overall_score).

CLI:
    python -m decision.eval.run_eval \\
        --oof outputs/decision/route_a/bge_m3_v0/oof_predictions.jsonl \\
        --out-dir outputs/decision/route_a/bge_m3_v0/eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from decision.data.targets import DOMAINS


def oof_to_csv(oof_path: Path, csv_path: Path) -> int:
    rows = []
    with open(oof_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pred = r["prediction_json"]
            row = {"datazone": r["datazone"]}
            for d in DOMAINS:
                row[d.lower()] = float(pred[d])
            row["overall"] = sum(row[d.lower()] for d in DOMAINS) / len(DOMAINS)
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return len(df)


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oof", required=True, help="OOF JSONL from cv_runner")
    p.add_argument("--out-dir", required=True, help="Directory for CSV + evaluation outputs")
    p.add_argument("--simd-csv", default="dataset/SIMD/SIMD_data.csv")
    p.add_argument("--shapefile", default="dataset/glasgow_datazone/glasgow_datazone.shp")
    p.add_argument("--no-spatial", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    pred_csv = out_dir / "oof_predictions.csv"
    report_csv = out_dir / "domain_evaluation_report.csv"

    n = oof_to_csv(Path(args.oof), pred_csv)
    print(f"[run_eval] wrote {n} rows → {pred_csv}")

    cmd = [
        sys.executable,
        "scripts/evaluation/evaluate_domain_scores.py",
        "--pred-csv", str(pred_csv),
        "--simd-csv", args.simd_csv,
        "--shapefile", args.shapefile,
        "--output-csv", str(report_csv),
    ]
    if args.no_spatial:
        cmd.append("--no-spatial")
    print(f"[run_eval] $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    _main()
