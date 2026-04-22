"""
Build the decision-layer dataset by joining perception records with SIMD scores.

Output JSONL schema (one dict per datazone):
  datazone   str
  sat        str   [SAT] text segment
  ntl        str   [NTL] text segment
  sv         str   [SV] text segment
  poi_text   str   [POI_TEXT] text segment
  poi_counts dict  {type: count}
  poi_total  int
  targets    dict  {domain: normalised_logit_float}   (7 values)
  targets_raw dict {domain: int}                       (original 1-10)

CLI:
  python -m decision.data.build_dataset \\
    --perception outputs/perception/qwen3vl_8b_perception.jsonl \\
    --simd dataset/SIMD/SIMD_score.csv \\
    --out outputs/decision/dataset_v0.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from decision.data.parse_perception import load_all
from decision.data.normalize_evidence import build_segments
from decision.data.targets import DOMAINS, normalise_row


def load_simd(path: str | Path) -> dict[str, dict[str, int]]:
    """Return {datazone: {domain: score}} from SIMD_score.csv."""
    simd: dict[str, dict[str, int]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dz = row["datazone"]
            simd[dz] = {d: int(row[d]) for d in DOMAINS if d in row}
    return simd


def build(
    perception_path: str | Path,
    simd_path: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> list[dict]:
    records = load_all(perception_path, limit=limit)
    simd = load_simd(simd_path)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    with open(out_path, "w") as f:
        for rec in records:
            dz = rec["datazone"]
            if dz not in simd:
                dropped.append(dz)
                continue
            scores_raw = simd[dz]
            segments = build_segments(rec)
            sample = {
                "datazone": dz,
                **segments,
                "poi_counts": rec["poi_counts"],
                "poi_total": rec["poi_total"],
                "targets": normalise_row(scores_raw),
                "targets_raw": scores_raw,
            }
            f.write(json.dumps(sample) + "\n")
            kept.append(sample)

    print(f"Wrote {len(kept)} samples → {out_path}")
    if dropped:
        print(f"Dropped {len(dropped)} datazones (no SIMD match): {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    return kept


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", required=True)
    parser.add_argument("--simd", default="dataset/SIMD/SIMD_score.csv")
    parser.add_argument("--out", default="outputs/decision/dataset_v0.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build(args.perception, args.simd, args.out, limit=args.limit)


if __name__ == "__main__":
    _main()
