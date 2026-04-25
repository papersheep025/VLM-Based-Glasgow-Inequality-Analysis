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


def load_centroids(path: str | Path) -> dict[str, tuple[float, float]]:
    """Return {datazone: (easting, northing)} in EPSG:27700 from satellite metadata.

    The CSV stores WGS84 centroid_lat/lon; we project to British National Grid
    so feature units are meters (matches the project's other spatial code).
    """
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    out: dict[str, tuple[float, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dz = row["datazone"]
            try:
                lon = float(row["centroid_lon"])
                lat = float(row["centroid_lat"])
            except (KeyError, ValueError):
                continue
            x, y = transformer.transform(lon, lat)
            out[dz] = (float(x), float(y))
    return out


def build(
    perception_path: str | Path,
    simd_path: str | Path,
    out_path: str | Path,
    satellite_meta_path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    records = load_all(perception_path, limit=limit)
    simd = load_simd(simd_path)
    centroids = load_centroids(satellite_meta_path) if satellite_meta_path else {}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    n_missing_centroid = 0
    n_missing_indicators = 0
    with open(out_path, "w") as f:
        for rec in records:
            dz = rec["datazone"]
            if dz not in simd:
                dropped.append(dz)
                continue
            scores_raw = simd[dz]
            segments = build_segments(rec)
            cx, cy = centroids.get(dz, (0.0, 0.0))
            if dz not in centroids and centroids:
                n_missing_centroid += 1
            if rec.get("indicators_missing", 1):
                n_missing_indicators += 1
            sample = {
                "datazone": dz,
                **segments,
                "poi_counts": rec["poi_counts"],
                "poi_total": rec["poi_total"],
                "indicators_vec": rec.get("indicators_vec", [0.0] * 17),
                "indicators_missing": int(rec.get("indicators_missing", 1)),
                "centroid_x": cx,
                "centroid_y": cy,
                "targets": normalise_row(scores_raw),
                "targets_raw": scores_raw,
            }
            f.write(json.dumps(sample) + "\n")
            kept.append(sample)

    print(f"Wrote {len(kept)} samples → {out_path}")
    if dropped:
        print(f"Dropped {len(dropped)} datazones (no SIMD match): {dropped[:5]}{'...' if len(dropped)>5 else ''}")
    if centroids:
        print(f"Centroids loaded for {len(centroids)} datazones; missing for {n_missing_centroid} kept samples")
    print(f"Indicators missing for {n_missing_indicators}/{len(kept)} samples")
    return kept


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", required=True)
    parser.add_argument("--simd", default="dataset/SIMD/SIMD_score.csv")
    parser.add_argument("--satellite-meta",
                        default="dataset/satellite_dataset/satellite_metadata.csv")
    parser.add_argument("--out", default="outputs/decision/dataset_v0.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    build(args.perception, args.simd, args.out,
          satellite_meta_path=args.satellite_meta, limit=args.limit)


if __name__ == "__main__":
    _main()
