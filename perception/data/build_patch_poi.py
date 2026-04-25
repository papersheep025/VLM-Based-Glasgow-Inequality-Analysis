"""Build patch_poi.csv: POI type counts per patch from datazone_poi.csv.

Output: dataset/poi_dataset/patch_poi.csv
Columns: patch_id, total, amenity, craft, emergency, healthcare, historic,
         leisure, office, public_transport, railway, shop, sport, tourism

Usage:
    python -m perception.data.build_patch_poi
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAT_META = ROOT / "dataset/satellite_dataset/satellite_metadata.csv"
POI_CSV  = ROOT / "dataset/poi_dataset/datazone_poi.csv"
OUT_CSV  = ROOT / "dataset/poi_dataset/patch_poi.csv"

POI_TYPES = [
    "amenity", "craft", "emergency", "healthcare", "historic",
    "leisure", "office", "public_transport", "railway",
    "shop", "sport", "tourism",
]


def main():
    # Load all patch ids from satellite metadata
    patch_ids: list[str] = []
    with open(SAT_META, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            patch_ids.append(r["patch_id"] or r["datazone"])

    # Aggregate poi type counts per datazone (patch_id == datazone here)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(POI_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dz = r["datazone"]
            t  = r.get("poi_type", "").strip()
            if t:
                counts[dz][t] += 1

    # Write output
    fieldnames = ["patch_id", "total"] + POI_TYPES
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pid in patch_ids:
            row_counts = counts.get(pid, {})
            total = sum(row_counts.values())
            row = {"patch_id": pid, "total": total}
            for t in POI_TYPES:
                row[t] = row_counts.get(t, 0)
            writer.writerow(row)

    print(f"Written {len(patch_ids)} rows → {OUT_CSV}")
    no_poi = sum(1 for pid in patch_ids if not counts.get(pid))
    print(f"  {no_poi}/{len(patch_ids)} patches have 0 POIs")


if __name__ == "__main__":
    main()
