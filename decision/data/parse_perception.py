"""
Parse perception JSONL into structured per-datazone records.

Output schema (one dict per datazone):
  datazone    str        e.g. "S01009758"
  satellite   list[str]  satellite evidence phrases
  nightlight  list[str]  nightlight evidence phrases
  streetview  list[list[str]]  per-image phrase lists, ordered by streetview_indices
  poi_text    str        original POI summary string
  poi_counts  dict[str,int]  parsed {type: count}; {} if no POI
  poi_total   int        sum of counts; 0 if no POI
"""

from __future__ import annotations

import json
import re
import argparse
from pathlib import Path
from typing import Iterator


_POI_PATTERN = re.compile(r"(\w+)\s*[×x]\s*(\d+)")


def _parse_poi(poi_str: str) -> tuple[dict[str, int], int]:
    """Parse POI summary string → (counts_dict, total)."""
    if not poi_str or "(no POI recorded)" in poi_str:
        return {}, 0
    counts: dict[str, int] = {}
    for m in _POI_PATTERN.finditer(poi_str):
        counts[m.group(1)] = int(m.group(2))
    total = int(m2.group(1)) if (m2 := re.search(r"^(\d+)\s+POIs? total", poi_str)) else sum(counts.values())
    return counts, total


def parse_record(record: dict) -> dict:
    """Parse a single JSONL record into a structured dict."""
    dz = record["datazone"]
    evidence = record["reasoning_json"]["evidence"]
    indices = record.get("streetview_indices", [])

    satellite = evidence.get("satellite", [])
    nightlight = evidence.get("nightlight", [])

    streetview: list[list[str]] = []
    for idx in indices:
        key = f"streetview_{idx:02d}"
        phrases = evidence.get(key, [])
        streetview.append(phrases)

    poi_raw = evidence.get("POI", ["(no POI recorded)"])
    if isinstance(poi_raw, dict):
        poi_counts = {k: v for k, v in poi_raw.items() if k != "total"}
        poi_total = poi_raw.get("total", sum(poi_counts.values()))
        poi_str = f"{poi_total} POIs total: " + ", ".join(
            f"{k} ×{v}" for k, v in poi_counts.items() if v > 0
        ) if poi_total else "(no POI recorded)"
    else:
        poi_str = poi_raw[0] if poi_raw else "(no POI recorded)"
        poi_counts, poi_total = _parse_poi(poi_str)

    return {
        "datazone": dz,
        "satellite": satellite,
        "nightlight": nightlight,
        "streetview": streetview,
        "poi_text": poi_str,
        "poi_counts": poi_counts,
        "poi_total": poi_total,
    }


def load_perception(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    """Yield parsed records from a perception JSONL file."""
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            yield parse_record(json.loads(line))


def load_all(path: str | Path, limit: int | None = None) -> list[dict]:
    return list(load_perception(path, limit=limit))


# ---------------------------------------------------------------------------
# CLI: python -m decision.data.parse_perception --input <path> [--limit N]
# ---------------------------------------------------------------------------

def _main() -> None:
    parser = argparse.ArgumentParser(description="Parse perception JSONL and print structured records.")
    parser.add_argument("--input", required=True, help="Path to perception JSONL")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    for rec in load_perception(args.input, limit=args.limit):
        sv_counts = [len(sv) for sv in rec["streetview"]]
        print(
            f"datazone={rec['datazone']}"
            f"  sat={len(rec['satellite'])}"
            f"  ntl={len(rec['nightlight'])}"
            f"  sv_imgs={len(rec['streetview'])} phrases={sv_counts}"
            f"  poi_total={rec['poi_total']}"
            f"  poi_counts={rec['poi_counts']}"
        )


if __name__ == "__main__":
    _main()
