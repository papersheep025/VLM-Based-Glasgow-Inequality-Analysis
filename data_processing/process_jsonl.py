from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


VISUAL_KEYS = (
    "density",
    "greenery",
    "lighting",
    "infrastructure",
    "building_condition",
    "land_use_mix",
    "cleanliness",
    "accessibility",
    "vehicle_presence",
    "housing_type",
    "vacancy",
)

RELATIVE_KEYS = ("density", "greenery", "lighting")
EPS = 1e-8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def get_prediction_json(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("prediction_json", {})
    return value if isinstance(value, dict) else {}


def get_visual_indicators(record: dict[str, Any]) -> dict[str, float | None]:
    prediction_json = get_prediction_json(record)
    raw = prediction_json.get("visual_indicators", {})
    if not isinstance(raw, dict):
        raw = {}
    return {key: to_float(raw.get(key)) for key in VISUAL_KEYS}


def dedupe_by_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        if record_id and record_id in seen:
            continue
        if record_id:
            seen.add(record_id)
        deduped.append(record)
    return deduped


def min_max_scale(value: float | None, values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if value is None:
        return None
    if not clean:
        return None
    lo = min(clean)
    hi = max(clean)
    if hi == lo:
        return 0.5
    scaled = (value - lo) / (hi - lo)
    return max(0.0, min(1.0, scaled))


def round_value(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def compute_spatial_heterogeneity(records: list[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.0

    per_key_stds: list[float] = []
    for key in VISUAL_KEYS:
        values = [get_visual_indicators(record).get(key) for record in records]
        clean = [value for value in values if value is not None]
        if len(clean) < 2:
            continue
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / len(clean)
        per_key_stds.append(variance ** 0.5)

    if not per_key_stds:
        return 0.0

    heterogeneity = (sum(per_key_stds) / len(per_key_stds)) / 0.5
    return max(0.0, min(1.0, heterogeneity))


def build_enriched_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    positions: dict[tuple[str, str], int] = {}

    for record in records:
        datazone = str(record.get("datazone", ""))
        record_id = str(record.get("id", ""))
        positions[(datazone, record_id)] = len(grouped[datazone])
        grouped[datazone].append(record)

    datazone_metrics: dict[str, dict[str, Any]] = {}
    for datazone, group in grouped.items():
        visual_rows = [get_visual_indicators(record) for record in group]
        spatial_heterogeneity = compute_spatial_heterogeneity(group)
        relative_maps: dict[str, list[float | None]] = {}
        for key in RELATIVE_KEYS:
            values = [row.get(key) for row in visual_rows]
            relative_maps[key] = [min_max_scale(value, values) for value in values]
        datazone_metrics[datazone] = {
            "spatial_heterogeneity": spatial_heterogeneity,
            "relative_maps": relative_maps,
        }

    enriched: list[dict[str, Any]] = []
    for record in records:
        datazone = str(record.get("datazone", ""))
        record_id = str(record.get("id", ""))
        pos = positions.get((datazone, record_id), 0)
        metrics = datazone_metrics.get(datazone, {})
        relative_maps = metrics.get("relative_maps", {})
        prediction_json = get_prediction_json(record)

        enriched_record: dict[str, Any] = {
            "id": record.get("id"),
            "datazone": record.get("datazone"),
            "prediction_json": prediction_json,
            "spatial_heterogeneity": round_value(metrics.get("spatial_heterogeneity", 0.0)),
            "relative_density": round_value(relative_maps.get("density", [None])[pos] if relative_maps.get("density") else None),
            "relative_greenery": round_value(relative_maps.get("greenery", [None])[pos] if relative_maps.get("greenery") else None),
            "relative_lighting": round_value(relative_maps.get("lighting", [None])[pos] if relative_maps.get("lighting") else None),
        }

        if "model" in record:
            enriched_record["model"] = record["model"]

        enriched.append(enriched_record)

    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process prediction JSONL into a slim analysis file.")
    parser.add_argument("--input-jsonl", required=True, help="Input prediction JSONL file.")
    parser.add_argument("--output-jsonl", default=None, help="Output enriched JSONL file. Takes priority over output directory/name settings.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the processed file. Defaults to a processed_data folder next to the input file.",
    )
    parser.add_argument(
        "--output-name",
        default="processed_data.jsonl",
        help="Output file name inside the output directory. Defaults to processed_data.jsonl.",
    )
    parser.add_argument("--dedupe-by-id", action="store_true", default=True, help="Drop duplicate ids.")
    parser.add_argument("--no-dedupe-by-id", dest="dedupe_by_id", action="store_false", help="Keep duplicate ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "processed_data"
        output_path = output_dir / args.output_name

    records = load_jsonl(input_path)
    if args.dedupe_by_id:
        records = dedupe_by_id(records)

    enriched = build_enriched_records(records)
    dump_jsonl(output_path, enriched)
    print(f"Wrote {len(enriched)} records to {output_path}")


if __name__ == "__main__":
    main()
