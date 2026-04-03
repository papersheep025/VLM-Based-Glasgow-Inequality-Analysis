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


def clamp01(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def get_prediction_json(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("prediction_json", {})
    return value if isinstance(value, dict) else {}


def get_visual_indicators(record: dict[str, Any]) -> dict[str, float | None]:
    prediction_json = get_prediction_json(record)
    raw = prediction_json.get("visual_indicators", {})
    if not isinstance(raw, dict):
        raw = {}
    return {key: to_float(raw.get(key)) for key in VISUAL_KEYS}


def get_metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is not None:
        return clamp01(to_float(value))
    prediction_json = get_prediction_json(record)
    nested = prediction_json.get(key)
    return clamp01(to_float(nested))


def inverse(value: float | None) -> float | None:
    if value is None:
        return None
    return clamp01(1.0 - value)


def weighted_average(pairs: list[tuple[float | None, float]]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        total += float(value) * weight
        weight_sum += weight
    if weight_sum == 0.0:
        return None
    return clamp01(total / weight_sum)


def compute_domain_scores(record: dict[str, Any]) -> dict[str, float | None]:
    vis = get_visual_indicators(record)
    density = get_metric(record, "relative_density")
    greenery = get_metric(record, "relative_greenery")
    lighting = get_metric(record, "relative_lighting")
    spatial_heterogeneity = get_metric(record, "spatial_heterogeneity")

    income = weighted_average(
        [
            (vis.get("density"), 0.20),
            (density, 0.20),
            (inverse(vis.get("greenery")), 0.20),
            (inverse(greenery), 0.15),
            (inverse(vis.get("lighting")), 0.15),
            (inverse(vis.get("building_condition")), 0.10),
        ]
    )

    employment = weighted_average(
        [
            (inverse(vis.get("accessibility")), 0.25),
            (inverse(vis.get("infrastructure")), 0.20),
            (inverse(vis.get("land_use_mix")), 0.15),
            (inverse(vis.get("lighting")), 0.15),
            (inverse(lighting), 0.15),
            (spatial_heterogeneity, 0.10),
        ]
    )

    education = weighted_average(
        [
            (inverse(vis.get("accessibility")), 0.25),
            (inverse(vis.get("infrastructure")), 0.20),
            (inverse(vis.get("cleanliness")), 0.20),
            (inverse(vis.get("greenery")), 0.15),
            (inverse(greenery), 0.10),
            (inverse(vis.get("building_condition")), 0.10),
        ]
    )

    health = weighted_average(
        [
            (inverse(vis.get("greenery")), 0.25),
            (inverse(greenery), 0.15),
            (inverse(vis.get("lighting")), 0.20),
            (inverse(lighting), 0.10),
            (inverse(vis.get("cleanliness")), 0.15),
            (vis.get("vacancy"), 0.15),
        ]
    )

    housing = weighted_average(
        [
            (vis.get("density"), 0.25),
            (vis.get("vacancy"), 0.20),
            (inverse(vis.get("building_condition")), 0.20),
            (vis.get("housing_type"), 0.15),
            (inverse(vis.get("greenery")), 0.10),
            (spatial_heterogeneity, 0.10),
        ]
    )

    crime = weighted_average(
        [
            (inverse(vis.get("lighting")), 0.25),
            (inverse(lighting), 0.10),
            (vis.get("vacancy"), 0.20),
            (inverse(vis.get("cleanliness")), 0.20),
            (inverse(vis.get("accessibility")), 0.15),
            (spatial_heterogeneity, 0.10),
        ]
    )

    environment = weighted_average(
        [
            (inverse(vis.get("greenery")), 0.20),
            (inverse(greenery), 0.10),
            (inverse(vis.get("lighting")), 0.20),
            (inverse(lighting), 0.10),
            (inverse(vis.get("cleanliness")), 0.15),
            (inverse(vis.get("infrastructure")), 0.10),
            (vis.get("vacancy"), 0.10),
            (spatial_heterogeneity, 0.05),
        ]
    )

    return {
        "income": round3(income),
        "employment": round3(employment),
        "education": round3(education),
        "health": round3(health),
        "housing": round3(housing),
        "crime": round3(crime),
        "environment": round3(environment),
    }


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


def build_processed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for record in records:
        scores = compute_domain_scores(record)
        processed.append(
            {
                "id": record.get("id"),
                "datazone": record.get("datazone"),
                "income": scores["income"],
                "employment": scores["employment"],
                "education": scores["education"],
                "health": scores["health"],
                "housing": scores["housing"],
                "crime": scores["crime"],
            "environment": scores["environment"],
            }
        )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate processed prediction JSONL into domain scores.")
    parser.add_argument("--input-jsonl", required=True, help="Input processed JSONL file.")
    parser.add_argument("--output-jsonl", default=None, help="Output JSONL file. Takes priority over output directory/name settings.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--output-name",
        default="aggregated_data.jsonl",
        help="Output file name inside the output directory.",
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
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        output_path = output_dir / args.output_name

    records = load_jsonl(input_path)
    if args.dedupe_by_id:
        records = dedupe_by_id(records)

    processed = build_processed_records(records)
    dump_jsonl(output_path, processed)
    print(f"Wrote {len(processed)} records to {output_path}")


if __name__ == "__main__":
    main()
