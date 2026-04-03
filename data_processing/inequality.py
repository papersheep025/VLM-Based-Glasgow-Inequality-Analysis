from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DOMAIN_KEYS = ("income", "employment", "education", "health", "housing", "crime")
DEFAULT_WEIGHTS = {
    "income": 1.0 / 6.0,
    "employment": 1.0 / 6.0,
    "education": 1.0 / 6.0,
    "health": 1.0 / 6.0,
    "housing": 1.0 / 6.0,
    "crime": 1.0 / 6.0,
}


def load_data(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON input must be a list of objects.")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    raise ValueError(f"Unsupported input format: {suffix}")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def dedupe_by_datazone(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        datazone = str(row.get("datazone", ""))
        if datazone and datazone in seen:
            continue
        if datazone:
            seen.add(datazone)
        deduped.append(row)
    return deduped


def weighted_average(values: list[tuple[float | None, float]]) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for value, weight in values:
        if value is None:
            continue
        total += value * weight
        weight_sum += weight
    if weight_sum == 0.0:
        return None
    return total / weight_sum


def get_domain_value(row: dict[str, Any], key: str) -> float | None:
    return to_float(row.get(key))


def compute_inequality_index(row: dict[str, Any], weights: dict[str, float]) -> float | None:
    pairs = [(get_domain_value(row, key), weights.get(key, 0.0)) for key in DOMAIN_KEYS]
    return weighted_average(pairs)


def enrich_rows(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                "datazone": row.get("datazone"),
                "inequality_index": round3(compute_inequality_index(row, weights)),
            }
        )
    return enriched


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["datazone", "inequality_index"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_weights(items: list[str] | None) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_WEIGHTS)
    weights = dict(DEFAULT_WEIGHTS)
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid weight specification: {item}")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in DOMAIN_KEYS:
            raise ValueError(f"Unknown domain key: {key}")
        weights[key] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weight sum must be positive.")
    return {key: value / total for key, value in weights.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a datazone-level inequality index from aggregated indicators.")
    parser.add_argument("--input-file", required=True, help="Input file. Supports .csv, .json, or .jsonl.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to processed_data/inequality_data.")
    parser.add_argument("--output-name", default="inequality_data", help="Output base name without extension.")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=None,
        help="Optional weights like income=1 employment=1 ... . Defaults to equal weights.",
    )
    parser.add_argument("--dedupe-by-datazone", action="store_true", default=True, help="Keep first row per datazone.")
    parser.add_argument("--no-dedupe-by-datazone", dest="dedupe_by_datazone", action="store_false", help="Do not dedupe by datazone.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else Path("processed_data") / "inequality_data"
    output_base = output_dir / args.output_name

    rows = load_data(input_path)
    if args.dedupe_by_datazone:
        rows = dedupe_by_datazone(rows)

    weights = parse_weights(args.weights)
    enriched = enrich_rows(rows, weights)

    write_csv(output_base.with_suffix(".csv"), enriched)
    write_json(output_base.with_suffix(".json"), enriched)
    write_jsonl(output_base.with_suffix(".jsonl"), enriched)
    print(f"Wrote {len(enriched)} rows to {output_dir}")


if __name__ == "__main__":
    main()
