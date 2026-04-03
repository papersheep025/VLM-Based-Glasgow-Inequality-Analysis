from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


INDICATOR_KEYS = ("income", "employment", "education", "health", "housing", "crime")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def round3(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


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


def get_metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    return to_float(value)


def aggregate_by_datazone(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        datazone = str(record.get("datazone", ""))
        if datazone:
            grouped[datazone].append(record)

    output: list[dict[str, Any]] = []
    for datazone, group in sorted(grouped.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {"datazone": datazone}
        for key in INDICATOR_KEYS:
            pairs = []
            for record in group:
                value = get_metric(record, key)
                if value is not None:
                    # Equal weights within the same datazone; this is a transparent group mean.
                    pairs.append((value, 1.0))
            row[key] = round3(weighted_average(pairs))
        output.append(row)
    return output


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
    fieldnames = ["datazone", *INDICATOR_KEYS]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate processed data to datazone-level CSV/JSON/JSONL.")
    parser.add_argument("--input-jsonl", required=True, help="Input aggregated JSONL file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a datazone_data folder next to the input file.",
    )
    parser.add_argument(
        "--output-name",
        default="datazone_data",
        help="Output base name without extension. Defaults to datazone_data.",
    )
    parser.add_argument("--dedupe-by-id", action="store_true", default=True, help="Drop duplicate ids before aggregation.")
    parser.add_argument("--no-dedupe-by-id", dest="dedupe_by_id", action="store_false", help="Keep duplicate ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "datazone_data"
    output_base = output_dir / args.output_name

    records = load_jsonl(input_path)
    if args.dedupe_by_id:
        records = dedupe_by_id(records)

    rows = aggregate_by_datazone(records)
    write_csv(output_base.with_suffix('.csv'), rows)
    write_json(output_base.with_suffix('.json'), rows)
    write_jsonl(output_base.with_suffix('.jsonl'), rows)
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
