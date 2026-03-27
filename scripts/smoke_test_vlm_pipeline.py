# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight smoke test for the Glasgow VLM pipeline.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("dataset") / "vlm_data" / "dual_ordinal_train.jsonl",
        help="Path to a VLM JSONL file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to check.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_image(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        return img.width, img.height


def main():
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    rows = load_jsonl(args.jsonl)
    if not rows:
        raise RuntimeError(f"No rows found in {args.jsonl}")

    n = min(args.num_samples, len(rows))
    print(f"Loaded {len(rows)} rows from {args.jsonl}")
    print(f"Checking first {n} samples...")

    errors = []
    for i, row in enumerate(rows[:n], start=1):
        for key in ("id", "datazone", "streetview_path", "satellite_path", "prompt", "answer_json"):
            if key not in row or row[key] in (None, ""):
                errors.append(f"Sample {i}: missing or empty field '{key}'")

        streetview_path = Path(row["streetview_path"])
        satellite_path = Path(row["satellite_path"])
        if not streetview_path.exists():
            errors.append(f"Sample {i}: streetview image not found: {streetview_path}")
        if not satellite_path.exists():
            errors.append(f"Sample {i}: satellite patch not found: {satellite_path}")
        if "ntl_path" in row and row["ntl_path"] not in (None, ""):
            ntl_path = Path(row["ntl_path"])
            if not ntl_path.exists():
                errors.append(f"Sample {i}: nightlight patch not found: {ntl_path}")

        try:
            sv_w, sv_h = check_image(str(streetview_path))
            sat_w, sat_h = check_image(str(satellite_path))
            message = f"[{i}] {row['id']} | datazone={row['datazone']} | streetview={sv_w}x{sv_h} | satellite={sat_w}x{sat_h}"
            if "ntl_path" in row and row["ntl_path"] not in (None, ""):
                ntl_w, ntl_h = check_image(str(Path(row["ntl_path"])))
                message += f" | ntl={ntl_w}x{ntl_h}"
            print(message)
        except Exception as exc:
            errors.append(f"Sample {i}: failed to read image: {exc}")

        try:
            answer = json.loads(row["answer_json"])
            if "predicted_quintile" not in answer and "predicted_rank_band" not in answer:
                errors.append(f"Sample {i}: answer_json does not contain expected prediction field")
        except Exception as exc:
            errors.append(f"Sample {i}: invalid answer_json: {exc}")

    if errors:
        print("\nSmoke test failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
