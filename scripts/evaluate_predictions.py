# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glasgow_vlm.metrics import classification_report, regression_report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM predictions.")
    parser.add_argument("--gold-jsonl", type=Path, required=True)
    parser.add_argument("--pred-jsonl", type=Path, required=True)
    parser.add_argument("--task", choices=("ordinal", "rank", "explain"), default="ordinal")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_int(text: str) -> int | None:
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None


def main():
    args = parse_args()
    gold = {row["id"]: row for row in load_jsonl(args.gold_jsonl)}
    preds = load_jsonl(args.pred_jsonl)

    y_true_cls: list[int] = []
    y_pred_cls: list[int] = []
    y_true_reg: list[float] = []
    y_pred_reg: list[float] = []

    for row in preds:
        gold_row = gold.get(row["id"])
        if gold_row is None:
            continue
        gold_q = gold_row.get("deprivation_quintile")
        if gold_q is None:
            continue
        pred_text = row.get("prediction_text", "")
        pred_json = row.get("prediction_json", {})
        pred_q = pred_json.get("predicted_quintile") if isinstance(pred_json, dict) else None
        if pred_q is None:
            pred_q = extract_int(str(pred_text))
        if pred_q is None:
            continue
        y_true_cls.append(int(gold_q))
        y_pred_cls.append(int(pred_q))
        y_true_reg.append(float(gold_q))
        y_pred_reg.append(float(pred_q))

    report = {
        "classification": classification_report(y_true_cls, y_pred_cls),
        "regression": regression_report(y_true_reg, y_pred_reg),
        "n": len(y_true_cls),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

