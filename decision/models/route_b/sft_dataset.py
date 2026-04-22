"""
Turn validated rationales into SFT training rows.

Each output row is a HuggingFace-style chat dict:
    {"datazone": str, "messages": [system, user, assistant]}

where ``assistant`` is the teacher rationalisation string (CoT + JSON scores),
and ``user`` is the prediction-time prompt — no ground truth — so the model
learns to produce the CoT+JSON from evidence alone.

CLI:
    python -m decision.models.route_b.sft_dataset \\
        --dataset outputs/decision/dataset_v0.jsonl \\
        --rationales outputs/decision/route_b/rationales_v0.jsonl \\
        --out outputs/decision/route_b/sft_v0.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision.models.route_b.prompt_template import build_predict_messages


def build(dataset_path: str | Path, rationales_path: str | Path, out_path: str | Path) -> int:
    by_dz = {}
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_dz[r["datazone"]] = r

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(rationales_path) as rf, open(out_path, "w") as wf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dz = row["datazone"]
            sample = by_dz.get(dz)
            if sample is None:
                continue

            predict_msgs = build_predict_messages(sample)
            messages = predict_msgs + [{"role": "assistant", "content": row["completion"]}]
            wf.write(json.dumps({"datazone": dz, "messages": messages}) + "\n")
            n += 1

    print(f"[sft_dataset] wrote {n} rows → {out_path}")
    return n


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--rationales", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build(args.dataset, args.rationales, args.out)


if __name__ == "__main__":
    _main()
