"""Spearman ρ between each VLM domain_indicator and each SIMD domain.

Reads a perception JSONL (with domain_indicators populated) and SIMD_score.csv,
prints a 17×7 ρ table and the strongest indicator→domain pairs. Use the table
to decide which indicators carry signal before running an expensive CV.

Usage:
    python evaluation/indicator_diagnostics.py \
        --perception outputs/perception/qwen3vl_8b_perception_v2.jsonl \
        --simd dataset/SIMD/SIMD_score.csv \
        --out outputs/evaluation/indicator_rho.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision.data.parse_perception import extract_domain_indicators  # noqa: E402
from decision.data.targets import DOMAINS  # noqa: E402
from perception.prompts.perception import INDICATOR_KEYS  # noqa: E402
from src.glasgow_vlm.metrics import spearmanr  # noqa: E402


def load_simd(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            dz = row["datazone"]
            out[dz] = {d: int(row[d]) for d in DOMAINS if d in row}
    return out


def load_indicator_rows(path: Path) -> dict[str, list[float]]:
    """Return {datazone: [score_0, ..., score_16]} for rows with valid indicators."""
    out: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            vec, missing = extract_domain_indicators(row)
            if missing:
                continue
            out[row["datazone"]] = vec.tolist()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--simd", default=ROOT / "dataset/SIMD/SIMD_score.csv", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    indicators = load_indicator_rows(args.perception)
    simd = load_simd(args.simd)
    common = sorted(set(indicators) & set(simd))
    print(f"Indicators present: {len(indicators)}; SIMD: {len(simd)}; common: {len(common)}")
    if not common:
        print("No overlap — nothing to compute.")
        return

    rho: dict[tuple[str, str], float] = {}
    for i, ind_name in enumerate(INDICATOR_KEYS):
        x = [indicators[d][i] for d in common]
        for dom in DOMAINS:
            y = [simd[d][dom] for d in common]
            rho[(ind_name, dom)] = spearmanr(x, y)

    # Print 17 × 7 table
    header = "indicator".ljust(28) + "".join(d.ljust(11) for d in DOMAINS)
    print()
    print(header)
    print("-" * len(header))
    for ind_name in INDICATOR_KEYS:
        row = ind_name.ljust(28) + "".join(
            f"{rho[(ind_name, d)]:+.3f}".ljust(11) for d in DOMAINS
        )
        print(row)

    # Top-K |ρ|
    print()
    print(f"Top {args.top_k} |ρ| pairs:")
    pairs = sorted(rho.items(), key=lambda kv: -abs(kv[1]))[: args.top_k]
    for (ind, dom), r in pairs:
        print(f"  {ind:<28} → {dom:<11} ρ={r:+.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["indicator"] + list(DOMAINS))
            for ind_name in INDICATOR_KEYS:
                w.writerow([ind_name] + [f"{rho[(ind_name, d)]:.4f}" for d in DOMAINS])
        print(f"\nWrote → {args.out}")


if __name__ == "__main__":
    main()
