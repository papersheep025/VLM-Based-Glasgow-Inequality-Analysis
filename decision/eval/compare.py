"""Compare decision-layer OOF runs.

Two usage modes:

1. Direct run directories:

   python -m decision.eval.compare \
     --runs outputs/decision/route_c/modality_sep_v1 \
            outputs/decision/route_c/stacking_v1 \
     --out-csv outputs/evaluation/paper_v1.csv

2. YAML manifest:

   python -m decision.eval.compare \
     --manifest decision/experiments/manifests/paper_v1.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from decision.eval.oof import compare_oof_runs, format_markdown_table


def _run_name(path: Path) -> str:
    if path.name == "oof_predictions.jsonl":
        return path.parent.name
    return path.name


def _oof_path(path: Path) -> Path:
    return path if path.name == "oof_predictions.jsonl" else path / "oof_predictions.jsonl"


def _runs_from_paths(paths: list[Path]) -> list[tuple[str, Path]]:
    return [(_run_name(p), _oof_path(p)) for p in paths]


def _runs_from_manifest(path: Path) -> tuple[list[tuple[str, Path]], Path | None]:
    cfg = yaml.safe_load(path.read_text()) or {}
    base_dir = Path(cfg.get("base_dir", "."))
    runs = []
    for row in cfg.get("runs", []):
        name = row["name"]
        run_path = Path(row["path"])
        if not run_path.is_absolute():
            run_path = base_dir / run_path
        runs.append((name, _oof_path(run_path)))
    out_csv = cfg.get("out_csv")
    if out_csv:
        out_csv = Path(out_csv)
        if not out_csv.is_absolute():
            out_csv = base_dir / out_csv
    return runs, out_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.manifest:
        runs, manifest_out_csv = _runs_from_manifest(args.manifest)
        out_csv = args.out_csv or manifest_out_csv
    else:
        runs = _runs_from_paths(args.runs)
        out_csv = args.out_csv

    if not runs:
        parser.error("Provide --runs or a manifest with at least one run.")

    missing = [str(path) for _, path in runs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing OOF files:\n" + "\n".join(missing))

    df = compare_oof_runs(runs)
    print()
    print(format_markdown_table(df))

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, float_format="%.4f")
        print(f"\n[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
