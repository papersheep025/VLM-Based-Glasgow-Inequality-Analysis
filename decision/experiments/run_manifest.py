"""Run a YAML experiment manifest.

The manifest intentionally stores shell commands as argument lists, not opaque
strings. That keeps each experiment copy-pastable while avoiding quoting games.

Example:

    python -m decision.experiments.run_manifest \
      --manifest decision/experiments/manifests/paper_v1.yaml \
      --only compare
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _load_manifest(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _selected_experiments(manifest: dict, only: set[str] | None) -> list[dict]:
    experiments = manifest.get("experiments", [])
    if not only:
        return experiments
    return [exp for exp in experiments if exp.get("name") in only]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional experiment names to run.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    base_dir = Path(manifest.get("base_dir", ".")).resolve()
    selected = _selected_experiments(
        manifest,
        set(args.only) if args.only else None,
    )
    compare_only = args.only and set(args.only) == {"compare"}
    if not selected and not compare_only:
        parser.error("No experiments selected.")

    for exp in selected:
        name = exp["name"]
        cmd = exp["cmd"]
        print(f"\n[{name}]")
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=base_dir, check=True)

    compare_cmd = manifest.get("compare_cmd")
    if compare_cmd and (not args.only or "compare" in set(args.only)):
        cmd = [sys.executable if arg == "{python}" else arg for arg in compare_cmd]
        print("\n[compare]")
        print("$ " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=base_dir, check=True)


if __name__ == "__main__":
    main()
