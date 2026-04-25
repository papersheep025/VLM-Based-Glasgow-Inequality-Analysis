"""Shared dataset loading helper for current decision-layer CV runners.

The original Route A CV runner has moved to ``legacy/decision/train/cv_runner.py``.
Current Route C runners still import ``load_dataset`` from this module, so this
small compatibility shim keeps that interface stable without pulling in Route A
dependencies.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_dataset(path: str | Path) -> list[dict]:
    samples: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples
