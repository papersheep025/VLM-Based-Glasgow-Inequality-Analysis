"""
Route B inference: load the SFT'd LLM (via remote LLM server or locally via
adapter), generate the CoT+JSON per datazone, parse scores, fallback to
per-domain train-set mean on parse failure.

Output schema matches project convention so existing evaluators work:
    {datazone, prediction_json: {Income: …, …}, target_raw: {…}}

CLI:
    python -m decision.infer.route_b_predict \\
        --config decision/configs/route_b_llm_sft.yaml \\
        --dataset outputs/decision/dataset_v0.jsonl \\
        --split-col fold      # optional: per-fold OOF style
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from decision.data.targets import DOMAINS
from decision.models.route_b.prompt_template import build_predict_messages
from decision.utils.remote import extract_json, load_remote


_DOMAIN_KEYS = [d.lower() for d in DOMAINS]


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _train_mean(samples: list[dict]) -> dict[str, float]:
    arr = np.array([[s["targets_raw"][d] for d in DOMAINS] for s in samples], dtype=float)
    mu = arr.mean(axis=0).tolist()
    return {d: float(m) for d, m in zip(DOMAINS, mu)}


def _parse_scores(completion: str) -> dict[str, float] | None:
    parsed = extract_json(completion)
    if parsed is None:
        return None
    out = {}
    for key, dom in zip(_DOMAIN_KEYS, DOMAINS):
        v = parsed.get(key)
        if v is None:
            return None
        try:
            out[dom] = float(v)
        except (TypeError, ValueError):
            return None
    return out


def predict(cfg: dict, dataset_path: str | Path, out_path: str | Path | None = None) -> Path:
    samples = _load_jsonl(dataset_path)
    fallback = _train_mean(samples)

    infer_cfg = cfg.get("infer", {})
    llm = load_remote(cfg)

    out_path = Path(out_path or infer_cfg.get("out_path", "outputs/decision/route_b/predictions.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fallback = 0
    total = len(samples)
    print(f"[predict] {total} samples → {out_path}", flush=True)
    with open(out_path, "w") as wf:
        for i, s in enumerate(samples):
            msgs = build_predict_messages(s)
            try:
                completion = llm.chat(
                    msgs,
                    temperature=infer_cfg.get("temperature", 0.0),
                    max_tokens=infer_cfg.get("max_tokens", 1024),
                )
                scores = _parse_scores(completion)
            except Exception as e:
                print(f"[{i+1}/{total}] LLM error: {e}", flush=True)
                scores = None

            if scores is None:
                scores = dict(fallback)
                n_fallback += 1
            else:
                n_ok += 1

            print(f"[{i+1}/{total}] {s['datazone']}  ok={n_ok} fallback={n_fallback}", flush=True)

            wf.write(json.dumps({
                "datazone": s["datazone"],
                "prediction_json": {d: float(round(scores[d], 4)) for d in DOMAINS},
                "target_raw": {d: float(s["targets_raw"][d]) for d in DOMAINS},
            }) + "\n")

    print(f"[done] {n_ok} parsed, {n_fallback} fell back → {out_path}")
    return out_path


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    predict(cfg, args.dataset, args.out)


if __name__ == "__main__":
    _main()
