"""
Teacher CoT rationaliser for Route B.

For each training sample:
  1. Show the model the 4-segment evidence + the 7 ground-truth scores.
  2. Ask it to produce the CoT (3 STEPs) that justifies those scores, with
     cite-only constraint on every ``support`` phrase.
  3. Validate:
       - JSON parses,
       - STEP 3 scores match ground truth exactly,
       - every ``support`` string is a verbatim substring of the evidence.

Failed rows are dropped; per-run statistics are printed so the plan's
"cite pass rate ≥ 80%" threshold can be checked.

CLI:
    python -m decision.models.route_b.rationalizer \\
        --config decision/configs/route_b_llm_sft.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from decision.data.targets import DOMAINS
from decision.models.route_b.prompt_template import (
    _format_evidence,
    build_rationalize_messages,
)
from decision.utils.remote import extract_json, load_remote


_DOMAIN_KEYS = [d.lower() for d in DOMAINS]


def _load_dataset(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _collect_supports(obj: Any) -> list[str]:
    """Recursively collect all ``support`` phrase strings from the reasoning dict."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "support" and isinstance(v, list):
                found.extend(str(x) for x in v)
            else:
                found.extend(_collect_supports(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_collect_supports(item))
    return found


def _validate(rationale: dict, sample: dict) -> tuple[bool, str]:
    scores_gt = sample["targets_raw"]
    for k, d in zip(_DOMAIN_KEYS, DOMAINS):
        if int(rationale.get(k, -1)) != int(scores_gt[d]):
            return False, f"score mismatch on {k}"

    evidence_blob = _format_evidence(sample).lower()
    for phrase in _collect_supports(rationale.get("reasoning", {})):
        if phrase and phrase.lower() not in evidence_blob:
            return False, f"uncited phrase: {phrase[:80]}"
    return True, "ok"


def rationalize(cfg: dict) -> dict:
    ds = _load_dataset(cfg["dataset"])
    rat_cfg = cfg.get("rationalize", {})
    llm = load_remote(cfg)

    out_path = Path(rat_cfg.get("out_path", "outputs/decision/route_b/rationales.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_fail = 0
    reasons: dict[str, int] = {}
    with open(out_path, "w") as f:
        for i, sample in enumerate(ds):
            msgs = build_rationalize_messages(sample, sample["targets_raw"])
            try:
                completion = llm.chat(
                    msgs,
                    temperature=rat_cfg.get("temperature", 0.3),
                    max_tokens=rat_cfg.get("max_tokens", 1024),
                )
            except Exception as e:
                n_fail += 1
                reasons["llm_error"] = reasons.get("llm_error", 0) + 1
                print(f"[{i+1}/{len(ds)}] LLM error: {e} ok={n_ok} fail={n_fail}")
                continue

            parsed = extract_json(completion)
            if parsed is None:
                n_fail += 1
                reasons["json_parse"] = reasons.get("json_parse", 0) + 1
                print(f"[{i+1}/{len(ds)}] FAIL (json_parse) ok={n_ok} fail={n_fail} | completion[:200]: {completion[:200]!r}")
                continue

            ok, why = _validate(parsed, sample) if rat_cfg.get("cite_only", True) else (True, "skipped")
            if not ok:
                n_fail += 1
                reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
                print(f"[{i+1}/{len(ds)}] FAIL ({why}) ok={n_ok} fail={n_fail}")
                continue

            f.write(json.dumps({
                "datazone": sample["datazone"],
                "prompt_messages": build_rationalize_messages(sample, sample["targets_raw"]),
                "completion": completion,
                "parsed": parsed,
            }) + "\n")
            f.flush()
            n_ok += 1
            print(f"[{i+1}/{len(ds)}] ok={n_ok} fail={n_fail}")

    total = n_ok + n_fail
    pass_rate = n_ok / total if total else 0.0
    threshold = rat_cfg.get("min_cite_pass_rate", 0.80)
    summary = {
        "total": total,
        "ok": n_ok,
        "fail": n_fail,
        "pass_rate": round(pass_rate, 4),
        "threshold": threshold,
        "failure_reasons": reasons,
        "out_path": str(out_path),
    }
    (out_path.parent / (out_path.stem + "_summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[done] {n_ok}/{total} passed ({pass_rate:.1%}; threshold {threshold:.0%})  → {out_path}")
    if pass_rate < threshold:
        print("[warn] pass rate below threshold — review prompt before SFT.")
    return summary


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    rationalize(cfg)


if __name__ == "__main__":
    _main()
