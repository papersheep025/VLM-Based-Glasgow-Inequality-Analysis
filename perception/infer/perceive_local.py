# -*- coding: utf-8 -*-
"""Perception-layer inference with Qwen3-VL-8B.

For each datazone, feeds satellite + nightlight + up to N street-view images + POI summary
to the model and collects descriptive evidence phrases per modality.

Output JSONL row:
  {"patch_id": "...", "datazone": "...", "streetview_indices": [0,1,...],
   "reasoning_json": {"evidence": {"satellite": [...], "nightlight": [...],
                                    "streetview_00": [...], ..., "POI": [...], "general": [...]}}}

Usage:
    python perception/infer/perceive_local.py \
        --output-jsonl outputs/perception/qwen3vl_8b_perception.jsonl \
        --max-streetviews 20 --max-samples 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.prompts import perception as prompt_module


def parse_args():
    p = argparse.ArgumentParser(description="Perception-layer inference with Qwen3-VL-8B.")
    p.add_argument("--satellite-meta", type=Path,
                   default=ROOT / "dataset/satellite_dataset/satellite_metadata.csv")
    p.add_argument("--streetview-dir", type=Path,
                   default=ROOT / "dataset/streetview_dataset")
    p.add_argument("--poi-csv", type=Path,
                   default=ROOT / "dataset/poi_dataset/patch_poi.csv")
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--base-model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--adapter-path", type=Path, default=None,
                   help="Optional LoRA adapter directory.")
    p.add_argument("--max-streetviews", type=int, default=20,
                   help="Cap streetviews per datazone (ordered by pano_index).")
    p.add_argument("--max-new-tokens", type=int, default=6144)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all datazones.")
    p.add_argument("--only-datazone", type=str, default=None,
                   help="Run a single datazone id (for debugging).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-quantization", action="store_true",
                   help="Load BF16 without 4-bit quantization.")
    p.add_argument("--merge-adapter", action="store_true")
    p.add_argument("--skip-indicators", action="store_true",
                   help="Skip the text-only domain_indicators second-pass (debug).")
    p.add_argument("--require-indicators", action="store_true",
                   help="Resume: only count a row as completed if it has both "
                        "non-empty evidence and non-empty domain_indicators.")
    p.add_argument("--patch-indicators-only", action="store_true",
                   help="Read existing JSONL, only fill missing domain_indicators "
                        "for rows that already have evidence; rewrite file in place. "
                        "Mutually exclusive with the main inference loop.")
    p.add_argument("--indicator-max-new-tokens", type=int, default=2048,
                   help="max_new_tokens for the text-only indicator call.")
    return p.parse_args()


def _resolve_path(p: str) -> str:
    """Handle absolute paths from other machines by re-anchoring to ROOT."""
    if Path(p).exists():
        return p
    for marker in ("dataset/", "outputs/"):
        idx = p.find(marker)
        if idx >= 0:
            candidate = ROOT / p[idx:]
            if candidate.exists():
                return str(candidate)
    return p


def load_satellite_meta(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dz = r["datazone"]
            pid = r.get("patch_id") or dz
            rows[pid] = {
                "patch_id": pid,
                "datazone": dz,
                "satellite_path": _resolve_path(r["satellite_patch"]),
                "ntl_path": _resolve_path(r["ntl_patch"]),
                "has_streetview": (r.get("has_streetview", "").lower() == "true"),
            }
    return rows


def load_streetviews_for_patch(patch_id: str, streetview_dir: Path) -> list[dict]:
    patch_dir = streetview_dir / patch_id
    if not patch_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in patch_dir.iterdir() if p.suffix.lower() in exts)
    result = []
    for i, p in enumerate(files):
        try:
            pano_index = int(p.stem)
        except ValueError:
            pano_index = i
        result.append({"pano_index": pano_index, "image_path": str(p)})
    return result


def load_poi(path: Path) -> dict[str, dict[str, int]]:
    """Load patch_poi.csv (wide format). Returns {patch_id: {poi_type: count}}."""
    if not path.exists():
        return {}
    result: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        non_type_cols = {"patch_id", "total"}
        for r in reader:
            pid = r["patch_id"]
            result[pid] = {k: int(v) for k, v in r.items()
                           if k not in non_type_cols and int(v) > 0}
    return result


def summarize_poi(counts: dict[str, int]) -> str:
    """Format pre-aggregated POI type counts as natural language for VLM."""
    if not counts:
        return "(no POI recorded)"
    total = sum(counts.values())
    type_lines = ", ".join(
        f"{t} ×{n}"
        for t, n in sorted(counts.items(), key=lambda x: -x[1])
    )
    return f"{total} POIs total: {type_lines}"


def build_record(sat: dict, svs: list[dict], poi_counts: dict[str, int],
                 max_streetviews: int) -> dict:
    svs = svs[:max_streetviews]
    return {
        "patch_id": sat["patch_id"],
        "datazone": sat["datazone"],
        "satellite_path": sat["satellite_path"],
        "ntl_path": sat["ntl_path"],
        "streetview_paths": [s["image_path"] for s in svs],
        "streetview_indices": [s["pano_index"] for s in svs],
        "poi_summary": summarize_poi(poi_counts),
    }


def build_messages(record: dict):
    prompt_text = prompt_module.build_prompt(record)
    images: list[Image.Image] = []
    content: list[dict] = [{"type": "text", "text": prompt_text}]

    images.append(Image.open(record["satellite_path"]).convert("RGB"))
    content.append({"type": "image"})
    images.append(Image.open(record["ntl_path"]).convert("RGB"))
    content.append({"type": "image"})
    for p in record["streetview_paths"]:
        images.append(Image.open(p).convert("RGB"))
        content.append({"type": "image"})

    messages = [
        {"role": "system", "content": prompt_module.SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages, images


def build_single_image_messages(image: Image.Image, modality: str, fallback: bool = False):
    prompt_text = prompt_module.build_single_image_prompt(modality, fallback=fallback)
    messages = [
        {"role": "system", "content": prompt_module.SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image"},
        ]},
    ]
    return messages, [image]


def _validate_indicators(parsed: dict) -> dict:
    """Ensure every key in INDICATOR_KEYS is present with score in [0,4] and a string cue."""
    di = parsed.get("domain_indicators") if isinstance(parsed, dict) else None
    if not isinstance(di, dict):
        return {}
    out: dict[str, dict] = {}
    for k in prompt_module.INDICATOR_KEYS:
        v = di.get(k)
        if not isinstance(v, dict):
            return {}
        score = v.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return {}
        score_int = int(score)
        if score_int < 0 or score_int > 4:
            return {}
        cue = v.get("cue", "")
        if cue is None:
            cue = ""
        out[k] = {"score": score_int, "cue": str(cue)[:120]}
    return out


def perceive_indicators(
    model, processor, evidence: dict, poi_summary: str,
    max_new_tokens: int, temperature: float,
) -> dict:
    """Text-only second pass: score 17 domain indicators from existing evidence."""
    for attempt in range(2):
        prompt_text = prompt_module.build_indicator_prompt(
            evidence, poi_summary or None, minimal=(attempt == 1),
        )
        messages = [
            {"role": "system", "content": prompt_module.SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]
        text = run_inference(model, processor, messages, None,
                             max_new_tokens, temperature)
        parsed = extract_json(text)
        out = _validate_indicators(parsed)
        if out:
            return out
        if attempt == 0:
            print("    [indicators] empty/invalid, retrying with minimal prompt...")
    print("    [indicators] WARNING: returned empty after retry")
    return {}


def perceive_image(model, processor, image: Image.Image, modality: str,
                   max_new_tokens: int, temperature: float) -> list[str]:
    for attempt in range(2):
        fallback = (attempt == 1 and modality == "streetview")
        messages, images = build_single_image_messages(image, modality, fallback=fallback)
        text = run_inference(model, processor, messages, images, max_new_tokens, temperature)
        parsed = extract_json(text)
        phrases = parsed.get("phrases", [])
        if isinstance(phrases, list):
            result = [str(p) for p in phrases if p]
            if result:
                return result
        if attempt == 0:
            print(f"    [{modality}] empty result, retrying with fallback prompt...")
    print(f"    [{modality}] WARNING: returned empty after retry")
    return []


def load_model(args):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as QwenVLModel

    if args.no_quantization:
        bnb_config = None
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading base model: {args.base_model_id}")
    model = QwenVLModel.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    processor_src = args.base_model_id
    if args.adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
        if args.merge_adapter:
            print("Merging adapter...")
            model = model.merge_and_unload()
        processor_src = str(args.adapter_path)

    model.eval()
    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    return model, processor


def extract_json(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
    candidate = text[start:end] if end is not None else text[start:]
    try:
        return json.loads(candidate)
    except Exception:
        pass
    open_b = candidate.count('{') - candidate.count('}')
    open_sq = candidate.count('[') - candidate.count(']')
    repaired = candidate + (']' * max(0, open_sq)) + ('}' * max(0, open_b))
    try:
        return json.loads(repaired)
    except Exception:
        return {}


def normalise_reasoning(parsed: dict) -> dict:
    if not isinstance(parsed, dict):
        return {"evidence": {}}
    if "reasoning_json" in parsed and isinstance(parsed["reasoning_json"], dict):
        inner = parsed["reasoning_json"]
    else:
        inner = parsed
    if "evidence" not in inner or not isinstance(inner["evidence"], dict):
        return {"evidence": {}}
    return {"evidence": inner["evidence"]}


def load_processed_ids(path: Path, require_indicators: bool = False) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = row.get("patch_id")
            reasoning = row.get("reasoning_json") or {}
            evidence = reasoning.get("evidence")
            ok = bool(pid) and isinstance(evidence, dict) and bool(evidence)
            if ok and require_indicators:
                indicators = reasoning.get("domain_indicators")
                ok = isinstance(indicators, dict) and bool(indicators)
            if ok:
                processed.add(str(pid))
    return processed


def run_inference(model, processor, messages, images, max_new_tokens, temperature) -> str:
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text_input, images=images or None, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0.0}
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0]


def patch_indicators_only(args, model, processor) -> None:
    """Read existing JSONL and fill missing domain_indicators in place.

    Only rows with non-empty evidence and missing/empty domain_indicators are
    updated. The file is rewritten atomically (write to .tmp then rename).
    """
    src = args.output_jsonl
    if not src.exists():
        print(f"[patch] file not found: {src}")
        return
    tmp = src.with_suffix(src.suffix + ".tmp")
    rows: list[dict] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    n_updated = 0
    n_skipped = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            reasoning = row.get("reasoning_json") or {}
            evidence = reasoning.get("evidence")
            existing = reasoning.get("domain_indicators")
            need = (
                isinstance(evidence, dict) and bool(evidence)
                and not (isinstance(existing, dict) and existing)
            )
            if not need:
                n_skipped += 1
            else:
                pid = row.get("patch_id", "?")
                print(f"[patch {idx}/{len(rows)}] {pid} — scoring indicators")
                poi_list = evidence.get("POI", [])
                poi_summary = poi_list[0] if poi_list else ""
                indicators = perceive_indicators(
                    model, processor, evidence, poi_summary,
                    args.indicator_max_new_tokens, args.temperature,
                )
                reasoning["domain_indicators"] = indicators
                row["reasoning_json"] = reasoning
                if indicators:
                    n_updated += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(src)
    print(f"\n[patch] done. updated={n_updated}, skipped={n_skipped}, total={len(rows)}")


def main():
    args = parse_args()

    if args.patch_indicators_only:
        model, processor = load_model(args)
        patch_indicators_only(args, model, processor)
        return

    sat_meta = load_satellite_meta(args.satellite_meta)
    poi_meta = load_poi(args.poi_csv)

    all_patch_ids = sorted(sat_meta.keys())
    if args.only_datazone:
        all_patch_ids = [p for p in all_patch_ids
                         if sat_meta[p]["datazone"] == args.only_datazone]

    records = []
    for pid in all_patch_ids:
        svs = load_streetviews_for_patch(pid, args.streetview_dir)
        if not svs:
            continue
        records.append(build_record(sat_meta[pid], svs, poi_meta.get(pid, {}),
                                    args.max_streetviews))

    if args.max_samples > 0:
        records = records[:args.max_samples]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    resume = args.output_jsonl.exists() and not args.overwrite
    processed = (
        load_processed_ids(args.output_jsonl, require_indicators=args.require_indicators)
        if resume else set()
    )
    mode = "a" if resume else "w"

    model, processor = load_model(args)
    print(f"Total records: {len(records)}; already processed: {len(processed)}")

    with open(args.output_jsonl, mode, encoding="utf-8") as f:
        for idx, rec in enumerate(records, start=1):
            pid = rec["patch_id"]
            dz = rec["datazone"]
            if pid in processed:
                print(f"[{idx}/{len(records)}] skip {pid}")
                continue
            n_sv = len(rec["streetview_paths"])
            n_calls = 2 + n_sv  # satellite + ntl + streetviews
            print(f"[{idx}/{len(records)}] {pid} ({dz}) — {n_sv} streetview(s), {n_calls} calls")

            evidence: dict = {}
            try:
                evidence["satellite"] = perceive_image(
                    model, processor,
                    Image.open(rec["satellite_path"]).convert("RGB"),
                    "satellite", args.max_new_tokens, args.temperature)
                print(f"  satellite: {evidence['satellite']}")

                evidence["nightlight"] = perceive_image(
                    model, processor,
                    Image.open(rec["ntl_path"]).convert("RGB"),
                    "nightlight", args.max_new_tokens, args.temperature)
                print(f"  nightlight: {evidence['nightlight']}")

                for i, sv_path in enumerate(rec["streetview_paths"]):
                    key = f"streetview_{i:02d}"
                    evidence[key] = perceive_image(
                        model, processor,
                        Image.open(sv_path).convert("RGB"),
                        "streetview", args.max_new_tokens, args.temperature)

            except Exception as e:
                print(f"  error: {e}")

            poi_summary = rec.get("poi_summary", "")
            evidence["POI"] = [poi_summary] if poi_summary else ["no POI recorded"]

            indicators: dict = {}
            if not args.skip_indicators and evidence:
                indicators = perceive_indicators(
                    model, processor, evidence, poi_summary,
                    args.indicator_max_new_tokens, args.temperature,
                )
                if indicators:
                    print(f"  indicators: {len(indicators)} scored")

            reasoning = {"evidence": evidence, "domain_indicators": indicators}
            row = {
                "patch_id": rec["patch_id"],
                "datazone": dz,
                "streetview_indices": rec["streetview_indices"],
                "reasoning_json": reasoning,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nDone. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
