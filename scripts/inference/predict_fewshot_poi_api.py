# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from predict_qwen3_vl_plus_api import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DASHSCOPE_API_KEY,
    call_model,
    encode_image_data_url,
    extract_json,
    get_api_key,
    load_processed_ids,
    normalize_prediction_json,
    verify_api_connection,
    _derive_modalities,
)
from glasgow_vlm.data import GlasgowVLMJsonlDataset
from glasgow_vlm.prompts import structured_fewshot_poi as poi_mod


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-Plus with few-shot + POI context via DashScope API.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("streetview", "satellite", "dual", "satellite_ntl", "triple"), default="dual")
    parser.add_argument("--task", choices=("ordinal", "explain"), default="ordinal")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0, help="Only run the first N samples. 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output JSONL.")
    parser.add_argument("--few-shot-json", type=Path, default=poi_mod.FEW_SHOT_JSON_PATH, help="Path to few-shot examples JSON.")
    parser.add_argument("--few-shot-count", type=int, default=None, help="Number of few-shot examples to use (default: all).")
    parser.add_argument("--few-shot-quintiles", type=int, nargs="+", default=None, help="Specific quintiles to include (e.g. 1 3 5).")
    parser.add_argument("--poi-csv", type=Path, default=poi_mod.DEFAULT_POI_CSV, help="Path to datazone POI CSV.")
    return parser.parse_args()


def build_fewshot_poi_messages(
    sample: dict,
    input_mode: str,
    task: str,
    few_shot_examples: list[dict],
    poi_lookup: dict,
    prompt_module,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": prompt_module.SYSTEM_PROMPT}]

    for ex in few_shot_examples:
        modalities, primary_modality = _derive_modalities(
            {"secondary_modality": "satellite"}, input_mode,
        )
        ex_dz = ex.get("datazone", "")
        ex_poi_ctx = prompt_module.format_poi_context(poi_lookup.get(ex_dz))
        prompt_text = prompt_module.build_prompt(
            {}, task, modalities=modalities, primary_modality=primary_modality,
            poi_context=ex_poi_ctx,
        )
        user_content = prompt_module.build_fewshot_user_content(
            ex, input_mode, prompt_text, encode_image_data_url,
        )
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": prompt_module.build_fewshot_response(ex)})

    record = sample["record"]
    modalities, primary_modality = _derive_modalities(record, input_mode)
    dz = record.get("datazone", "")
    poi_ctx = prompt_module.format_poi_context(poi_lookup.get(dz))
    prompt_text = prompt_module.build_prompt(
        record, task, modalities=modalities, primary_modality=primary_modality,
        poi_context=poi_ctx,
    )
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    if input_mode in ("streetview", "dual", "triple"):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(record["streetview_path"])},
        })
    if input_mode in ("satellite", "dual", "triple", "satellite_ntl"):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(record["satellite_path"])},
        })
    if input_mode in ("satellite_ntl", "triple"):
        ntl_path = record.get("ntl_path")
        if not ntl_path:
            raise ValueError(f"Record {sample['id']} is missing ntl_path for {input_mode} input mode")
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(ntl_path)},
        })
    messages.append({"role": "user", "content": content})
    return messages


def main():
    args = parse_args()

    poi_lookup = poi_mod.load_poi_lookup(args.poi_csv)
    print(f"POI lookup loaded: {len(poi_lookup)} datazones from {args.poi_csv}")

    few_shot_examples = poi_mod.load_few_shot_examples(
        json_path=args.few_shot_json,
        quintiles=args.few_shot_quintiles,
        count=args.few_shot_count,
    )
    quintile_list = [e["quintile"] for e in few_shot_examples]
    images_per_example = {"streetview": 1, "satellite": 1, "dual": 2, "satellite_ntl": 2, "triple": 3}
    img_count = images_per_example.get(args.input_mode, 1)
    total_images = img_count * (len(few_shot_examples) + 1)
    print(f"Few-shot config: {len(few_shot_examples)} example(s), quintiles={quintile_list}")
    print(f"Images per API call: {total_images} ({img_count} x {len(few_shot_examples)} examples + {img_count} query)")
    print(f"Prompt module: structured_fewshot_poi")

    api_key = get_api_key(args)
    verify_api_connection(args.base_url, api_key, args.model)
    dataset = GlasgowVLMJsonlDataset(args.input_jsonl, input_mode=args.input_mode, task=args.task)
    samples = dataset.records[:args.max_samples] if args.max_samples and args.max_samples > 0 else dataset.records

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    resume_enabled = (args.resume or args.output_jsonl.exists()) and not args.overwrite
    processed_ids = load_processed_ids(args.output_jsonl) if resume_enabled else set()
    mode = "a" if resume_enabled and args.output_jsonl.exists() else "w"
    skipped = 0
    written = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(args.output_jsonl, mode, encoding="utf-8") as f:
        print(f"Loaded {len(samples)} sample(s) from {args.input_jsonl}")
        if processed_ids:
            print(f"Resuming: already processed {len(processed_ids)} id(s).")
        for idx, record in enumerate(samples, start=1):
            sample_id = str(record["id"])
            if sample_id in processed_ids:
                skipped += 1
                print(f"[{idx}/{len(samples)}] Skipping {sample_id}")
                continue
            sample = {"id": sample_id, "record": record}
            print(f"[{idx}/{len(samples)}] Calling model for {sample['id']}")
            messages = build_fewshot_poi_messages(
                sample, args.input_mode, args.task,
                few_shot_examples, poi_lookup, poi_mod,
            )
            text, usage = call_model(args.base_url, api_key, args.model, messages, args.max_new_tokens, args.temperature)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            print(f"  tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
            print(f"  cumulative: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_prompt_tokens + total_completion_tokens}")
            row = {
                "id": sample["id"],
                "datazone": sample["record"]["datazone"],
                "prediction_json": normalize_prediction_json(extract_json(text), task=args.task),
                "model": args.model,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1

    print(f"Saved predictions to {args.output_jsonl}")
    print(f"Written {written} new sample(s); skipped {skipped} existing sample(s).")
    print(f"Total tokens: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_prompt_tokens + total_completion_tokens}")


if __name__ == "__main__":
    main()
