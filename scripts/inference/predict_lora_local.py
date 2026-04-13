# -*- coding: utf-8 -*-
"""Local inference with LoRA-adapted Qwen3-VL-8B-Instruct.

Loads the base model, merges the LoRA adapter, then runs predictions
on a JSONL file in the same format as predict_qwen3_vl_plus_api.py.

Usage:
    # Preview (5 samples)
    python scripts/inference/predict_lora_local.py \
        --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
        --output-jsonl outputs/predictions/lora_poi_triple_preview.jsonl \
        --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
        --input-mode triple --task explain --max-samples 5

    # Full run
    python scripts/inference/predict_lora_local.py \
        --input-jsonl dataset/sat_ntl_svi_aligned/vlm_data/triple_explain_test.jsonl \
        --output-jsonl outputs/predictions/lora_poi_triple.jsonl \
        --adapter-path outputs/lora_adapters_poi_cuda/final_adapter \
        --input-mode triple --task explain
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glasgow_vlm.data import GlasgowVLMJsonlDataset

PROMPT_MODULES = {
    "default":               "glasgow_vlm.prompts.default",
    "structured":            "glasgow_vlm.prompts.structured",
    "structured_reasoning":  "glasgow_vlm.prompts.structured_reasoning",
    "structured_plus":       "glasgow_vlm.prompts.structured_plus",
    "simple":                "glasgow_vlm.prompts.simple",
}


def load_prompt_module(name: str):
    module_path = PROMPT_MODULES.get(name)
    if module_path is None:
        raise ValueError(f"Unknown prompt module '{name}'. Available: {list(PROMPT_MODULES)}")
    return importlib.import_module(module_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Local inference with LoRA-adapted Qwen3-VL-8B.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True,
                        help="Path to the saved LoRA adapter directory (final_adapter/).")
    parser.add_argument("--base-model-id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model ID or local path for the base model.")
    parser.add_argument("--input-mode",
                        choices=("streetview", "satellite", "dual", "satellite_ntl", "triple"),
                        default="triple")
    parser.add_argument("--task", choices=("ordinal", "explain"), default="explain")
    parser.add_argument("--prompt", choices=list(PROMPT_MODULES), default="structured")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit to first N samples (0 = all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output JSONL.")
    parser.add_argument("--no-quantization", action="store_true",
                        help="Load in BF16 without 4-bit quantization (more VRAM).")
    parser.add_argument("--merge-adapter", action="store_true",
                        help="Merge LoRA weights into base model before inference (faster, more VRAM).")
    return parser.parse_args()


def load_model_and_processor(args):
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as QwenVLModel
    from peft import PeftModel

    print(f"Loading base model: {args.base_model_id}")

    if args.no_quantization:
        bnb_config = None
        dtype = torch.bfloat16
    else:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16

    model = QwenVLModel.from_pretrained(
        args.base_model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)

    if args.merge_adapter:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()

    processor = AutoProcessor.from_pretrained(
        str(args.adapter_path), trust_remote_code=True
    )

    return model, processor


def _derive_modalities(record: dict, input_mode: str) -> tuple[tuple[str, ...], str]:
    secondary_modality = record.get("secondary_modality", "satellite")
    if input_mode == "triple":
        modalities: tuple[str, ...] = ("satellite", "ntl")
    elif input_mode == "satellite_ntl":
        modalities = ("ntl",)
    elif input_mode == "dual" and secondary_modality == "ntl":
        modalities = ("ntl",)
    elif input_mode in ("satellite", "streetview"):
        modalities = ()
    else:
        modalities = ("satellite", secondary_modality)
    primary_modality = "satellite" if input_mode == "satellite_ntl" else "streetview"
    return tuple(dict.fromkeys(modalities)), primary_modality


def build_messages_local(record: dict, input_mode: str, task: str, prompt_module) -> tuple[list[dict], list[Image.Image]]:
    modalities, primary_modality = _derive_modalities(record, input_mode)
    prompt_text = prompt_module.build_prompt(record, task, modalities=modalities, primary_modality=primary_modality)

    images: list[Image.Image] = []
    content: list[dict] = [{"type": "text", "text": prompt_text}]

    if input_mode in ("streetview", "dual", "triple"):
        img = Image.open(record["streetview_path"]).convert("RGB")
        images.append(img)
        content.append({"type": "image"})

    if input_mode in ("satellite", "dual", "triple", "satellite_ntl"):
        img = Image.open(record["satellite_path"]).convert("RGB")
        images.append(img)
        content.append({"type": "image"})

    if input_mode in ("satellite_ntl", "triple"):
        ntl_path = record.get("ntl_path")
        if not ntl_path:
            raise ValueError(f"Record is missing ntl_path for {input_mode} input mode")
        img = Image.open(ntl_path).convert("RGB")
        images.append(img)
        content.append({"type": "image"})

    messages = [
        {"role": "system", "content": prompt_module.SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages, images


def extract_json(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            pass
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

    def _try_load(fragment: str) -> dict:
        fragment = fragment.strip()
        if not fragment:
            return {}
        open_braces = fragment.count('{')
        close_braces = fragment.count('}')
        open_brackets = fragment.count('[')
        close_brackets = fragment.count(']')
        repaired = fragment + ('}' * max(0, open_braces - close_braces)) + (']' * max(0, open_brackets - close_brackets))
        try:
            return json.loads(repaired)
        except Exception:
            return {}

    parsed = _try_load(candidate)
    if parsed:
        return parsed
    for cut in range(len(candidate) - 1, 1, -1):
        parsed = _try_load(candidate[:cut])
        if parsed:
            return parsed
    return {}


def load_processed_ids(path: Path) -> set[str]:
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
            row_id = row.get("id")
            prediction = row.get("prediction_json")
            if row_id and isinstance(prediction, dict) and prediction:
                processed.add(str(row_id))
    return processed


def run_inference(model, processor, messages: list[dict], images: list[Image.Image],
                  max_new_tokens: int, temperature: float) -> str:
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=text_input,
        images=images if images else None,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def main():
    args = parse_args()
    prompt_module = load_prompt_module(args.prompt)
    print(f"Prompt module: {args.prompt}")

    model, processor = load_model_and_processor(args)

    dataset = GlasgowVLMJsonlDataset(args.input_jsonl, input_mode=args.input_mode, task=args.task)
    samples = dataset.records[: args.max_samples] if args.max_samples > 0 else dataset.records

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    resume_enabled = args.output_jsonl.exists() and not args.overwrite
    processed_ids = load_processed_ids(args.output_jsonl) if resume_enabled else set()
    mode = "a" if resume_enabled and args.output_jsonl.exists() else "w"

    skipped = 0
    written = 0

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

            print(f"[{idx}/{len(samples)}] Running inference for {sample_id}")
            messages, images = build_messages_local(record, args.input_mode, args.task, prompt_module)
            text = run_inference(model, processor, messages, images, args.max_new_tokens, args.temperature)
            prediction = extract_json(text)

            row = {
                "id": sample_id,
                "datazone": record["datazone"],
                "prediction_json": prediction,
                "model": str(args.adapter_path),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
            print(f"  -> {json.dumps(prediction, ensure_ascii=False)[:120]}")

    print(f"\nDone. Written {written} new sample(s); skipped {skipped}.")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
