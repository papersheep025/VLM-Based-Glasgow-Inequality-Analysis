# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import importlib

from glasgow_vlm.config import DEFAULT_BASE_URL, DEFAULT_MODEL, DASHSCOPE_API_KEY
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
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-Plus via DashScope OpenAI-compatible API.")
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
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSONL by skipping completed ids.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite any existing output JSONL and start from scratch.")
    parser.add_argument("--prompt", choices=list(PROMPT_MODULES), default="structured", help="Prompt module to use.")
    return parser.parse_args()


def get_api_key(args) -> str:
    key = args.api_key or DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set in code, args, or environment.")
    return key


def encode_image_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        suffix = path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(suffix, "application/octet-stream")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


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


def build_messages(sample: dict, input_mode: str, task: str, prompt_module) -> list[dict]:
    record = sample["record"]
    modalities, primary_modality = _derive_modalities(record, input_mode)
    prompt_text = prompt_module.build_prompt(record, task, modalities=modalities, primary_modality=primary_modality)
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    if input_mode in ("streetview", "dual", "triple"):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(record["streetview_path"])},
            }
        )
    if input_mode in ("satellite", "dual", "triple", "satellite_ntl"):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(record["satellite_path"])},
            }
        )
    if input_mode in ("satellite_ntl", "triple"):
        ntl_path = record.get("ntl_path")
        if not ntl_path:
            raise ValueError(f"Record {sample['id']} is missing ntl_path for {input_mode} input mode")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(ntl_path)},
            }
        )
    return [
        {"role": "system", "content": prompt_module.SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


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

    # If the model output was truncated, walk backward and try to recover the
    # longest valid JSON prefix instead of dropping the whole sample.
    for cut in range(len(candidate) - 1, 1, -1):
        parsed = _try_load(candidate[:cut])
        if parsed:
            return parsed

    return {}


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_prediction_json(payload: dict, task: str = "ordinal") -> dict:
    if not isinstance(payload, dict):
        return {}

    return _sanitize_json_value(payload)


def call_model(base_url: str, api_key: str, model: str, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, dict]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"], data.get("usage", {})
    except Exception as exc:
        raise RuntimeError(f"Unexpected response format: {data}") from exc


def verify_api_connection(base_url: str, api_key: str, model: str) -> None:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Reply with exactly: OK"}]},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        text = json.dumps(data, ensure_ascii=False)
    print("API connected.")
    print(f"Model: {model}")
    print(f"Response: {text}")


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
            if not row_id:
                continue
            if isinstance(prediction, dict) and prediction:
                processed.add(str(row_id))
    return processed


def main():
    args = parse_args()
    prompt_module = load_prompt_module(args.prompt)
    print(f"Prompt module: {args.prompt} ({PROMPT_MODULES[args.prompt]})")
    api_key = get_api_key(args)
    verify_api_connection(args.base_url, api_key, args.model)
    dataset = GlasgowVLMJsonlDataset(args.input_jsonl, input_mode=args.input_mode, task=args.task)
    samples = dataset.records[: args.max_samples] if args.max_samples and args.max_samples > 0 else dataset.records

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
            print(f"Resuming from existing output file; already processed {len(processed_ids)} id(s).")
        for idx, record in enumerate(samples, start=1):
            sample_id = str(record["id"])
            if sample_id in processed_ids:
                skipped += 1
                print(f"[{idx}/{len(samples)}] Skipping already processed {sample_id}")
                continue
            sample = {
                "id": sample_id,
                "record": record,
            }
            print(f"[{idx}/{len(samples)}] Calling model for {sample['id']}")
            messages = build_messages(sample, args.input_mode, args.task, prompt_module)
            text, usage = call_model(args.base_url, api_key, args.model, messages, args.max_new_tokens, args.temperature)
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            print(f"  tokens this call: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
            print(f"  tokens cumulative: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_prompt_tokens + total_completion_tokens}")
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
    print(f"Total tokens consumed: prompt={total_prompt_tokens}, completion={total_completion_tokens}, total={total_prompt_tokens + total_completion_tokens}")


if __name__ == "__main__":
    main()




