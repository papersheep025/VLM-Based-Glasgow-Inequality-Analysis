# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glasgow_vlm.data import GlasgowVLMJsonlDataset
from glasgow_vlm.prompts import SYSTEM_PROMPT


DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Paste your DashScope API key here.
DASHSCOPE_API_KEY = "sk-540a1528d0c24d50b7ababd5b3e42871"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3-VL-Plus via DashScope OpenAI-compatible API.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("streetview", "satellite", "dual", "triple"), default="dual")
    parser.add_argument("--task", choices=("ordinal", "rank", "explain"), default="ordinal")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0, help="Only run the first N samples. 0 means all.")
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


def build_messages(sample: dict, input_mode: str, task: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": sample["prompt"]}]
    if input_mode in ("streetview", "dual", "triple"):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(sample["record"]["streetview_path"])},
            }
        )
    if input_mode in ("satellite", "dual", "triple"):
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(sample["record"]["satellite_path"])},
            }
        )
    if input_mode == "triple":
        ntl_path = sample["record"].get("ntl_path")
        if not ntl_path:
            raise ValueError(f"Record {sample['id']} is missing ntl_path for triple input mode")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_url(ntl_path)},
            }
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def call_model(base_url: str, api_key: str, model: str, messages: list[dict], max_new_tokens: int, temperature: float) -> str:
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
        return data["choices"][0]["message"]["content"]
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


def main():
    args = parse_args()
    api_key = get_api_key(args)
    verify_api_connection(args.base_url, api_key, args.model)
    dataset = GlasgowVLMJsonlDataset(args.input_jsonl, input_mode=args.input_mode, task=args.task)
    samples = dataset.records[: args.max_samples] if args.max_samples and args.max_samples > 0 else dataset.records

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        print(f"Loaded {len(samples)} sample(s) from {args.input_jsonl}")
        for idx, record in enumerate(samples, start=1):
            sample = {
                "id": record["id"],
                "record": record,
                "prompt": record.get("prompt", ""),
            }
            print(f"[{idx}/{len(samples)}] Calling model for {sample['id']}")
            messages = build_messages(sample, args.input_mode, args.task)
            text = call_model(args.base_url, api_key, args.model, messages, args.max_new_tokens, args.temperature)
            row = {
                "id": sample["id"],
                "datazone": sample["record"]["datazone"],
                "prediction_text": text,
                "prediction_json": extract_json(text),
                "model": args.model,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Saved predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
