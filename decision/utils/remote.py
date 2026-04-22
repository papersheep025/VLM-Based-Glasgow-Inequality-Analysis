"""
Unified LLM client for Route B (teacher rationalisation + inference).

Three backends, selected via YAML ``remote.backend``:

* ``openai_compat``  — HTTPS call to any OpenAI-compatible endpoint
                       (vLLM, DashScope, OpenAI). Default.
* ``vllm_http``      — alias for openai_compat routed at a vLLM server.
* ``local_hf``       — in-process HuggingFace ``transformers`` generation
                       (imported lazily; use only on the remote GPU host).

All backends expose the same ``chat(messages, **gen_kwargs) -> str`` API.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RemoteConfig:
    backend: str = "openai_compat"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    api_key_env: str = "ROUTE_B_API_KEY"
    timeout_s: int = 120
    max_retries: int = 3
    # For local_hf backend only: path to a PEFT LoRA adapter directory.
    # Leave empty to use the base model (e.g., for rationalisation).
    adapter_path: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RemoteConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class RemoteLLM:
    def __init__(self, cfg: RemoteConfig) -> None:
        self.cfg = cfg
        self._hf_pipe = None

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self.cfg.backend in ("openai_compat", "vllm_http"):
            return self._chat_openai(messages, temperature, max_tokens)
        if self.cfg.backend == "local_hf":
            return self._chat_local_hf(messages, temperature, max_tokens)
        raise ValueError(f"unknown backend: {self.cfg.backend}")

    # ------------------------------------------------------------------
    def _chat_openai(self, messages, temperature, max_tokens) -> str:
        import requests  # lazy import

        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = os.getenv(self.cfg.api_key_env, "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"remote LLM call failed after {self.cfg.max_retries} retries: {last_err}")

    # ------------------------------------------------------------------
    def _chat_local_hf(self, messages, temperature, max_tokens) -> str:
        if self._hf_pipe is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.cfg.model)
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model, torch_dtype="auto", device_map="auto")
            if self.cfg.adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.cfg.adapter_path)
                model = model.merge_and_unload()
            self._hf_pipe = pipeline("text-generation", model=model, tokenizer=tok)

        tok = self._hf_pipe.tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = self._hf_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            return_full_text=False,
        )
        return out[0]["generated_text"]


def load_remote(yaml_cfg: dict) -> RemoteLLM:
    """Build RemoteLLM from a parsed YAML ``remote`` section."""
    return RemoteLLM(RemoteConfig.from_dict(yaml_cfg.get("remote", {})))


def extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction from an LLM completion."""
    # Strip <think>...</think> block if present (Qwen3 thinking mode)
    think_end = text.find("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None
