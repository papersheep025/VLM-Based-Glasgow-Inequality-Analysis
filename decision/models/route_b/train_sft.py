"""
PEFT LoRA SFT for Route B.  Runs on the remote GPU host.

Dependencies (installed on the remote host): transformers, peft, datasets,
accelerate, torch.  All imports are lazy so this module can still be imported
on the author's laptop without those packages available.

CLI:
    python -m decision.models.route_b.train_sft \\
        --config decision/configs/route_b_llm_sft.yaml \\
        --sft-data outputs/decision/route_b/sft_v0.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_sft(cfg: dict, sft_data_path: str | Path) -> None:
    # Lazy imports — remote-host-only deps.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    sft_cfg = cfg["sft"]
    base = sft_cfg["base_model"]
    lora = sft_cfg["lora"]
    tr = sft_cfg["train"]
    out_dir = Path(sft_cfg["output_adapter_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(sft_data_path)
    print(f"[sft] loaded {len(rows)} rows from {sft_data_path}")

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def render(example: dict) -> dict:
        text = tok.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        enc = tok(text, truncation=True, max_length=tr["max_seq_len"], padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages", "datazone"])

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lcfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation"],
        learning_rate=tr["lr"],
        warmup_ratio=tr["warmup_ratio"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        seed=tr["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[sft] saved adapter → {out_dir}")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sft-data", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    run_sft(cfg, args.sft_data)


if __name__ == "__main__":
    _main()
