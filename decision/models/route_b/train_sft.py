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


def run_sft(cfg: dict, sft_data_path: str | Path, dry_run: int = 0) -> None:
    # Lazy imports — remote-host-only deps.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
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

    max_len = tr["max_seq_len"]

    def render(example: dict) -> dict:
        messages = example["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("expected last message role=assistant for SFT completion-only masking")
        prompt_ids = tok.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
        full_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        full_ids = full_ids[:max_len]
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + list(full_ids[prompt_len:])
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages", "datazone"])

    if dry_run > 0:
        n = min(dry_run, len(ds))
        print(f"[sft][dry-run] inspecting first {n} rendered rows")
        for i in range(n):
            ex = ds[i]
            ids = ex["input_ids"]
            labels = ex["labels"]
            masked = sum(1 for t in labels if t == -100)
            kept = len(labels) - masked
            try:
                first_kept_idx = next(j for j, t in enumerate(labels) if t != -100)
            except StopIteration:
                first_kept_idx = -1
            completion_ids = [t for t in labels if t != -100]
            completion_text = tok.decode(completion_ids, skip_special_tokens=False)
            gold_assistant = rows[i]["messages"][-1]["content"]
            print(f"\n[row {i}] total={len(ids)}  masked={masked}  kept={kept}  prompt_len={first_kept_idx}")
            print(f"  decoded kept segment (first 300 chars):\n    {completion_text[:300]!r}")
            print(f"  gold assistant content (first 300 chars):\n    {gold_assistant[:300]!r}")
            norm_kept = completion_text.replace(tok.eos_token or "", "").strip()
            match = norm_kept.startswith(gold_assistant.strip()[:200])
            print(f"  ✓ prefix match (first 200 chars of gold in decoded kept): {match}")
        print("[sft][dry-run] done — skipping model load / training")
        return

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": 0},
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
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation"],
        learning_rate=tr["lr"],
        warmup_ratio=tr["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
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
        processing_class=tok,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True),
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[sft] saved adapter → {out_dir}")


def _main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sft-data", required=True)
    p.add_argument("--dry-run", type=int, default=0,
                   help="If >0, render this many rows, print label-mask diagnostics, and exit without training.")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    run_sft(cfg, args.sft_data, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
