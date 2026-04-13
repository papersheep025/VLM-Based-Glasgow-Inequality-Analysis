# -*- coding: utf-8 -*-
"""QLoRA fine-tuning for Qwen3-VL-8B on Glasgow deprivation data (no POI).
CUDA/Linux version — PyTorch + HuggingFace PEFT.

Identical to train_qwen3_vl_lora_poi_cuda.py except it uses the dataset
without POI context (lora_training_data instead of lora_training_data_poi).

Usage:
    python scripts/training/train_qwen3_vl_lora_cuda.py \
        --max-samples 20 --epochs 1 --print-every 5

    accelerate launch --num_processes 2 \
        scripts/training/train_qwen3_vl_lora_cuda.py

    python scripts/training/train_qwen3_vl_lora_cuda.py \
        --path-prefix-remap /Users/papersheep/projects/VLM-Based-Glasgow-Inequality-Analysis:/home/user/glasgow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_DATASET_DIR = str(ROOT / "dataset" / "lora_training_data")
DEFAULT_OUTPUT_DIR = str(ROOT / "outputs" / "lora_adapters_cuda")


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL-8B (CUDA, no POI)")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-quantization", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--path-prefix-remap", type=str, default=None,
                   help="Remap image path prefix: OLD_PREFIX:NEW_PREFIX")
    return p.parse_args()


class GlasgowVisionDataset(Dataset):
    def __init__(self, hf_dataset, old_prefix=None, new_prefix=None):
        self.dataset = hf_dataset
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix

    def __len__(self):
        return len(self.dataset)

    def _remap(self, path):
        if self.old_prefix and path.startswith(self.old_prefix):
            return self.new_prefix + path[len(self.old_prefix):]
        return path

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = json.loads(item["messages"]) if isinstance(item["messages"], str) else item["messages"]
        image_paths = [self._remap(p) for p in item["images"]]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return {"messages": messages, "images": images, "id": item["id"]}


def build_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    labels = input_ids.clone()
    ids = input_ids.tolist()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    last_im_start = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            last_im_start = i
            break

    if last_im_start < 0:
        labels[:] = -100
        return labels

    header_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    response_start = last_im_start + 1 + len(header_tokens)
    labels[:response_start] = -100
    return labels


def make_collate_fn(processor, max_length):
    def collate_fn(batch):
        texts, all_images = [], []
        for item in batch:
            text = processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.extend(item["images"])

        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        labels = torch.stack([
            build_labels(inputs["input_ids"][i], processor.tokenizer)
            for i in range(len(batch))
        ])
        inputs["labels"] = labels
        return dict(inputs)

    return collate_fn


def _save(model, processor, path, accelerator=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model) if accelerator else model
    unwrapped.save_pretrained(str(path))
    processor.save_pretrained(str(path))


def main():
    args = parse_args()

    try:
        from accelerate import Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        is_main = True

    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as QwenVLModel
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    bnb_config = None if args.no_quantization else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if is_main:
        logger.info(f"Loading model: {args.model_id}")
    model = QwenVLModel.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if accelerator is None else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    if not args.no_quantization:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_checkpoint
        )
    elif args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ))

    if is_main:
        model.print_trainable_parameters()

    old_prefix, new_prefix = None, None
    if args.path_prefix_remap:
        parts = args.path_prefix_remap.split(":", 1)
        if len(parts) == 2:
            old_prefix, new_prefix = parts

    ds = load_from_disk(args.dataset_dir)
    hf_dataset = ds[args.split]
    if args.max_samples > 0:
        hf_dataset = hf_dataset.select(range(min(args.max_samples, len(hf_dataset))))
    if is_main:
        logger.info(f"Training samples: {len(hf_dataset)}")

    dataset = GlasgowVisionDataset(hf_dataset, old_prefix, new_prefix)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(processor, args.max_length),
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    if accelerator is not None:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        accum_count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)

        for batch in progress:
            if accelerator is None:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            loss = model(**batch).loss / args.grad_accum_steps
            (accelerator.backward(loss) if accelerator else loss.backward())
            running_loss += loss.item() * args.grad_accum_steps
            accum_count += 1

            if accum_count % args.grad_accum_steps == 0:
                (accelerator.clip_grad_norm_ if accelerator else torch.nn.utils.clip_grad_norm_)(
                    model.parameters(), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss / args.grad_accum_steps
                running_loss = 0.0
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

                if is_main and global_step % args.print_every == 0:
                    tqdm.write(f"  Epoch {epoch+1} | Step {global_step} | Loss: {avg_loss:.4f}")

                if is_main and global_step % args.save_every == 0:
                    ckpt = output_dir / f"checkpoint_ep{epoch+1}_step{global_step}"
                    _save(model, processor, ckpt, accelerator)
                    tqdm.write(f"  Checkpoint saved: {ckpt}")

    if is_main:
        final_path = output_dir / "final_adapter"
        _save(model, processor, final_path, accelerator)
        logger.info(f"Final adapter saved to {final_path}")


if __name__ == "__main__":
    main()
