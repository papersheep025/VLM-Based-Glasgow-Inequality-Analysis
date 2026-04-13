# -*- coding: utf-8 -*-
"""QLoRA fine-tuning for Qwen3-VL-8B on Glasgow deprivation data WITH POI context.
CUDA/Linux version — PyTorch + HuggingFace PEFT.

Dataset is the same HuggingFace DatasetDict built by build_lora_training_data_poi.py.
No dataset rebuild needed; just transfer the dataset directory and remap image paths.

Usage:
    # Single GPU (smoke test)
    python scripts/training/train_qwen3_vl_lora_poi_cuda.py \
        --max-samples 20 --epochs 1 --print-every 5

    # Single GPU (full run)
    python scripts/training/train_qwen3_vl_lora_poi_cuda.py

    # Multi-GPU via accelerate (recommended, uses GPU 0 & 1)
    accelerate launch --num_processes 2 \
        scripts/training/train_qwen3_vl_lora_poi_cuda.py

    # With image path remapping (Mac -> server)
    python scripts/training/train_qwen3_vl_lora_poi_cuda.py \
        --path-prefix-remap /Users/papersheep/projects/VLM-Based-Glasgow-Inequality-Analysis:/home/user/glasgow

    # Resume from checkpoint
    python scripts/training/train_qwen3_vl_lora_poi_cuda.py \
        --resume-from outputs/lora_adapters_poi_cuda/checkpoint_ep1_step500
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
DEFAULT_DATASET_DIR = str(ROOT / "dataset" / "lora_training_data_poi")
DEFAULT_OUTPUT_DIR = str(ROOT / "outputs" / "lora_adapters_poi_cuda")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL-8B + POI (CUDA)")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                   help="HuggingFace model ID or local path")
    p.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a saved PEFT checkpoint directory to resume from.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps).")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=2048,
                   help="Maximum token sequence length (truncate if longer).")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500,
                   help="Save checkpoint every N optimizer steps.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-quantization", action="store_true",
                   help="Load model in BF16 without 4-bit quantization (requires more VRAM).")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing (saves VRAM, slower).")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Limit training samples for smoke test (0=all).")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker processes.")
    p.add_argument("--path-prefix-remap", type=str, default=None,
                   help="Remap image path prefix: OLD_PREFIX:NEW_PREFIX")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GlasgowVisionDataset(Dataset):
    def __init__(self, hf_dataset, old_prefix: str = None, new_prefix: str = None):
        self.dataset = hf_dataset
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix

    def __len__(self):
        return len(self.dataset)

    def _remap(self, path: str) -> str:
        if self.old_prefix and path.startswith(self.old_prefix):
            return self.new_prefix + path[len(self.old_prefix):]
        return path

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = json.loads(item["messages"]) if isinstance(item["messages"], str) else item["messages"]
        image_paths = [self._remap(p) for p in item["images"]]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return {"messages": messages, "images": images, "id": item["id"]}


# ---------------------------------------------------------------------------
# Label masking
# ---------------------------------------------------------------------------

def build_labels(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Return labels tensor: -100 for prompt tokens, real IDs for response tokens."""
    labels = input_ids.clone()
    ids = input_ids.tolist()

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # Find last <|im_start|> = start of assistant turn
    last_im_start = -1
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            last_im_start = i
            break

    if last_im_start < 0:
        labels[:] = -100
        return labels

    # Skip: <|im_start|> + "assistant" token(s) + "\n" token(s)
    header_tokens = tokenizer.encode("assistant\n", add_special_tokens=False)
    response_start = last_im_start + 1 + len(header_tokens)

    labels[:response_start] = -100
    return labels


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def make_collate_fn(processor, max_length: int):
    def collate_fn(batch):
        texts, all_images = [], []
        for item in batch:
            text = processor.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- accelerate (optional multi-GPU) ----
    try:
        from accelerate import Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
        is_main = accelerator.is_main_process
    except ImportError:
        accelerator = None
        is_main = True

    if is_main:
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Dataset: {args.dataset_dir}")
        logger.info(f"Output: {args.output_dir}")

    # ---- imports ----
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen2_5VLForConditionalGeneration as QwenVLModel
    except ImportError:
        from transformers import AutoModelForImageTextToText as QwenVLModel

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

    # ---- model ----
    if args.no_quantization:
        bnb_config = None
        dtype = torch.bfloat16
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = torch.bfloat16

    if is_main:
        logger.info("Loading model...")
    model = QwenVLModel.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
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

    # ---- LoRA ----
    if args.resume_from:
        if is_main:
            logger.info(f"Resuming from {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # ---- dataset ----
    old_prefix, new_prefix = None, None
    if args.path_prefix_remap:
        parts = args.path_prefix_remap.split(":", 1)
        if len(parts) == 2:
            old_prefix, new_prefix = parts
            if is_main:
                logger.info(f"Path remap: {old_prefix} → {new_prefix}")

    if is_main:
        logger.info("Loading dataset...")
    ds = load_from_disk(args.dataset_dir)
    hf_dataset = ds[args.split]
    if args.max_samples and args.max_samples > 0:
        hf_dataset = hf_dataset.select(range(min(args.max_samples, len(hf_dataset))))
    if is_main:
        logger.info(f"Training samples: {len(hf_dataset)}")

    dataset = GlasgowVisionDataset(hf_dataset, old_prefix, new_prefix)
    collate_fn = make_collate_fn(processor, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # ---- accelerate wrapping ----
    if accelerator is not None:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # ---- output dir ----
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- training ----
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        accum_count = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=not is_main,
        )

        for step, batch in enumerate(progress):
            if accelerator is None:
                batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # forward
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            running_loss += loss.item() * args.grad_accum_steps
            accum_count += 1

            # optimizer step
            if accum_count % args.grad_accum_steps == 0:
                if accelerator is not None:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

    # ---- final save ----
    if is_main:
        final_path = output_dir / "final_adapter"
        _save(model, processor, final_path, accelerator)
        logger.info(f"Final adapter saved to {final_path}")


def _save(model, processor, path: Path, accelerator=None):
    path.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(str(path))
    else:
        model.save_pretrained(str(path))
    processor.save_pretrained(str(path))


if __name__ == "__main__":
    main()
