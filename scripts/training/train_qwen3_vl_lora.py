# -*- coding: utf-8 -*-
"""QLoRA fine-tuning for Qwen3-VL-8B on Glasgow deprivation data using mlx-vlm.

Loads the training dataset built by build_lora_training_data.py,
dynamically loads images at training time, and runs LoRA fine-tuning
via the mlx-vlm trainer.

Usage:
    python scripts/training/train_qwen3_vl_lora.py

    python scripts/training/train_qwen3_vl_lora.py \
        --model-path /path/to/Qwen3-VL-8B-Instruct-MLX-4bit \
        --epochs 2 --batch-size 1 --learning-rate 2e-5

    # Resume from checkpoint
    python scripts/training/train_qwen3_vl_lora.py \
        --adapter-path outputs/lora_adapters/adapters.safetensors
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from mlx_vlm.trainer import Trainer, save_adapter
from mlx_vlm.trainer.trainer import get_prompt, grad_checkpoint
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load, load_image_processor, prepare_inputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = str(
    Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3-VL-8B-Instruct-MLX-4bit"
)
DEFAULT_DATASET_DIR = str(ROOT / "dataset" / "lora_training_data")
DEFAULT_OUTPUT_PATH = str(ROOT / "outputs" / "lora_adapters" / "adapters.safetensors")


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for Qwen3-VL-8B")
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--adapter-path", type=str, default=None, help="Resume from adapter checkpoint.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--steps", type=int, default=0, help="Steps per epoch (0=auto).")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps.")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    p.add_argument("--max-samples", type=int, default=0, help="Limit training samples (0=all).")
    p.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing.")
    return p.parse_args()


class GlasgowVisionDataset:
    """Dataset that lazily loads images from file paths at training time."""

    def __init__(self, hf_dataset, config, processor, image_resize_shape=None):
        self.dataset = hf_dataset
        self.config = config
        self.processor = processor
        self.image_resize_shape = image_resize_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            items = [self._process_single(i) for i in indices]
            if not items:
                return {}
            batch = {}
            for key in items[0]:
                vals = [item[key] for item in items]
                if vals[0] is None:
                    batch[key] = None
                else:
                    batch[key] = mx.concatenate(vals, axis=0) if isinstance(vals[0], mx.array) else vals
            return batch
        return self._process_single(idx)

    def _process_single(self, idx):
        item = self.dataset[idx]

        image_paths = item["images"]
        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            if self.image_resize_shape:
                img = img.resize(tuple(self.image_resize_shape))
            images.append(img)

        messages_raw = item["messages"]
        if isinstance(messages_raw, str):
            conversations = json.loads(messages_raw)
        else:
            conversations = messages_raw

        prompt = get_prompt(self.config["model_type"], self.processor, conversations)

        image_token_index = self.config.get("image_token_index") or self.config.get("image_token_id")

        inputs = prepare_inputs(
            self.processor,
            images,
            [prompt],
            image_token_index,
            self.image_resize_shape,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = mx.ones_like(input_ids)

        kwargs = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": mask,
            **kwargs,
        }


def main():
    args = parse_args()

    logger.info(f"Loading model from {args.model_path}")
    model, processor = load(
        args.model_path, processor_config={"trust_remote_code": True}
    )
    config = model.config.__dict__
    logger.info(f"Model type: {config.get('model_type')}")

    if args.adapter_path:
        logger.info(f"Loading adapter from {args.adapter_path}")
        from mlx.utils import tree_unflatten
        import safetensors.mlx
        adapter_weights = safetensors.mlx.load(args.adapter_path)
        model.load_weights(list(adapter_weights.items()), strict=False)

    logger.info(f"Loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    hf_dataset = ds[args.split]

    if args.max_samples and args.max_samples > 0:
        hf_dataset = hf_dataset.select(range(min(args.max_samples, len(hf_dataset))))

    logger.info(f"Dataset size: {len(hf_dataset)} samples")

    dataset = GlasgowVisionDataset(
        hf_dataset, config, processor,
        image_resize_shape=args.image_resize_shape,
    )

    if args.grad_checkpoint:
        logger.info("Enabling gradient checkpointing")
        for layer in model.language_model.model.layers:
            grad_checkpoint(layer)

    logger.info("Setting up LoRA")
    list_of_modules = find_all_linear_names(model.language_model)
    model = get_peft_model(
        model, list_of_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    trainable = sum(p.size for _, p in model.trainable_parameters().items()) if hasattr(model.trainable_parameters(), 'items') else sum(p.size for _, p in model.trainable_parameters())
    total = sum(p.size for _, p in mx.utils.tree_flatten(model.parameters()))
    logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    logger.info("Setting up optimizer")
    optimizer = optim.Adam(learning_rate=args.learning_rate)

    logger.info("Setting up trainer")
    trainer = Trainer(model, optimizer)

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    logger.info("Starting training")
    for epoch in range(args.epochs):
        steps = args.steps if args.steps > 0 else len(dataset) // args.batch_size

        progress_bar = tqdm(range(steps), desc=f"Epoch {epoch+1}/{args.epochs}", position=0, leave=True)
        for i in progress_bar:
            batch = dataset[i * args.batch_size : (i + 1) * args.batch_size]
            loss = trainer.train_step(batch)
            mx.eval(model.parameters(), optimizer.state)

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": f"{i+1}/{steps}",
            })

            if (i + 1) % args.print_every == 0:
                tqdm.write(f"  Epoch {epoch+1} | Step {i+1}/{steps} | Loss: {loss.item():.4f}")

            if (i + 1) % args.save_every == 0:
                ckpt_path = str(output_dir / f"checkpoint_ep{epoch+1}_step{i+1}.safetensors")
                save_adapter(model, ckpt_path)
                tqdm.write(f"  Checkpoint saved: {ckpt_path}")

    save_adapter(model, args.output_path)
    logger.info(f"Final adapter saved to {args.output_path}")


if __name__ == "__main__":
    main()
