from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .prompts import build_answer, build_prompt


def load_jsonl(path: str | Path) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class GlasgowVLMJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, input_mode: str = "dual", task: str = "ordinal"):
        self.path = Path(jsonl_path)
        self.records = load_jsonl(self.path)
        self.input_mode = input_mode
        self.task = task

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        images: list[Image.Image] = []
        if self.input_mode in ("streetview", "dual", "triple"):
            images.append(self._load_image(record["streetview_path"]))
        if self.input_mode in ("satellite", "dual", "triple", "satellite_ntl"):
            images.append(self._load_image(record["satellite_path"]))
        if self.input_mode in ("triple", "satellite_ntl"):
            ntl_path = record.get("ntl_path")
            if not ntl_path:
                raise ValueError(f"Record {record.get('id')} is missing ntl_path for {self.input_mode} input mode")
            images.append(self._load_image(ntl_path))
        secondary_modality = record.get("secondary_modality", "satellite")
        tertiary_modality = record.get("tertiary_modality")
        if self.input_mode == "triple":
            modalities = ("satellite", "ntl")
        elif self.input_mode == "satellite_ntl":
            modalities = ("ntl",)
        elif self.input_mode == "dual" and secondary_modality == "ntl":
            modalities = ("ntl",)
        elif self.input_mode in ("satellite", "streetview"):
            modalities = ()
        else:
            modalities = ("satellite", secondary_modality)
        return {
            "id": record["id"],
            "record": record,
            "images": images,
            "prompt": build_prompt(
                record,
                self.task,
                secondary_modality=secondary_modality,
                tertiary_modality=tertiary_modality,
                modalities=modalities,
                primary_modality=("satellite" if self.input_mode == "satellite_ntl" else "streetview"),
            ),
            "answer": record.get("answer_json") or build_answer(record, self.task),
        }
