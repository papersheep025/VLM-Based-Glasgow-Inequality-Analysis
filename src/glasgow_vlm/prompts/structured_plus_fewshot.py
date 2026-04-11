from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from glasgow_vlm.prompts.structured_plus import (
    SYSTEM_PROMPT,
    build_instruction,
    build_prompt,
    to_abs_uri,
)

__all__ = [
    "SYSTEM_PROMPT",
    "build_instruction",
    "build_prompt",
    "to_abs_uri",
    "load_few_shot_examples",
    "derive_intermediate_scores",
    "build_fewshot_response",
    "build_fewshot_user_content",
]

FEW_SHOT_JSON_PATH = Path(__file__).resolve().parents[3] / "dataset" / "few_shot_examples" / "few_shot_examples.json"


def _clamp(v: float, lo: int = 1, hi: int = 10) -> int:
    return max(lo, min(hi, round(v)))


def load_few_shot_examples(
    json_path: str | Path = FEW_SHOT_JSON_PATH,
    quintiles: list[int] | None = None,
    count: int | None = None,
) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    examples.sort(key=lambda x: x["quintile"])
    if quintiles is not None:
        qs = set(quintiles)
        examples = [e for e in examples if e["quintile"] in qs]
    if count is not None:
        examples = examples[:count]
    return examples


def derive_intermediate_scores(ground_truth: dict) -> dict:
    inc = ground_truth["income"]
    emp = ground_truth["employment"]
    hea = ground_truth["health"]
    hou = ground_truth["housing"]
    acc = ground_truth["access"]
    cri = ground_truth["crime"]
    return {
        "built_environment_quality_score": _clamp(hou),
        "infrastructure_adequacy_score": _clamp(0.5 * acc + 0.5 * hou),
        "residential_environment_quality_score": _clamp(0.6 * hou + 0.4 * hea),
        "connectivity_accessibility_score": _clamp(acc),
        "commercial_economic_activity_score": _clamp(0.5 * emp + 0.5 * inc),
        "night_time_activity_coverage_score": _clamp(0.4 * acc + 0.3 * emp + 0.3 * cri),
        "open_space_greenery_score": _clamp(hea),
        "spatial_marginality_score": _clamp(11 - acc),
    }


def build_fewshot_response(example: dict) -> str:
    gt = example["ground_truth_scores"]
    result = {
        "income": gt["income"],
        "employment": gt["employment"],
        "health": gt["health"],
        "education": gt["education"],
        "housing": gt["housing"],
        "access": gt["access"],
        "crime": gt["crime"],
        "overall": gt["overall"],
    }
    result.update(derive_intermediate_scores(gt))
    return json.dumps(result, ensure_ascii=False)


def build_fewshot_user_content(
    example: dict,
    input_mode: str,
    prompt_text: str,
    encode_fn: Callable[[str | Path], str],
) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt_text}]
    if input_mode in ("streetview", "dual", "triple"):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_fn(example["streetview_path"])},
        })
    if input_mode in ("satellite", "dual", "triple", "satellite_ntl"):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_fn(example["satellite_path"])},
        })
    if input_mode in ("satellite_ntl", "triple"):
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_fn(example["ntl_path"])},
        })
    return content
