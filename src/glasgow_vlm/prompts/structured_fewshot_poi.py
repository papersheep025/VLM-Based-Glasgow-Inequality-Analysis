from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

from glasgow_vlm.prompts.structured_plus import (
    SYSTEM_PROMPT,
    build_instruction,
    build_prompt as _build_prompt_base,
    to_abs_uri,
)
from glasgow_vlm.prompts.structured_plus_fewshot import (
    FEW_SHOT_JSON_PATH,
    load_few_shot_examples,
    derive_intermediate_scores,
    build_fewshot_response,
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
    "load_poi_lookup",
    "format_poi_context",
    "FEW_SHOT_JSON_PATH",
    "DEFAULT_POI_CSV",
]

DEFAULT_POI_CSV = Path(__file__).resolve().parents[3] / "dataset" / "osm_poi" / "datazone_poi.csv"

_NOISE_SUBTYPES = frozenset({
    "bench", "waste_basket", "grit_bin", "post_box", "telephone",
    "signal", "switch", "buffer_stop", "bicycle_parking",
    "recycling", "vending_machine", "motorcycle_parking",
    "charging_station", "parking_space",
})

_NOISE_TYPES = frozenset({
    "historic", "craft", "sport",
})

_TYPE_DISPLAY = {
    "amenity": "Amenities",
    "shop": "Shops",
    "public_transport": "Public transport",
    "railway": "Railway",
    "leisure": "Leisure",
    "tourism": "Tourism",
    "office": "Offices",
    "emergency": "Emergency",
    "healthcare": "Healthcare",
}

_MAX_SUBTYPES_PER_TYPE = 8


def load_poi_lookup(
    csv_path: str | Path = DEFAULT_POI_CSV,
) -> dict[str, dict[str, Counter]]:
    lookup: dict[str, dict[str, Counter]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dz = row["datazone"]
            ptype = row["poi_type"]
            psub = row["poi_subtype"]
            if ptype in _NOISE_TYPES or psub in _NOISE_SUBTYPES:
                continue
            lookup.setdefault(dz, {}).setdefault(ptype, Counter())
            lookup[dz][ptype][psub] += 1
    return lookup


def format_poi_context(poi_by_type: dict[str, Counter] | None) -> str:
    if not poi_by_type:
        return ""
    lines = []
    for ptype in ("amenity", "shop", "public_transport", "railway",
                   "leisure", "tourism", "office", "emergency", "healthcare"):
        counter = poi_by_type.get(ptype)
        if not counter:
            continue
        label = _TYPE_DISPLAY.get(ptype, ptype)
        total = sum(counter.values())
        top = counter.most_common(_MAX_SUBTYPES_PER_TYPE)
        parts = [f"{cnt}x {sub}" for sub, cnt in top]
        lines.append(f"  {label} ({total}): {', '.join(parts)}")
    if not lines:
        return ""
    return (
        "\n\nSupplementary POI context (OpenStreetMap, this datazone):\n"
        + "\n".join(lines)
    )


def build_prompt(
    record: dict,
    task: str = "structured_plus",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
    primary_modality: str = "streetview",
    poi_context: str = "",
) -> str:
    base = _build_prompt_base(
        record, task, secondary_modality, tertiary_modality,
        modalities, primary_modality,
    )
    if poi_context:
        return base + poi_context
    return base


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
