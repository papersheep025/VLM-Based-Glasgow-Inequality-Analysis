from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a geospatial inequality analyst. "
    "Use the provided images to infer deprivation."
)


def deprivation_level_from_quintile(quintile: int | None) -> str:
    if quintile is None:
        return "unknown"
    if quintile <= 2:
        return "high deprivation"
    return "low deprivation"


def modality_label(modality: str = "satellite") -> str:
    if modality == "ntl":
        return "nightlight patch"
    if modality == "satellite":
        return "satellite patch"
    return modality.replace("_", " ")


def describe_modalities(modalities: tuple[str, ...]) -> str:
    labels = [modality_label(modality) for modality in modalities]
    if not labels:
        return "the secondary image"
    if len(labels) == 1:
        return f"the {labels[0]}"
    if len(labels) == 2:
        return f"the {labels[0]} and {labels[1]}"
    return ", ".join(f"the {label}" for label in labels[:-1]) + f", and the {labels[-1]}"


def build_instruction(
    task: str = "ordinal",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
) -> str:
    if modalities is None:
        resolved_modalities = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved_modalities = resolved_modalities + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(mod for mod in resolved_modalities if mod))
    modality_phrase = describe_modalities(modalities)
    if task == "ordinal":
        return (
            f"Predict the Glasgow SIMD deprivation quintile for this location using the street-view image together with {modality_phrase}. "
            "Return only valid JSON with fields: predicted_quintile, confidence, evidence."
        )
    if task == "rank":
        return (
            f"Predict the SIMD rank band for this location using the street-view image together with {modality_phrase}. "
            "Return only valid JSON with fields: predicted_rank_band, confidence, evidence."
        )
    if task == "explain":
        return (
            f"Describe the shared visual cues across the street-view image together with {modality_phrase}, "
            "then predict whether the location is above or below the Glasgow median deprivation level. "
            "Return only valid JSON with fields: above_median_deprivation, confidence, evidence."
        )
    return (
        "Analyze the location and return only valid JSON with fields: "
        "predicted_quintile, confidence, evidence."
    )


def build_answer(record: dict, task: str = "ordinal") -> str:
    quintile = record.get("deprivation_quintile")
    rank = record.get("deprivation_rank")
    payload: dict[str, object]

    if task == "rank":
        band = "unknown"
        if isinstance(rank, (int, float)):
            rank_int = int(rank)
            low = max(1, (rank_int // 1000) * 1000)
            high = min(5000, low + 999)
            band = f"{low}-{high}"
        payload = {
            "predicted_rank_band": band,
            "confidence": 1.0,
            "evidence": [],
        }
    elif task == "explain":
        payload = {
            "above_median_deprivation": bool(quintile is not None and int(quintile) >= 3),
            "confidence": 1.0,
            "evidence": [],
        }
    else:
        payload = {
            "predicted_quintile": int(quintile) if quintile is not None else None,
            "confidence": 1.0,
            "evidence": [],
        }
    return json.dumps(payload, ensure_ascii=False)


def build_prompt(
    record: dict,
    task: str = "ordinal",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
) -> str:
    datazone = record.get("datazone", "unknown")
    lat = record.get("lat")
    lon = record.get("lon")
    if modalities is None:
        resolved_modalities = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved_modalities = resolved_modalities + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(mod for mod in resolved_modalities if mod))
    modality_phrase = describe_modalities(modalities)
    location = f"DataZone={datazone}, lat={lat}, lon={lon}"
    return f"{build_instruction(task, secondary_modality, tertiary_modality, modalities)} Location metadata: {location}. Image modalities: {modality_phrase}."


def to_abs_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()
