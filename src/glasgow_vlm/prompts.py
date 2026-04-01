from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a geospatial inequality analyst. "
    "Infer deprivation from visual cues. "
    "Think step-by-step internally but do not reveal reasoning. "
    "Return JSON only."
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


def primary_modality_label(modality: str = "streetview") -> str:
    if modality == "streetview":
        return "street-view image"
    if modality == "satellite":
        return "satellite image"
    if modality == "ntl":
        return "nightlight image"
    return f"{modality.replace('_', ' ')} image"


def build_instruction(
    task: str = "ordinal",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
    primary_modality: str = "streetview",
) -> str:
    if modalities is None:
        resolved_modalities = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved_modalities = resolved_modalities + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(mod for mod in resolved_modalities if mod))
    modality_phrase = describe_modalities(modalities)
    primary_phrase = primary_modality_label(primary_modality)
    core_rules = (
        "Reason internally, but do not reveal reasoning. "
        "Use consistent visual cues across modalities when possible. "
        "Output one valid JSON object only. All fields must always be present, even if empty. Before outputting, verify that the JSON is complete and valid. "
        "Evidence must be a JSON object with keys streetview, satellite, and nightlight; each value must be a list of 3-5 short phrases (<8 words). Do not leave evidence empty; if a modality is unavailable, return [] only for that modality. "
        "visual_indicators must contain density, greenery, lighting, infrastructure, building_condition, land_use_mix, cleanliness, accessibility, vehicle_presence, housing_type, and vacancy, each as a float between 0 and 1. "
        "Infer these indicators from the images only; do not derive them from labels or prior knowledge. "
        "Use 0 for very low presence, 0.5 for moderate presence, and 1 for very high presence. "
        "High density usually maps to 0.8-1.0; sparse buildings usually map to 0.2-0.4. "
        "confidence must be a float between 0 and 1; lower it when evidence is weak, modalities disagree, or features are unclear. "
        "If unsure about any field, still return a best guess instead of leaving it empty. Evidence should not be omitted, and should contain at least one short phrase for each available modality. "
        "Do not output evidence as a paragraph. "
        "Do not use prior knowledge about the location. Nightlight evidence must describe only light intensity and spatial distribution. Do not infer object types or sources. Avoid brand names, place names, or semantic labels. Use only generic visual descriptions."
    )
    schema_example = (
        '{"predicted_quintile": 3, "confidence": 0.6, '
        '"evidence": {"streetview": ["brick houses", "narrow street", "few trees"], '
        '"satellite": ["dense buildings", "road grid", "small gardens"], '
        '"nightlight": ["dim lighting", "patchy glow", "dark edges"]}, '
        '"visual_indicators": {"density": 0.8, "greenery": 0.3, "lighting": 0.4, "infrastructure": 0.6, '
        '"building_condition": 0.5, "land_use_mix": 0.5, "cleanliness": 0.6, "accessibility": 0.5, '
        '"vehicle_presence": 0.4, "housing_type": 0.6, "vacancy": 0.2}}'
    )
    if task == "ordinal":
        return (
            f"Predict the deprivation quintile for this location using the {primary_phrase} together with {modality_phrase}. "
            f"{core_rules} Return JSON with fields: predicted_quintile, confidence, evidence, visual_indicators. "
            f"Example JSON: {schema_example}"
        )
    if task == "explain":
        return (
            f"Describe the shared visual cues across the {primary_phrase} together with {modality_phrase}, "
            "then predict whether the location is above or below the median deprivation level. "
            f"{core_rules} Return JSON with fields: above_median_deprivation, predicted_rank_band, confidence, evidence, visual_indicators. "
            f"Example JSON: {schema_example}"
        )
    return (
        f"Analyze the location using the {primary_phrase} together with {modality_phrase}. {core_rules} Return JSON with fields: predicted_quintile, confidence, evidence, visual_indicators. "
        f"Example JSON: {schema_example}"
    )


def build_answer(record: dict, task: str = "ordinal") -> str:
    quintile = record.get("deprivation_quintile")
    rank = record.get("deprivation_rank")
    if task == "explain":
        band = "unknown"
        if isinstance(rank, (int, float)):
            rank_int = int(rank)
            low = max(1, (rank_int // 1000) * 1000)
            high = min(5000, low + 999)
            band = f"{low}-{high}"
        payload = {
            "above_median_deprivation": bool(quintile is not None and int(quintile) >= 3),
            "predicted_rank_band": band,
            "confidence": 1.0,
        }
    else:
        payload = {
            "predicted_quintile": int(quintile) if quintile is not None else None,
            "confidence": 1.0,
        }
    return json.dumps(payload, ensure_ascii=False)

def build_prompt(
    record: dict,
    task: str = "ordinal",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
    primary_modality: str = "streetview",
) -> str:
    if modalities is None:
        resolved_modalities = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved_modalities = resolved_modalities + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(mod for mod in resolved_modalities if mod))
    modality_phrase = describe_modalities(modalities)
    location = "Location metadata unavailable."
    return f"{build_instruction(task, secondary_modality, tertiary_modality, modalities, primary_modality)} {location} Image modalities: {modality_phrase}."


def to_abs_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()





