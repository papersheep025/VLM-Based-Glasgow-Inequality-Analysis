from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a geospatial inequality analyst specialising in urban deprivation assessment in the United Kingdom. "
    "You assess deprivation by observing visual cues such as building fabric, street maintenance, "
    "commercial activity, environmental quality, and signs of disorder — "
    "consistent with frameworks such as the Scottish Index of Multiple Deprivation (SIMD). "
    "Infer deprivation from visual evidence in the provided images only. "
    "Do not use prior knowledge about specific locations, place names, or regional stereotypes. "
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
        "visual_indicators must contain exactly these 15 keys: density, greenery, lighting, infrastructure, building_condition, land_use_mix, cleanliness, accessibility, vehicle_presence, housing_type, vacancy, commercial_activity, industrial_presence, graffiti_vandalism, security_features; each as a float between 0 and 1. "
        "Infer these indicators from the images only; do not derive them from labels or prior knowledge. "
        "Use 0 for very low presence, 0.5 for moderate presence, and 1 for very high presence. "
        "Calibration anchors: "
        "density: 0=open fields/sparse buildings, 0.8-1.0=dense urban. "
        "commercial_activity: 0=no shops/abandoned, 0.5=some retail, 1=active high street. "
        "industrial_presence: 0=no industrial land, 1=warehouses/factories dominate (infer from satellite). "
        "graffiti_vandalism: 0=no visible graffiti or damage, 0.5=scattered tags, 1=extensive vandalism (infer from street-view). "
        "security_features: 0=no visible security measures, 0.5=some bars or cameras, 1=heavy fortification (infer from street-view). "
        "graffiti_vandalism, security_features, and commercial_activity are best observed from street-view; industrial_presence is best inferred from satellite. "
        "If unsure about any field, still return a best guess instead of leaving it empty. Evidence should not be omitted, and should contain at least one short phrase for each available modality. "
        "Do not output evidence as a paragraph. "
        "Do not use prior knowledge about the location. Nightlight evidence must describe only light intensity and spatial distribution. Do not infer object types or sources. Avoid brand names, place names, or semantic labels. Use only generic visual descriptions."
    )
    schema_example = (
        '{"predicted_quintile": 3, '
        '"evidence": {"streetview": ["brick houses", "narrow street", "few trees"], '
        '"satellite": ["dense buildings", "road grid", "small gardens"], '
        '"nightlight": ["dim lighting", "patchy glow", "dark edges"]}, '
        '"visual_indicators": {"density": 0.8, "greenery": 0.3, "lighting": 0.4, "infrastructure": 0.6, '
        '"building_condition": 0.5, "land_use_mix": 0.5, "cleanliness": 0.6, "accessibility": 0.5, '
        '"vehicle_presence": 0.4, "housing_type": 0.6, "vacancy": 0.2, '
        '"commercial_activity": 0.3, "industrial_presence": 0.1, "graffiti_vandalism": 0.4, "security_features": 0.5}}'
    )
    explain_schema_example = (
        '{"evidence": {"streetview": ["brick houses", "narrow street", "few trees"], '
        '"satellite": ["dense buildings", "road grid", "small gardens"], '
        '"nightlight": ["dim lighting", "patchy glow", "dark edges"]}, '
        '"visual_indicators": {"density": 0.8, "greenery": 0.3, "lighting": 0.4, "infrastructure": 0.6, '
        '"building_condition": 0.5, "land_use_mix": 0.5, "cleanliness": 0.6, "accessibility": 0.5, '
        '"vehicle_presence": 0.4, "housing_type": 0.6, "vacancy": 0.2, '
        '"commercial_activity": 0.3, "industrial_presence": 0.1, "graffiti_vandalism": 0.4, "security_features": 0.5}}'
    )
    if task == "ordinal":
        return (
            f"Predict the deprivation quintile for this location using the {primary_phrase} together with {modality_phrase}. "
            f"{core_rules} Return JSON with fields: predicted_quintile, evidence, visual_indicators. "
            f"Example JSON: {schema_example}"
        )
    if task == "explain":
        return (
            f"Describe the shared visual cues across the {primary_phrase} together with {modality_phrase}, "
            "without predicting deprivation labels, rank bands, or confidence scores. "
            f"{core_rules} Return JSON with fields: evidence, visual_indicators. "
            f"Example JSON: {explain_schema_example}"
        )
    return (
        f"Analyze the location using the {primary_phrase} together with {modality_phrase}. {core_rules} Return JSON with fields: predicted_quintile, evidence, visual_indicators. "
        f"Example JSON: {schema_example}"
    )


def build_answer(record: dict, task: str = "ordinal") -> str:
    quintile = record.get("deprivation_quintile")
    if task == "explain":
        payload = {}
    else:
        payload = {
            "predicted_quintile": int(quintile) if quintile is not None else None,
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




