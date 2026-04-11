from __future__ import annotations

import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a geospatial inequality analyst specialising in urban deprivation assessment "
    "in the United Kingdom, grounded in the Scottish Index of Multiple Deprivation (SIMD) framework. "
    "Infer deprivation from visual evidence in the provided images only. "
    "Do not use prior knowledge about specific locations, place names, or regional stereotypes. "
    "Do not directly equate a single visual cue to household income, personal health status, or actual crime rates. "
    "Think step-by-step internally but do not reveal reasoning. "
    "Return JSON only."
)


def modality_label(modality: str = "satellite") -> str:
    if modality == "ntl":
        return "nightlight patch"
    if modality == "satellite":
        return "satellite patch"
    return modality.replace("_", " ")


def describe_modalities(modalities: tuple[str, ...]) -> str:
    labels = [modality_label(m) for m in modalities]
    if not labels:
        return "the secondary image"
    if len(labels) == 1:
        return f"the {labels[0]}"
    if len(labels) == 2:
        return f"the {labels[0]} and {labels[1]}"
    return ", ".join(f"the {l}" for l in labels[:-1]) + f", and the {labels[-1]}"


def primary_modality_label(modality: str = "streetview") -> str:
    if modality == "streetview":
        return "street-view image"
    if modality == "satellite":
        return "satellite image"
    if modality == "ntl":
        return "nightlight image"
    return f"{modality.replace('_', ' ')} image"


_EVIDENCE_RULES = (
    "STEP 1 — Evidence extraction (describe only; do not draw deprivation conclusions here). "
    "streetview: observe street furniture, pavement condition, building facades and windows, road surface, "
    "sky openness, grass/shrubs/trees, people and vehicles, commercial frontages (open/closed/vacant), "
    "graffiti and vandalism, vacancy signs, environmental cleanliness. "
    "satellite: observe building density and layout, housing roof quality, road network and connectivity, "
    "vegetation cover, land-use patterns, bare or brownfield land. "
    "nightlight: describe ONLY light intensity (bright/dim/dark), spatial distribution (concentrated/dispersed/patchy), "
    "continuity (continuous/fragmented), and centroid position; do NOT infer object types, land use, or income. "
)

_INTERMEDIATE_RULES = (
    "STEP 2 — Derive the following eight intermediate variables from Step 1 evidence. "
    "For each, cite the supporting evidence phrases and give a score 1–10 (1=poorest quality, 10=best quality). "
    "Variables: "
    "(a) built_environment_quality — building condition, density, and urban layout; "
    "(b) infrastructure_adequacy — road network quality and completeness; "
    "(c) residential_environment_quality — housing maintenance, upkeep signs, surrounding environment; "
    "(d) connectivity_accessibility — road connectivity, transit indicators, spatial centrality; "
    "(e) commercial_economic_activity — commercial frontage activity, vacancy, and business signs; "
    "(f) night_time_activity_coverage — night-light intensity and spatial coverage as proxy for nocturnal activity and infrastructure; "
    "(g) open_space_greenery — visible vegetation, parks, green land cover; "
    "(h) spatial_marginality — edge location, poor connectivity, isolation cues. "
    "If evidence conflicts across modalities, explicitly note the conflict. "
    "If evidence is weak for a variable, use a mid-range score (4–7) and note uncertainty. "
)

_DIMENSION_RULES = (
    "STEP 3 — Internally score each SIMD dimension 1–10 (1=most deprived, 10=least deprived), "
    "drawing only on Step 1 evidence and Step 2 variables. "
    "Dimension-specific guidance: "
    "income — proxy via residential quality, maintenance, spatial vacancy, and commercial decline indicators; must be cautious. "
    "employment — proxy via commercial/industrial activity, transport links, and area vitality; do NOT equate 'industrial land nearby' with poor employment. "
    "health — proxy via green space, walkability environment, pollution exposure cues, open space, and active travel infrastructure. "
    "education — image evidence is weak; use school/educational facility visibility and environmental quality as weak proxies only; if uncertain, default to 5. "
    "housing — infer from housing condition, density, maintenance, roof quality (satellite), and surrounding environment. "
    "access — focus on road network, centrality, connectivity, transit cues, and night-time lighting continuity. "
    "crime — infer only from environmental management cues: graffiti, vandalism, security fixtures, abandoned sites, street lighting, and activity level; "
    "do NOT interpret visual appearance of people as a crime indicator. "
    "Scoring calibration: "
    "1–3 = strong multi-modal evidence of deprivation; "
    "4–5 = some risk signals, inconsistent evidence; "
    "6–7 = generally adequate with residual uncertainty; "
    "8–10 = strong multi-modal evidence of low deprivation. "
    "Avoid extreme scores (1 or 10) unless evidence is very strong. "
)

_OVERALL_RULE = (
    "STEP 4 — Compute overall using: "
    "0.12*income + 0.12*employment + 0.02*health + 0.01*education + 0.06*housing + 0.06*access + 0.04*crime. "
    "Round to two decimal places. "
)

_FORMAT_RULES = (
    "Output one valid JSON object only. "
    "The JSON must contain exactly these eight keys: "
    "income, employment, health, education, housing, access, crime (each an integer 1–10), "
    "and overall (a float). "
    "Do not include evidence, reasoning, intermediate variables, or any other fields in the output. "
    "All eight keys must always be present. Before outputting, verify the JSON is complete and valid."
)

_SCHEMA_EXAMPLE = json.dumps({
    "income": 4,
    "employment": 4,
    "health": 4,
    "education": 5,
    "housing": 4,
    "access": 5,
    "crime": 4,
    "overall": 4.18
}, ensure_ascii=False)


def build_instruction(
    task: str = "structured",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
    primary_modality: str = "streetview",
) -> str:
    if modalities is None:
        resolved = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved = resolved + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(m for m in resolved if m))

    modality_phrase = describe_modalities(modalities)
    primary_phrase = primary_modality_label(primary_modality)

    return (
        f"Analyze urban deprivation for this location using the {primary_phrase} "
        f"together with {modality_phrase}. "
        f"{_EVIDENCE_RULES}"
        f"{_INTERMEDIATE_RULES}"
        f"{_DIMENSION_RULES}"
        f"{_OVERALL_RULE} "
        f"{_FORMAT_RULES} "
        f"Example JSON: {_SCHEMA_EXAMPLE}"
    )


def build_prompt(
    record: dict,
    task: str = "structured",
    secondary_modality: str = "satellite",
    tertiary_modality: str | None = None,
    modalities: tuple[str, ...] | None = None,
    primary_modality: str = "streetview",
) -> str:
    if modalities is None:
        resolved = ("satellite", secondary_modality)
        if tertiary_modality:
            resolved = resolved + (tertiary_modality,)
        modalities = tuple(dict.fromkeys(m for m in resolved if m))

    modality_phrase = describe_modalities(modalities)
    return (
        f"{build_instruction(task, secondary_modality, tertiary_modality, modalities, primary_modality)} "
        f"Location metadata unavailable. Image modalities: {modality_phrase}."
    )


def to_abs_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()
