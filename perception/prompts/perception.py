from __future__ import annotations

import json


SYSTEM_PROMPT = (
    "You are a visual perception assistant for urban scene analysis. "
    "Your role is to describe what is visible across the provided modalities and the tabular POI signal. "
    "Do NOT infer deprivation level, income, household wealth, crime rate, or make socioeconomic judgments. "
    "Stay strictly descriptive. Be concise. Return JSON only."
)


_EVIDENCE_RULES = (
    "For each modality, extract 3-6 short descriptive phrases (each <=8 words). "
    "satellite: building density and layout, roof quality, road network, vegetation cover, "
    "land-use pattern, brownfield or bare land. "
    "nightlight: ONLY light intensity (bright/dim/dark), spatial distribution "
    "(concentrated/dispersed/patchy), continuity (continuous/fragmented); do NOT infer object types or land use. "
    "streetview (per image, one entry per image): pavement condition, building facades and windows, "
    "road surface, street furniture, greenery, commercial frontages (open/closed/vacant), "
    "graffiti/vandalism, cleanliness, people and vehicles if visible. "
    "POI: using the POI summary table only, extract 3-6 phrases about facility mix, service access, "
    "and commercial density. "
    "general: 2-4 cross-modal observations noting agreement or conflict between modalities; "
    "do NOT summarise deprivation or socioeconomic status."
)


_FORMAT_RULES = (
    "Output exactly one JSON object with this shape: "
    "{\"reasoning_json\": {\"evidence\": {"
    "\"satellite\": [...], "
    "\"nightlight\": [...], "
    "\"streetview_00\": [...], \"streetview_01\": [...], ... , "
    "\"POI\": [...], "
    "\"general\": [...]"
    "}}} "
    "Use one streetview_NN key per provided street-view image, zero-padded 2-digit index, "
    "in the same order as the images. Each value is a list of short descriptive phrases. "
    "No extra keys, no prose outside JSON, no deprivation scoring."
)


_SCHEMA_EXAMPLE = json.dumps({
    "reasoning_json": {
        "evidence": {
            "satellite": ["dense terraced housing", "narrow road grid", "limited green space"],
            "nightlight": ["dim uniform glow", "patchy at edges", "concentrated centre"],
            "streetview_00": ["cracked pavement", "graffiti on walls", "closed shopfront"],
            "streetview_01": ["open grocery store", "well-kept facade", "light traffic"],
            "POI": ["mostly convenience amenities", "few healthcare facilities", "no schools nearby"],
            "general": ["streetview consistent with dense satellite layout",
                        "NTL brightness matches active POI cluster"]
        }
    }
}, ensure_ascii=False)


def _poi_block(poi_summary: str | None) -> str:
    if not poi_summary:
        return "POI data: none available."
    return f"POI data (tabular summary):\n{poi_summary}"


def build_instruction(n_streetviews: int, poi_summary: str | None = None) -> str:
    sv_keys = ", ".join(f"streetview_{i:02d}" for i in range(n_streetviews)) or "(none)"
    image_order = (
        f"Image order (must match JSON keys): satellite, nightlight, {sv_keys}."
        if n_streetviews > 0
        else "Image order (must match JSON keys): satellite, nightlight."
    )
    return (
        "Perception task: describe the urban scene across modalities for one datazone. "
        f"Inputs: 1 satellite patch, 1 nightlight patch, {n_streetviews} street-view image(s), "
        "and a POI summary. "
        f"{image_order} "
        f"{_EVIDENCE_RULES} "
        f"{_FORMAT_RULES} "
        f"{_poi_block(poi_summary)} "
        f"Example: {_SCHEMA_EXAMPLE}"
    )


def build_prompt(record: dict) -> str:
    n_sv = len(record.get("streetview_paths", []))
    return build_instruction(n_sv, record.get("poi_summary"))


# ── Single-image prompts ──────────────────────────────────────────────────────

_SATELLITE_SINGLE = (
    "Describe the urban scene in this satellite image. "
    "Extract 3-6 short descriptive phrases (each <=8 words) about: "
    "building density and layout, roof quality, road network, vegetation cover, "
    "land-use pattern, brownfield or bare land. "
    'Return JSON only: {"phrases": [...]}'
)

_NTL_SINGLE = (
    "Describe the nightlight pattern in this image. "
    "Extract 3-5 short descriptive phrases (each <=8 words) about: "
    "light intensity (bright/dim/dark), spatial distribution (concentrated/dispersed/patchy), "
    "continuity (continuous/fragmented). "
    "Do NOT infer object types or land use. "
    'Return JSON only: {"phrases": [...]}'
)

_STREETVIEW_SINGLE = (
    "Describe the urban scene in this street-view image. "
    "Extract 3-5 short descriptive phrases (each <=8 words). "
    "Only describe what is visually distinctive or noteworthy: "
    "pavement condition, building facades, road surface quality, "
    "street furniture, greenery, commercial frontages if present, "
    "graffiti or vandalism if present, vehicles or people if visible. "
    "Do NOT include generic negatives like 'no graffiti visible' or 'no people visible' "
    "unless the absence is striking in context. "
    'Return JSON only: {"phrases": [...]}'
)


_STREETVIEW_FALLBACK = (
    "Describe what you see in this street-view image. "
    "Extract 3-5 short phrases (each <=8 words) about: "
    "road surface, buildings, greenery, vehicles, street furniture. "
    'Return JSON only: {"phrases": [...]}'
)


def build_single_image_prompt(modality: str, fallback: bool = False) -> str:
    if modality == "satellite":
        return _SATELLITE_SINGLE
    if modality == "nightlight":
        return _NTL_SINGLE
    return _STREETVIEW_FALLBACK if fallback else _STREETVIEW_SINGLE
