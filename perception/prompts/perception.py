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


# ── Domain-indicator scoring (text-only second-pass) ─────────────────────────

INDICATOR_KEYS = [
    "physical_disorder",
    "streetlight_presence",
    "building_security_features",
    "building_facade_condition",
    "commercial_vitality",
    "vehicle_quality",
    "green_space_visible",
    "active_travel_infra",
    "food_retail_type",
    "industrial_land_presence",
    "housing_density_type",
    "residential_upkeep",
    "vegetation_satellite",
    "road_surface_quality",
    "land_use_diversity",
    "nightlight_intensity",
    "air_quality_proxy",
]


# (name, SIMD domain, what to look at, 0/2/4 anchor descriptions)
DOMAIN_INDICATORS_SPEC: list[tuple[str, str, str, str]] = [
    ("physical_disorder", "Crime",
     "graffiti, litter, vandalism, broken windows in streetviews",
     "0=none visible; 2=some litter or minor graffiti; 4=pervasive disorder"),
    ("streetlight_presence", "Crime",
     "lamp posts, lit street furniture in streetviews",
     "0=none visible; 2=occasional lamp posts; 4=dense, regular lighting"),
    ("building_security_features", "Crime",
     "shutters, bars, fencing, security grilles in streetviews",
     "0=none; 2=isolated shutters/grilles; 4=pervasive bars/fencing"),
    ("building_facade_condition", "Income",
     "facade upkeep: paint, render, masonry across streetviews",
     "0=well kept; 2=patchy wear, peeling paint; 4=widespread damage"),
    ("commercial_vitality", "Employment",
     "active vs vacant shopfronts in streetviews and POI mix",
     "0=mostly vacant or no commerce; 2=mixed; 4=many active shopfronts"),
    ("vehicle_quality", "Income",
     "age and type of parked/visible vehicles in streetviews",
     "0=mostly old/derelict; 2=mixed mid-age; 4=many newer cars"),
    ("green_space_visible", "Health",
     "trees, parks, hedges from streetviews",
     "0=no greenery; 2=scattered trees; 4=large park/canopy access"),
    ("active_travel_infra", "Access",
     "pavement width, cycle lanes, crossings in streetviews",
     "0=none/narrow; 2=basic pavement; 4=wide pavements + cycle infra"),
    ("food_retail_type", "Health",
     "food retail mix in POI list (takeaways vs supermarkets vs grocers)",
     "0=only fast-food/takeaways or none; 2=mixed; 4=fresh-food retailers present"),
    ("industrial_land_presence", "Health",
     "industrial sheds, yards, brownfield in satellite",
     "0=none visible; 2=small yards; 4=large industrial premises"),
    ("housing_density_type", "Housing",
     "housing morphology in satellite (detached → tenement)",
     "0=detached/low-density; 2=terraced; 4=dense tenement/flats"),
    ("residential_upkeep", "Housing",
     "roof, garden, refuse condition in streetviews and satellite",
     "0=well kept; 2=some moss/clutter; 4=widespread neglect"),
    ("vegetation_satellite", "Health",
     "tree canopy and green cover from above (satellite)",
     "0=bare/grey; 2=partial canopy; 4=extensive vegetation"),
    ("road_surface_quality", "Access",
     "tarmac condition, road markings, potholes in streetviews",
     "0=poor/patched; 2=worn but usable; 4=smooth, well marked"),
    ("land_use_diversity", "Access",
     "mix of residential/commercial/amenity from POI + satellite",
     "0=monofunctional; 2=mixed; 4=highly diverse"),
    ("nightlight_intensity", "Income",
     "brightness and concentration in nightlight patch",
     "0=dark; 2=dim/scattered; 4=bright and concentrated"),
    ("air_quality_proxy", "Health",
     "traffic volume, industrial emissions, congestion proxies",
     "0=quiet, no industry; 2=light traffic; 4=heavy traffic + industry"),
]


_INDICATOR_SYSTEM_TASK = (
    "Indicator scoring task: given the visual + tabular evidence already extracted "
    "for one Glasgow datazone, assign each of the listed indicators an integer score "
    "from 0 to 4 and a short cue (<=8 words) citing the evidence that justifies it. "
    "Score only what the listed evidence supports — do NOT fabricate observations and "
    "do NOT label the area's deprivation level. If evidence is silent on an indicator, "
    "use score 0 with cue \"no evidence\"."
)


def _format_indicator_spec() -> str:
    lines = []
    for name, domain, what, anchors in DOMAIN_INDICATORS_SPEC:
        lines.append(f"- {name} (SIMD {domain}): look at {what}. Anchors: {anchors}.")
    return "\n".join(lines)


def _format_evidence_block(evidence: dict, max_streetviews: int = 6) -> str:
    """Compact one-line-per-modality serialization of evidence dict."""
    def _join(values) -> str:
        if isinstance(values, list):
            return "; ".join(str(v) for v in values if v)
        return str(values) if values else ""

    parts: list[str] = []
    sat = _join(evidence.get("satellite", []))
    if sat:
        parts.append(f"satellite: {sat}")
    ntl = _join(evidence.get("nightlight", []))
    if ntl:
        parts.append(f"nightlight: {ntl}")

    sv_keys = sorted(k for k in evidence.keys() if k.startswith("streetview_"))
    for k in sv_keys[:max_streetviews]:
        sv = _join(evidence.get(k, []))
        if sv:
            parts.append(f"{k}: {sv}")
    if len(sv_keys) > max_streetviews:
        parts.append(f"(+{len(sv_keys) - max_streetviews} more streetviews omitted)")

    poi = _join(evidence.get("POI", []))
    if poi:
        parts.append(f"POI: {poi}")
    general = _join(evidence.get("general", []))
    if general:
        parts.append(f"general: {general}")
    return "\n".join(parts)


_INDICATOR_FORMAT_RULES = (
    "Output exactly one JSON object: "
    "{\"domain_indicators\": {\"<indicator_name>\": {\"score\": <int 0-4>, "
    "\"cue\": \"<=8 words\"}, ...}}. "
    "Include all 17 indicators listed below in that exact key order. "
    "No prose outside JSON, no extra keys."
)


_INDICATOR_EXAMPLE = json.dumps({
    "domain_indicators": {
        "physical_disorder": {"score": 2, "cue": "graffiti on fence, scattered litter"},
        "streetlight_presence": {"score": 3, "cue": "regular lamp posts on both sides"},
    }
}, ensure_ascii=False)


def build_indicator_prompt(
    evidence: dict,
    poi_summary: str | None = None,
    max_streetviews: int = 6,
    minimal: bool = False,
) -> str:
    """Build a text-only prompt that asks the VLM to score 17 indicators.

    minimal=True drops the cue requirement and uses a leaner spec — used as
    a fallback retry when the full prompt fails to parse.
    """
    evidence_block = _format_evidence_block(evidence, max_streetviews=max_streetviews)
    if poi_summary and "POI:" not in evidence_block:
        evidence_block += f"\nPOI: {poi_summary}"

    if minimal:
        spec_lines = "\n".join(f"- {n}" for n in INDICATOR_KEYS)
        return (
            f"{_INDICATOR_SYSTEM_TASK}\n\n"
            f"Evidence for this datazone:\n{evidence_block}\n\n"
            f"Indicators (score each 0-4 integer):\n{spec_lines}\n\n"
            "Return JSON only: {\"domain_indicators\": "
            "{\"<name>\": {\"score\": <int 0-4>, \"cue\": \"\"}, ...}}. "
            "Include all 17 indicators."
        )

    return (
        f"{_INDICATOR_SYSTEM_TASK}\n\n"
        f"Evidence for this datazone:\n{evidence_block}\n\n"
        f"Indicators to score (in this order):\n{_format_indicator_spec()}\n\n"
        f"{_INDICATOR_FORMAT_RULES}\n"
        f"Example (truncated): {_INDICATOR_EXAMPLE}"
    )
