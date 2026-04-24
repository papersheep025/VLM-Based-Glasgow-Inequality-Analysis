"""
Build a single-string caption from one dataset_v0 sample.

Two modes:
  - concat:    [SAT] ... [NTL] ... [SV] ... [POI] ...  (reuses format_input)
  - templated: CityLens-style English sentences grouped by streetview categories
               and POI counts (ported from the reference pipeline).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from decision.data.normalize_evidence import format_input


STREETVIEW_GROUPS: dict[str, list[str]] = {
    "sidewalk": ["pavement", "sidewalk", "curb", "path", "walkway"],
    "building": [
        "facade", "building", "buildings", "house", "houses", "apartment", "apartments",
        "roof", "roofs", "window", "windows", "balconies", "garage", "porch",
    ],
    "road": ["road", "asphalt", "driveway", "lane", "parking", "curb", "crossing"],
    "greenery": [
        "tree", "trees", "grass", "grassy", "bush", "bushes", "shrub", "shrubs",
        "hedge", "hedges", "lawn", "lawns", "flower", "flowers", "foliage",
    ],
    "street_furniture": [
        "lamp", "lamppost", "streetlight", "streetlights", "bin", "bins", "sign",
        "signage", "fence", "rail", "railing", "pole", "mailbox", "barrier",
    ],
    "mobility": [
        "car", "cars", "vehicle", "vehicles", "van", "bus", "bike", "bikes",
        "person", "people", "pedestrian", "pedestrians", "traffic",
    ],
    "condition": [
        "crack", "cracks", "worn", "wear", "uneven", "smooth", "clean", "debris",
        "graffiti", "stain", "stains", "puddle", "repair", "overgrown",
    ],
    "activity": [
        "shop", "shops", "store", "storefront", "frontage", "commercial", "industrial",
        "warehouse", "vacant", "sign", "bus shelter",
    ],
}

_GROUP_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    name: [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in keys]
    for name, keys in STREETVIEW_GROUPS.items()
}

_NO_EVIDENCE_MARKERS = {
    "[no satellite evidence]",
    "[no nightlight evidence]",
    "[no streetview evidence]",
}


def _split_segment(text: str) -> list[str]:
    """Invert the `; `-join used by normalize_evidence.build_segments."""
    if not text or text.strip() in _NO_EVIDENCE_MARKERS:
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.strip().replace("_", " ").split())


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _pick_top_phrases(phrases: list[str], limit: int = 4) -> list[str]:
    phrases = [_normalize_phrase(p) for p in phrases if str(p).strip()]
    counter: Counter[str] = Counter(p.lower() for p in phrases)
    display: dict[str, str] = {}
    for p in phrases:
        display.setdefault(p.lower(), p)
    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [display[key] for key, _ in ranked[:limit]]


def _top_grouped_streetview_phrases(sv_phrases: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {name: [] for name in STREETVIEW_GROUPS}
    remainder: list[str] = []
    for phrase in sv_phrases:
        p = _normalize_phrase(phrase)
        matched = False
        for group, patterns in _GROUP_PATTERNS.items():
            if any(pat.search(p) for pat in patterns):
                grouped[group].append(p)
                matched = True
        if not matched:
            remainder.append(p)
    grouped["other"] = remainder
    return {k: _pick_top_phrases(v, 3) for k, v in grouped.items() if v}


def _phrase_list_to_sentence(prefix: str, phrases: list[str]) -> str:
    phrases = _dedupe_keep_order(phrases)
    if not phrases:
        return ""
    if len(phrases) == 1:
        body = phrases[0].lower()
    else:
        body = ", ".join(p.lower() for p in phrases[:-1]) + f", and {phrases[-1].lower()}"
    return f"{prefix}{body}."


def _poi_counts_to_phrases(poi_counts: dict | None) -> list[str]:
    if not poi_counts:
        return ["no listed POI entries"]
    total = sum(int(v) for v in poi_counts.values())
    if total == 0:
        return ["no listed POI entries"]
    phrases: list[str] = []
    for key, label in (
        ("public_transport", "public transport"),
        ("shop", "retail"),
        ("amenity", "amenity"),
        ("healthcare", "healthcare"),
        ("railway", "railway"),
        ("office", "office"),
        ("tourism", "tourism"),
    ):
        cnt = int(poi_counts.get(key, 0))
        if cnt > 0:
            phrases.append(f"{cnt} {label} entries")
    if not phrases:
        phrases.append(f"{total} total POI entries")
    return phrases


def build_caption_concat(sample: dict) -> str:
    """Return the 4 tagged segments concatenated into one string."""
    return format_input({
        "sat": sample["sat"],
        "ntl": sample["ntl"],
        "sv": sample["sv"],
        "poi_text": sample["poi_text"],
    })


def build_caption_templated(sample: dict) -> str:
    """Return a CityLens-style English caption synthesised from the sample."""
    sat_phrases = _dedupe_keep_order(
        _normalize_phrase(p) for p in _split_segment(sample.get("sat", ""))
    )[:6]
    ntl_phrases = _dedupe_keep_order(
        _normalize_phrase(p) for p in _split_segment(sample.get("ntl", ""))
    )[:5]
    sv_all = _split_segment(sample.get("sv", ""))
    grouped = _top_grouped_streetview_phrases(sv_all)

    sentences: list[str] = [
        _phrase_list_to_sentence(
            "Satellite imagery suggests ",
            sat_phrases or ["a mixed urban form with limited explicit cues"],
        ),
        _phrase_list_to_sentence(
            "Nighttime light patterns show ",
            ntl_phrases or ["weak and spatially fragmented illumination"],
        ),
    ]

    dominant_sv: list[str] = []
    for name in ("road", "building", "sidewalk", "greenery",
                 "street_furniture", "mobility", "condition", "activity"):
        dominant_sv.extend(grouped.get(name, [])[:2])
    dominant_sv = _dedupe_keep_order(dominant_sv)[:10]
    sentences.append(_phrase_list_to_sentence(
        "Across the street-view images, recurring elements include ",
        dominant_sv or ["mixed street conditions and low-rise built forms"],
    ))

    sv_variation = grouped.get("other", [])[:4]
    if sv_variation:
        sentences.append(_phrase_list_to_sentence(
            "Local variation is also visible through ",
            sv_variation,
        ))

    poi_phrases = _poi_counts_to_phrases(sample.get("poi_counts"))
    sentences.append(_phrase_list_to_sentence(
        "The POI summary indicates ",
        poi_phrases,
    ))

    sentences.append(
        "Taken together, the modalities describe the built form, open-space pattern, "
        "lighting distribution, street-level maintenance cues, and local service "
        "context of this area."
    )
    return " ".join(s for s in sentences if s)


def build_captions_modality_sep(samples: list[dict]) -> list[dict[str, str]]:
    """Return per-modality text dicts for separate SBERT encoding.

    Keys: sat, ntl, sv, poi.  sv retains the semicolon-separated phrase list so
    the encoder can per-phrase-encode and mean-pool without hitting the 256-token
    limit.
    """
    return [
        {
            "sat": s.get("sat", "[no satellite evidence]"),
            "ntl": s.get("ntl", "[no nightlight evidence]"),
            "sv":  s.get("sv",  "[no streetview evidence]"),
            "poi": s.get("poi_text", "(no POI recorded)"),
        }
        for s in samples
    ]


def build_captions(samples: list[dict], mode: str) -> list[str]:
    if mode == "concat":
        return [build_caption_concat(s) for s in samples]
    if mode == "templated":
        return [build_caption_templated(s) for s in samples]
    raise ValueError(f"unknown caption mode: {mode!r} (expected 'concat' or 'templated')")
