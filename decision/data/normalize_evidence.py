"""
Convert a parsed perception record into four text segments for the encoder.

Segments:
  [SAT]      satellite evidence phrases (deduped)
  [NTL]      nightlight evidence phrases (deduped)
  [SV]       all streetview phrases merged and deduped (set-pooled)
  [POI_TEXT] original POI summary string
"""

from __future__ import annotations

_MAX_SV_PHRASES = 60  # cap before truncation / attention pool


def _dedup(phrases: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for p in phrases:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return out


def build_segments(record: dict, max_sv_phrases: int = _MAX_SV_PHRASES) -> dict[str, str]:
    """Return {sat, ntl, sv, poi_text} text strings from a parsed record."""
    sat_text = "; ".join(_dedup(record["satellite"])) or "[no satellite evidence]"
    ntl_text = "; ".join(_dedup(record["nightlight"])) or "[no nightlight evidence]"

    all_sv: list[str] = []
    for img_phrases in record["streetview"]:
        all_sv.extend(img_phrases)
    sv_phrases = _dedup(all_sv)[:max_sv_phrases]
    sv_text = "; ".join(sv_phrases) if sv_phrases else "[no streetview evidence]"

    poi_text = record.get("poi_text", "(no POI recorded)")

    return {"sat": sat_text, "ntl": ntl_text, "sv": sv_text, "poi_text": poi_text}


def format_input(segments: dict[str, str]) -> str:
    """Concatenate segments into a single tagged string for a single-encoder path."""
    return (
        f"[SAT] {segments['sat']} "
        f"[NTL] {segments['ntl']} "
        f"[SV] {segments['sv']} "
        f"[POI] {segments['poi_text']}"
    )
