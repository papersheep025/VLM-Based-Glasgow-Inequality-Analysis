"""
Prompt construction for Route B (LLM-as-regressor with CoT).

Evidence text → 3-step CoT prompt that asks the model to:
  STEP 1  summarise each modality's phrases (cite verbatim);
  STEP 2  derive 8 intermediate variables with supporting cites;
  STEP 3  assign 7 SIMD domain scores (integer 1–10, higher = more deprived).

Output format: single JSON object (``reasoning`` block + 7 integer scores).

For the rationaliser (teacher), we append the ground-truth scores to the
prompt and ask the model to produce the CoT that would have led to those
scores — still under the cite-only constraint.
"""

from __future__ import annotations

from decision.data.targets import DOMAINS


_DOMAIN_KEYS = [d.lower() for d in DOMAINS]  # ["income", ..., "housing"]


SYSTEM_PROMPT = (
    "You are a geospatial inequality analyst grounded in the Scottish Index of Multiple Deprivation (SIMD). "
    "You have ONLY the text evidence provided below — no prior knowledge of specific places. "
    "Every phrase you cite in support of an argument MUST appear verbatim in the provided evidence "
    "(case-sensitive substring match). If you cannot cite evidence for a claim, drop the claim. "
    "Think step-by-step; return JSON only."
)


_INSTRUCTION = f"""
STEP 1 — Evidence grounding. For each of the 4 segments ([SAT], [NTL], [SV], [POI]), list 2-4 salient
phrases verbatim from the evidence; do not paraphrase.

STEP 2 — Intermediate variables (each with score 1-10 where 1=poorest quality, 10=best quality,
cited support from STEP 1, and a short interpretation):
  built_environment_quality, infrastructure_adequacy, residential_environment_quality,
  connectivity_accessibility, commercial_economic_activity, night_time_activity_coverage,
  open_space_greenery, spatial_marginality.

STEP 3 — SIMD domain scores. Output an integer 1-10 for each of the 7 domains
({", ".join(_DOMAIN_KEYS)}), higher = MORE deprived. Provide per-domain reasoning citing STEP 1/2.

Return a single JSON object with this shape (no prose outside JSON):
{{
  "reasoning": {{
    "evidence":  {{ "sat": [...], "ntl": [...], "sv": [...], "poi": [...] }},
    "intermediate_variables": {{ "<name>": {{"score": int, "support": [str], "interpretation": str}}, ... }},
    "dimension_reasoning": {{ "<domain>": str, ... }}
  }},
  {", ".join(f'"{k}": int' for k in _DOMAIN_KEYS)}
}}
"""


def _format_evidence(sample: dict) -> str:
    """Render the 4 segments exactly as the model will see them."""
    sv_block = sample.get("sv") or "(none)"
    return (
        f"[SAT] {sample.get('sat') or '(none)'}\n"
        f"[NTL] {sample.get('ntl') or '(none)'}\n"
        f"[SV]  {sv_block}\n"
        f"[POI] {sample.get('poi_text') or '(no POI recorded)'}"
    )


def build_predict_messages(sample: dict) -> list[dict]:
    """Messages for plain prediction (no ground truth)."""
    user = (
        f"Evidence:\n{_format_evidence(sample)}\n\n"
        f"Task:\n{_INSTRUCTION.strip()}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_rationalize_messages(sample: dict, targets_raw: dict[str, int]) -> list[dict]:
    """Messages for teacher rationalisation: evidence + ground truth → CoT."""
    gt = ", ".join(f"{d.lower()}={int(targets_raw[d])}" for d in DOMAINS if d in targets_raw)
    user = (
        f"Evidence:\n{_format_evidence(sample)}\n\n"
        f"Known ground-truth domain scores (use them as the target; your job is to articulate the "
        f"reasoning that justifies these scores, strictly citing the evidence above):\n{gt}\n\n"
        f"Task:\n{_INSTRUCTION.strip()}\n\n"
        f"Your STEP 3 scores MUST equal the ground truth exactly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
