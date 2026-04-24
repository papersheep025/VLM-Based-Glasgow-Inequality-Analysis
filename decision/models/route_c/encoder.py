"""
Caption encoders for Route C.

Supported backends:
  - bert:  AutoModel + attention-mask mean pooling.
           Default model: ``bert-base-uncased``.
  - sbert: sentence_transformers.SentenceTransformer.
           Default model: ``sentence-transformers/all-MiniLM-L6-v2``.
           Uses ``normalize_embeddings=True`` for downstream Ridge stability.

Results are stored as a dict with datazone ordering + a ``meta`` block so the
CV runner can detect stale caches across backend / model / caption-mode changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _encode_bert(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    outs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**inputs).last_hidden_state              # (B, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1).float()   # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        outs.append(pooled.cpu().float().numpy())
    return np.vstack(outs)


def _encode_sbert(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


def encode_texts(
    texts: list[str],
    backend: str,
    model_name: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    device = device or _auto_device()
    if backend == "bert":
        return _encode_bert(texts, model_name, batch_size, max_length, device)
    if backend == "sbert":
        return _encode_sbert(texts, model_name, batch_size, device)
    raise ValueError(f"unknown encoder backend: {backend!r}")


def encode_modality_sep(
    segments: list[dict[str, str]],
    backend: str,
    model_name: str,
    batch_size: int = 32,
    max_length: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Encode sat / ntl / sv / poi modalities separately and concatenate.

    sv uses per-phrase mean-pooling: each "; "-separated phrase is encoded
    individually (≤8 words each), so no phrase is ever truncated by the
    256-token limit.  Returns shape (n_samples, 4 * embed_dim).
    """
    device = device or _auto_device()
    n = len(segments)

    # SAT, NTL, POI: short strings, encode per-sample as a flat batch.
    fixed_embs: dict[str, np.ndarray] = {}
    for key in ("sat", "ntl", "poi"):
        texts = [s[key] for s in segments]
        fixed_embs[key] = encode_texts(texts, backend, model_name, batch_size, max_length, device)

    dim = fixed_embs["sat"].shape[1]

    # SV: collect all phrases across samples, batch-encode, then mean-pool per sample.
    all_sv_phrases: list[str] = []
    boundaries: list[tuple[int, int]] = []

    for s in segments:
        sv = s["sv"].strip()
        if not sv or sv == "[no streetview evidence]":
            phrases = [sv or "no streetview evidence"]
        else:
            phrases = [p.strip() for p in sv.split(";") if p.strip()]
            if not phrases:
                phrases = [sv]
        boundaries.append((len(all_sv_phrases), len(all_sv_phrases) + len(phrases)))
        all_sv_phrases.extend(phrases)

    all_sv_embs = encode_texts(all_sv_phrases, backend, model_name, batch_size, max_length, device)

    sv_embs = np.zeros((n, dim), dtype=np.float32)
    for i, (start, end) in enumerate(boundaries):
        sv_embs[i] = all_sv_embs[start:end].mean(axis=0)

    return np.concatenate(
        [fixed_embs["sat"], fixed_embs["ntl"], sv_embs, fixed_embs["poi"]], axis=1
    ).astype(np.float32)


def save_caption_cache(
    path: str | Path,
    datazones: list[str],
    X: np.ndarray,
    meta: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"datazones": datazones, "X": torch.from_numpy(X), "meta": meta},
        path,
    )


def load_caption_cache(
    path: str | Path,
) -> tuple[list[str], np.ndarray, dict]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    X = blob["X"]
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    return blob["datazones"], np.asarray(X, dtype=np.float32), blob.get("meta", {})
