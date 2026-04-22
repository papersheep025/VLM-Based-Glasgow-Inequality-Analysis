"""
Frozen text encoder for Route A.

Default backend: ``BAAI/bge-m3`` (1024-d).  We do not install sentence-transformers;
the encoder is a thin wrapper around HuggingFace AutoModel + AutoTokenizer with
CLS-pooled + L2-normalised outputs (matching BGE's recommended usage).

Typical use::

    enc = FrozenTextEncoder("BAAI/bge-m3", device="cuda")
    emb = enc.encode(["high density residential...", "dim scattered lights..."])
    # -> torch.FloatTensor shape (2, 1024) on CPU by default
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "BAAI/bge-m3"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FrozenTextEncoder:
    """Encode strings with a frozen HF text encoder (CLS-pooled + L2-normed)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 256,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or _auto_device()
        self.max_length = max_length
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Probe embedding dim with a zero-length call (rare — use hidden size instead).
        self._dim = int(self.model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(
        self,
        texts: list[str] | Iterable[str],
        batch_size: int = 32,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        texts = list(texts)
        if not texts:
            return torch.empty(0, self._dim)

        outs: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            tok = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**tok).last_hidden_state  # (B, T, H)
            cls = hidden[:, 0]  # CLS pooling (BGE default)
            emb = F.normalize(cls, p=2, dim=-1).float()
            outs.append(emb.cpu() if return_cpu else emb)
        return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Segment-level helper: encode the 4 text segments of a dataset_v0 row.
# ---------------------------------------------------------------------------

_SEGMENTS = ("sat", "ntl", "sv", "poi_text")


def encode_segments(
    encoder: FrozenTextEncoder,
    samples: list[dict],
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Encode the 4 text segments across all samples.

    Returns dict {segment: tensor of shape (N, d)} in the order of `samples`.
    """
    out: dict[str, torch.Tensor] = {}
    for seg in _SEGMENTS:
        texts = [s[seg] for s in samples]
        out[seg] = encoder.encode(texts, batch_size=batch_size)
    return out


def save_segment_cache(tensors: dict[str, torch.Tensor], datazones: list[str], path: str | Path) -> None:
    """Persist segment embeddings + datazone order so training can skip re-encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"datazones": datazones, "segments": tensors}, path)


def load_segment_cache(path: str | Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return blob["datazones"], blob["segments"]
