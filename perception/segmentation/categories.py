"""ADE20K class mapping used by both SegFormer and MIT ResNet18+PPM branches.

ADE20K has 150 classes. The official release (and the raw segmentation masks
published by MIT) numbers them 1..150. PyTorch models trained on it (both HF
SegFormer and mit-semseg) emit a 150-channel logit tensor whose channel index
``i`` corresponds to ADE20K-1-indexed class ``i + 1``.

The keys below are written in the **0-indexed channel convention** (matching
both HF id2label and mit-semseg's softmax output). For example, channel ``1``
is ``building`` (which is ADE20K-1-indexed class 2).
"""
from __future__ import annotations

# Channel index (0-indexed, as emitted by HF / mit-semseg) → name
ADE20K_KEEP: dict[int, str] = {
    1:  "building",      # ADE20K class 2
    2:  "sky",           # ADE20K class 3
    4:  "tree",          # ADE20K class 5
    6:  "road",          # ADE20K class 7
    9:  "grass",         # ADE20K class 10
    11: "sidewalk",      # ADE20K class 12
    12: "person",        # ADE20K class 13
    13: "earth",         # ADE20K class 14
    17: "plant",         # ADE20K class 18
    20: "car",           # ADE20K class 21
    32: "fence",         # ADE20K class 33
    43: "signboard",     # ADE20K class 44
    87: "streetlight",   # ADE20K class 88
    93: "pole",          # ADE20K class 94
}

# Column order for parquet output (14 kept classes + "other")
SVF_COLUMNS: list[str] = [
    "building", "sky", "tree", "road", "grass", "sidewalk", "person",
    "earth", "plant", "car", "fence", "signboard", "streetlight", "pole",
    "other",
]

assert len(SVF_COLUMNS) == len(ADE20K_KEEP) + 1
