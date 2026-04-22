"""
Torch dataset construction for Route A.

Given:
  samples  : list[dict] from build_dataset.py (text segments + poi_counts + targets)
  segments : {"sat": (N,d), "ntl": (N,d), "sv": (N,d), "poi_text": (N,d)}  for all
             rows in `cache_datazones`  (output of encoder.encode_segments)
  dz_index : {datazone: row_index_in_segments}
  poi_fitter : fitted PoiFitter

we produce a TensorDataset of (e_sat, e_ntl, e_sv, e_poi_text, v_poi, y_logit)
plus a parallel y_raw tensor for evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from decision.data.poi_features import PoiFitter
from decision.data.targets import DOMAINS


def build_tensors(
    samples: list[dict],
    segments: dict[str, torch.Tensor],
    dz_index: dict[str, int],
    poi_fitter: PoiFitter,
) -> tuple[TensorDataset, torch.Tensor, list[str]]:
    """Return (tensor_dataset, y_raw_tensor, datazone_order)."""
    idx = torch.tensor([dz_index[s["datazone"]] for s in samples], dtype=torch.long)
    e_sat = segments["sat"].index_select(0, idx)
    e_ntl = segments["ntl"].index_select(0, idx)
    e_sv = segments["sv"].index_select(0, idx)
    e_poi_text = segments["poi_text"].index_select(0, idx)

    v_poi_np = np.stack([poi_fitter.transform(s) for s in samples])  # (N, V)
    v_poi = torch.from_numpy(v_poi_np).float()

    y_logit = torch.tensor(
        [[s["targets"][d] for d in DOMAINS] for s in samples], dtype=torch.float32
    )
    y_raw = torch.tensor(
        [[s["targets_raw"][d] for d in DOMAINS] for s in samples], dtype=torch.float32
    )

    datazones = [s["datazone"] for s in samples]
    ds = TensorDataset(e_sat, e_ntl, e_sv, e_poi_text, v_poi, y_logit)
    return ds, y_raw, datazones
