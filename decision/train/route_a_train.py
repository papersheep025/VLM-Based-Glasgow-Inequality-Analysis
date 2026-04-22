"""
Train Route A for a single (train_samples, val_samples) fold.

Produces:
  best_model_state  : state_dict of the best epoch (by val mean Spearman)
  val_pred_logit    : (N_val, 7)  predictions in logit space
  val_pred_raw      : (N_val, 7)  predictions in SIMD 1-10 space (denormalised)
  val_datazones     : list[str]   matching row order
  history           : dict of epoch metrics

This module exposes a single ``train_fold`` function; CV orchestration lives in
``cv_runner.py``.  Defaults follow the plan spec (Huber + Pearson + SoftSpearman
per head, Kendall uncertainty weighting, grad-clip 1.0, early stop on mean
val Spearman).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from decision.data.poi_features import PoiFitter
from decision.data.targets import DOMAINS, denormalise_array
from decision.models.route_a.heads import RouteAModel
from decision.train.dataset import build_tensors
from decision.train.losses import HeadLoss, UncertaintyWeighting
from src.glasgow_vlm.metrics import spearmanr


@dataclass
class RouteATrainConfig:
    # Model
    d_txt: int = 1024
    d_poi: int = 32
    d_trunk: int = 256
    head_hidden: int = 64
    dropout: float = 0.2
    # Optim
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 64
    grad_clip: float = 1.0
    # Loss
    lam_pearson: float = 0.5
    lam_spearman: float = 0.5
    huber_beta: float = 1.0
    soft_beta: float = 5.0
    # Early stop
    patience: int = 10
    # Misc
    device: str | None = None
    seed: int = 42
    domains: list[str] = field(default_factory=lambda: list(DOMAINS))


def _auto_device(pref: str | None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mean_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    vals = [spearmanr(target[:, k].tolist(), pred[:, k].tolist()) for k in range(pred.shape[1])]
    return float(np.mean(vals))


@torch.no_grad()
def _predict(model: RouteAModel, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    outs = []
    for e_sat, e_ntl, e_sv, e_poi, v_poi, _ in loader:
        pred = model(
            e_sat.to(device), e_ntl.to(device), e_sv.to(device),
            e_poi.to(device), v_poi.to(device),
        )
        outs.append(pred.cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0, len(DOMAINS)))


def train_fold(
    train_samples: list[dict],
    val_samples: list[dict],
    segments: dict[str, torch.Tensor],
    dz_index: dict[str, int],
    cfg: RouteATrainConfig | None = None,
) -> dict:
    cfg = cfg or RouteATrainConfig()
    _set_seed(cfg.seed)
    device = _auto_device(cfg.device)

    # POI scaler fit on train only (no leakage).
    poi_fitter = PoiFitter().fit(train_samples)

    train_ds, _, _ = build_tensors(train_samples, segments, dz_index, poi_fitter)
    val_ds, y_val_raw, val_dz = build_tensors(val_samples, segments, dz_index, poi_fitter)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = RouteAModel(
        d_txt=cfg.d_txt,
        d_poi_in=poi_fitter.dim,
        d_poi=cfg.d_poi,
        d_trunk=cfg.d_trunk,
        head_hidden=cfg.head_hidden,
        dropout=cfg.dropout,
        domains=cfg.domains,
    ).to(device)
    mtl = UncertaintyWeighting(n_tasks=len(cfg.domains)).to(device)
    head_loss = HeadLoss(
        huber_beta=cfg.huber_beta,
        lam_pearson=cfg.lam_pearson,
        lam_spearman=cfg.lam_spearman,
        soft_beta=cfg.soft_beta,
    ).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(mtl.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_score = -float("inf")
    best_state: dict | None = None
    best_epoch = -1
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in train_loader:
            e_sat, e_ntl, e_sv, e_poi, v_poi, y = [t.to(device) for t in batch]
            pred = model(e_sat, e_ntl, e_sv, e_poi, v_poi)     # (B, T)
            per_task = torch.stack([
                head_loss(pred[:, k], y[:, k]) for k in range(pred.shape[1])
            ])
            loss = mtl(per_task)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(mtl.parameters()), cfg.grad_clip,
            )
            opt.step()
            running += float(loss.item())
            n_batches += 1

        # ---- Validation ----
        val_pred_logit = _predict(model, val_loader, device)
        val_pred_raw = denormalise_array(val_pred_logit)
        val_score = _mean_spearman(val_pred_raw, y_val_raw.numpy())
        history.append({
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "val_mean_spearman": val_score,
        })

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "mtl": copy.deepcopy(mtl.state_dict()),
                "poi_fitter": {
                    "vocab": poi_fitter.vocab,
                    "mean": poi_fitter._mean.tolist(),
                    "std": poi_fitter._std.tolist(),
                    "top_k": poi_fitter.top_k,
                },
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    # ---- Restore best and produce final val predictions ----
    assert best_state is not None
    model.load_state_dict(best_state["model"])
    val_pred_logit = _predict(model, val_loader, device)
    val_pred_raw = denormalise_array(val_pred_logit)

    return {
        "best_state": best_state,
        "val_datazones": val_dz,
        "val_pred_logit": val_pred_logit,
        "val_pred_raw": val_pred_raw,
        "val_target_raw": y_val_raw.numpy(),
        "val_mean_spearman": best_score,
        "best_epoch": best_epoch,
        "history": history,
    }
