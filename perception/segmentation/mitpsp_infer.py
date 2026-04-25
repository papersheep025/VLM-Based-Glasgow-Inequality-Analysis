"""Run MIT mit-semseg ResNet18-dilated + PPM_deepsup over street-view images.

Inlines a slim re-implementation of the encoder / decoder that is
state_dict-compatible with the official checkpoints at
http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet18dilated-ppm_deepsup/

Output schema matches ``segformer_infer.py`` so that downstream ``aggregate.py``
treats both models identically.

CLI:
    python -m perception.segmentation.mitpsp_infer \
        --metadata dataset/streetview_metadata.csv \
        --raw-dir dataset/streetview_dataset_raw \
        --output outputs/perception/svf/image_level_mitpsp.parquet \
        --input-size 512 --batch-size 16 --device cuda:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from perception.segmentation.categories import ADE20K_KEEP, SVF_COLUMNS
from perception.segmentation.download_mit_weights import ensure_weights
from perception.segmentation.segformer_infer import (
    _append_parquet,
    _compute_svf_from_logits,
    _image_id,
    _load_done_ids,
    _resolve_raw_path,
)


# ---------------------------------------------------------------------------
# Model definitions (state_dict-compatible with mit-semseg official ckpts).
# Reference: github.com/CSAILVision/semantic-segmentation-pytorch
# ---------------------------------------------------------------------------

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    """ResNet with mit-semseg's 3-conv stem (conv1→conv2→conv3)."""

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 128
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResnetDilated(nn.Module):
    """Wraps a ResNet, converting the last 2 stages to dilated conv (stride=8)."""

    def __init__(self, orig_resnet: ResNet, dilate_scale: int = 8):
        super().__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(lambda m: self._nostride_dilate(m, 2))
            orig_resnet.layer4.apply(lambda m: self._nostride_dilate(m, 4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(lambda m: self._nostride_dilate(m, 2))

        self.conv1, self.bn1, self.relu1 = orig_resnet.conv1, orig_resnet.bn1, orig_resnet.relu1
        self.conv2, self.bn2, self.relu2 = orig_resnet.conv2, orig_resnet.bn2, orig_resnet.relu2
        self.conv3, self.bn3, self.relu3 = orig_resnet.conv3, orig_resnet.bn3, orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    @staticmethod
    def _nostride_dilate(m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            elif m.kernel_size == (3, 3):
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps: bool = False):
        conv_out = []
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out.append(x)
        x = self.layer2(x); conv_out.append(x)
        x = self.layer3(x); conv_out.append(x)
        x = self.layer4(x); conv_out.append(x)
        if return_feature_maps:
            return conv_out
        return [x]


class PPMDeepsup(nn.Module):
    """Pyramid Pooling Module with deep supervision (mit-semseg variant)."""

    def __init__(self, num_class: int = 150, fc_dim: int = 512,
                 pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(fc_dim, 512, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ) for s in pool_scales
        ])
        self.cbr_deepsup = nn.Sequential(
            nn.Conv2d(fc_dim // 2, fc_dim // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(fc_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, 1),
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, seg_size=None):
        conv5 = conv_out[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool in self.ppm:
            ppm_out.append(F.interpolate(
                pool(conv5), (input_size[2], input_size[3]),
                mode="bilinear", align_corners=False,
            ))
        ppm_out = torch.cat(ppm_out, 1)
        x = self.conv_last(ppm_out)
        if seg_size is not None:
            x = F.interpolate(x, size=seg_size, mode="bilinear", align_corners=False)
        return F.log_softmax(x, dim=1)


def build_model(encoder_ckpt: Path, decoder_ckpt: Path,
                device: str) -> tuple[ResnetDilated, PPMDeepsup]:
    enc_backbone = ResNet(BasicBlock, [2, 2, 2, 2])  # ResNet18
    encoder = ResnetDilated(enc_backbone, dilate_scale=8)
    decoder = PPMDeepsup(num_class=150, fc_dim=512, pool_scales=(1, 2, 3, 6))

    enc_sd = torch.load(encoder_ckpt, map_location="cpu")
    dec_sd = torch.load(decoder_ckpt, map_location="cpu")
    missing_e, unexpected_e = encoder.load_state_dict(enc_sd, strict=False)
    missing_d, unexpected_d = decoder.load_state_dict(dec_sd, strict=False)
    if missing_e or unexpected_e:
        print(f"[mitpsp] encoder missing={len(missing_e)} unexpected={len(unexpected_e)}")
    if missing_d or unexpected_d:
        print(f"[mitpsp] decoder missing={len(missing_d)} unexpected={len(unexpected_d)}")
    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    return encoder, decoder


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

# ImageNet mean/std, same as mit-semseg pre-processing.
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


def _prep_batch(imgs: list[Image.Image], input_size: int) -> torch.Tensor:
    out = []
    for img in imgs:
        img_r = img.resize((input_size, input_size), Image.BILINEAR)
        t = transforms.functional.to_tensor(img_r)
        out.append(_NORMALIZE(t))
    return torch.stack(out)


def run(
    metadata_csv: Path,
    raw_dir: Path,
    output_path: Path,
    weights_dir: Path | None,
    input_size: int,
    batch_size: int,
    device: str,
    flush_every: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(metadata_csv)
    meta["image_id"] = [
        _image_id(p, i) for p, i in zip(meta["patch_id"], meta["pano_index"])
    ]
    done = _load_done_ids(output_path)
    todo = meta[~meta["image_id"].isin(done)].reset_index(drop=True)
    print(f"[mitpsp] total={len(meta)} done={len(done)} todo={len(todo)}")
    if len(todo) == 0:
        print("[mitpsp] nothing to do.")
        return

    weight_paths = ensure_weights(weights_dir)
    encoder, decoder = build_model(
        weight_paths["encoder"], weight_paths["decoder"], device
    )

    ade_to_col = {
        ade_id: SVF_COLUMNS.index(name) for ade_id, name in ADE20K_KEEP.items()
    }
    n_cols = len(SVF_COLUMNS)

    buffer_rows: list[dict] = []
    n_missing = 0

    with torch.no_grad():
        for start in range(0, len(todo), batch_size):
            batch = todo.iloc[start:start + batch_size]
            imgs, keep_rows = [], []
            for _, row in batch.iterrows():
                img_path = _resolve_raw_path(row["image_path"], raw_dir)
                if not img_path.exists():
                    n_missing += 1
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    n_missing += 1
                    continue
                imgs.append(img)
                keep_rows.append(row)
            if not imgs:
                continue

            x = _prep_batch(imgs, input_size).to(device)
            feats = encoder(x, return_feature_maps=False)
            # seg_size=None → no upsample; we argmax on the feature-map grid.
            # With dilate_scale=8, grid is (input/8, input/8); argmax ratio
            # equals the full-res ratio up to bilinear resampling error, so
            # computing SVF directly here is accurate enough.
            log_probs = decoder(feats, seg_size=None)       # (B, 150, H/8, W/8)
            # log_softmax → argmax index, shift to 1-indexed via
            # _compute_svf_from_logits (pred+1 inside).
            pred_ids = log_probs.argmax(dim=1).long()
            svf = _compute_svf_from_logits(pred_ids, ade_to_col, n_cols)

            for row, svf_row in zip(keep_rows, svf):
                rec = {
                    "image_id": row["image_id"],
                    "patch_id": row["patch_id"],
                    "datazone": row["datazone"],
                    "pano_index": int(row["pano_index"]),
                }
                for col_name, v in zip(SVF_COLUMNS, svf_row):
                    rec[col_name] = float(v)
                buffer_rows.append(rec)

            if (start // batch_size) % 20 == 0:
                print(f"[mitpsp] {start + len(imgs)}/{len(todo)} "
                      f"buffered={len(buffer_rows)} missing={n_missing}")

            if len(buffer_rows) >= flush_every:
                _append_parquet(output_path, buffer_rows)
                buffer_rows = []

    if buffer_rows:
        _append_parquet(output_path, buffer_rows)
    print(f"[mitpsp] done. missing images: {n_missing}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Override MIT ckpt directory (default ~/.cache/mit-semseg/...)")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flush-every", type=int, default=1024)
    args = parser.parse_args()
    run(
        metadata_csv=args.metadata,
        raw_dir=args.raw_dir,
        output_path=args.output,
        weights_dir=args.weights_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        device=args.device,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    _main()
