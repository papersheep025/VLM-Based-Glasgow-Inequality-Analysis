"""Download MIT mit-semseg ResNet18-dilated + PPM_deepsup weights (ADE20K).

Mirrors: official http://sceneparsing.csail.mit.edu/model/pytorch/<folder>/
Falls back to user-supplied local files if provided.

Expected checkpoint files for ``ade20k-resnet18dilated-ppm_deepsup``:
    encoder_epoch_20.pth   (~ 45 MB)
    decoder_epoch_20.pth   (~ 20 MB)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

MODEL_DIR = "ade20k-resnet18dilated-ppm_deepsup"
FILES = ["encoder_epoch_20.pth", "decoder_epoch_20.pth"]
DEFAULT_URLS = [
    f"http://sceneparsing.csail.mit.edu/model/pytorch/{MODEL_DIR}/{fn}"
    for fn in FILES
]


def default_weights_dir() -> Path:
    return Path.home() / ".cache" / "mit-semseg" / MODEL_DIR


def ensure_weights(weights_dir: Path | None = None) -> dict[str, Path]:
    weights_dir = weights_dir or default_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fn, url in zip(FILES, DEFAULT_URLS):
        dst = weights_dir / fn
        if not dst.exists() or dst.stat().st_size < 1024:
            print(f"[mit-weights] downloading {fn} from {url}")
            urllib.request.urlretrieve(url, dst)
        paths["encoder" if "encoder" in fn else "decoder"] = dst
    return paths


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = ensure_weights(args.weights_dir)
    for k, v in paths.items():
        print(f"  {k}: {v}  ({v.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    _main()
