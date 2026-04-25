"""
Download Google Street View Static API images from gsv_preview_all.csv.
One image per row (pano). Resume-safe: skips already-downloaded files.

Usage:
    python data_preparation/scraper/download_gsv_images.py \
        --api-key YOUR_KEY \
        [--input outputs/gsv_preview_all.csv] \
        [--output-dir dataset/gsv_dataset] \
        [--size 640x640] [--fov 90] [--workers 8]
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

GSV_URL = "https://maps.googleapis.com/maps/api/streetview"


def download_one(
    row: dict, output_dir: Path, size: str, fov: int, api_key: str
) -> tuple[str, bool, str]:
    patch_id = row["patch_id"]
    pano_index = int(row["pano_index"])
    pano_id = row["google_pano_id"]

    save_path = output_dir / patch_id / f"{pano_index:04d}.jpg"
    if save_path.exists():
        return str(save_path), True, "skipped"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GSV_URL}?pano={pano_id}&size={size}&fov={fov}&key={api_key}"

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and content_type.startswith("image"):
                save_path.write_bytes(resp.content)
                return str(save_path), True, "downloaded"
            else:
                return str(save_path), False, f"http_{resp.status_code}"
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2**attempt)
            else:
                return str(save_path), False, str(e)

    return str(save_path), False, "max_retries"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("GSV_API_KEY"))
    parser.add_argument("--input", default="outputs/gsv_preview_all.csv")
    parser.add_argument("--output-dir", default="dataset/gsv_dataset")
    parser.add_argument("--size", default="640x640")
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Provide --api-key or set GSV_API_KEY env var")

    df = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict("records")
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row, output_dir, args.size, args.fov, args.api_key): row
            for row in rows
        }
        with tqdm(total=len(futures), unit="img") as pbar:
            for future in as_completed(futures):
                _, ok, status = future.result()
                if status == "skipped":
                    stats["skipped"] += 1
                elif ok:
                    stats["downloaded"] += 1
                else:
                    stats["failed"] += 1
                pbar.set_postfix(stats)
                pbar.update(1)

    print(
        f"\nDone — downloaded: {stats['downloaded']}, "
        f"skipped: {stats['skipped']}, failed: {stats['failed']}"
    )


if __name__ == "__main__":
    main()
