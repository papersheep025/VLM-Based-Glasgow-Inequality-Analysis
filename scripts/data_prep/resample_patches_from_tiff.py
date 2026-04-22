"""
Re-sample satellite and NTL patches from source TIFFs.
Each patch is a fixed 333 m x 333 m window centred on the datazone centroid
(centroid_lat, centroid_lon from satellite_metadata.csv).
Both modalities are sampled in EPSG:32630 so they cover the same physical extent.
NTL TIFF is reprojected on-the-fly via WarpedVRT.
Overwrites existing patch images.

SAT output: 384x384 PNG  (native EPSG:32630)
NTL output: 256x256 PNG  (reprojected to EPSG:32630 via WarpedVRT)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
from rasterio.vrt import WarpedVRT
from PIL import Image
from pyproj import Transformer
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
METADATA_CSV = ROOT / "dataset/satellite_dataset/satellite_metadata.csv"
SAT_TIFF = ROOT / "dataset/TIFF/glasgow/glasgow.tif"
NTL_TIFF = ROOT / "dataset/TIFF/glasgow_ntl/glasgow_ntl.tif"
SAT_SIZE = (384, 384)
NTL_SIZE = (256, 256)
HALF_SIDE = 333 / 2
TARGET_CRS = rasterio.crs.CRS.from_epsg(32630)


def centroid_bbox(lat, lon):
    t = Transformer.from_crs("EPSG:4326", 32630, always_xy=True)
    cx, cy = t.transform(lon, lat)
    return cx - HALF_SIDE, cy - HALF_SIDE, cx + HALF_SIDE, cy + HALF_SIDE


def crop_resize(src, left, bottom, right, top, out_size):
    win = from_bounds(left, bottom, right, top, src.transform)
    r0 = max(0, int(win.row_off))
    c0 = max(0, int(win.col_off))
    r1 = min(src.height, int(win.row_off + win.height))
    c1 = min(src.width, int(win.col_off + win.width))
    if r1 <= r0 or c1 <= c0:
        return None
    data = src.read(window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0))
    arr = np.transpose(data, (1, 2, 0))[:, :, :3]
    img = Image.fromarray(arr.astype(np.uint8))
    return img.resize(out_size, Image.LANCZOS)


def main():
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} datazones from {METADATA_CSV.name}")
    print("Both modalities sampled in EPSG:32630  window: 333mx333m")

    failed = []
    with rasterio.open(SAT_TIFF) as sat_src, \
         rasterio.open(NTL_TIFF) as ntl_raw, \
         WarpedVRT(ntl_raw, crs=TARGET_CRS) as ntl_src:

        for i, row in df.iterrows():
            dz = row.datazone
            lat, lon = row.centroid_lat, row.centroid_lon
            bbox = centroid_bbox(lat, lon)

            sat_path = Path(row.satellite_patch)
            ntl_path = Path(row.ntl_patch)
            sat_path.parent.mkdir(parents=True, exist_ok=True)
            ntl_path.parent.mkdir(parents=True, exist_ok=True)

            sat_img = crop_resize(sat_src, *bbox, SAT_SIZE)
            if sat_img is None:
                print(f"  SKIP SAT {dz}")
                failed.append(dz)
                continue
            sat_img.save(sat_path)

            ntl_img = crop_resize(ntl_src, *bbox, NTL_SIZE)
            if ntl_img is None:
                print(f"  SKIP NTL {dz}")
                failed.append(dz)
                continue
            ntl_img.save(ntl_path)

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(df)}")

    print(f"\nDone. Success: {len(df) - len(failed)}  Failed: {len(failed)}")
    if failed:
        print("Failed datazones:", failed)


if __name__ == "__main__":
    main()
