import math
import pandas as pd
import geopandas as gpd
import os
import requests
import random
import json
import csv
import time
import argparse
from datetime import datetime, timezone
from shapely.geometry import Point


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape Google Street View images by bounding box from datazone_all_patch_metadata.csv"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GOOGLE_STREETVIEW_API_KEY"),
        help="Google Street View API key (or set GOOGLE_STREETVIEW_API_KEY env var)",
    )
    parser.add_argument(
        "--input-csv",
        default="dataset/datazone_patches/datazone_all_patch_metadata.csv",
        help="Path to datazone_all_patch_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/streetview_dataset_gsv",
        help="Output directory",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Max images per bounding box (default: 20)",
    )
    parser.add_argument(
        "--min-spacing",
        type=float,
        default=20.0,
        help="Minimum spacing between accepted pano locations in metres (default: 20)",
    )
    parser.add_argument(
        "--candidate-points",
        type=int,
        default=200,
        help="Candidate random points per bbox to query metadata for",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=336,
        help="Output image size in pixels (square, default: 336)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Limit number of patches to process (for testing)",
    )
    parser.add_argument(
        "--patch-id",
        default=None,
        help="Process only this specific patch_id",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query metadata and print valid pano coordinates; do NOT download images",
    )
    parser.add_argument(
        "--preview-csv",
        default=None,
        metavar="PATH",
        help=(
            "Query metadata for all patches and save valid pano coordinates to PATH "
            "(free, no images downloaded). Implies --dry-run."
        ),
    )
    parser.add_argument(
        "--check-api-key",
        action="store_true",
        help="Test the API key with a single metadata request and exit",
    )
    parser.add_argument(
        "--svi-metadata",
        default="outputs/svi_metadata.csv",
        metavar="PATH",
        help="Path to write svi_metadata.csv (default: outputs/svi_metadata.csv)",
    )
    return parser.parse_args()


REQUEST_DELAY = 0.05
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
CHECKPOINT_EVERY_PATCHES = 5


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def min_dist_to_accepted(lat, lon, accepted):
    if not accepted:
        return float("inf")
    return min(haversine_m(lat, lon, a["lat"], a["lon"]) for a in accepted)


def generate_random_points_in_bbox(bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat, n, seed=None):
    rng = random.Random(seed)
    return [
        (rng.uniform(bbox_min_lat, bbox_max_lat), rng.uniform(bbox_min_lon, bbox_max_lon))
        for _ in range(n)
    ]


def atomic_write_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load checkpoint {path}: {e}")
        return None


def load_seen_pano_ids(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pid = row.get("google_pano_id")
                if pid:
                    seen.add(pid)
    except Exception as e:
        print(f"Warning: failed to read existing metadata CSV {path}: {e}")
    return seen


def request_with_retry(url, max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    print(f"  HTTP {resp.status_code}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"  Request error: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    return None


def gsv_metadata(lat, lon, api_key):
    url = (
        f"https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?location={lat},{lon}&key={api_key}"
    )
    time.sleep(REQUEST_DELAY)
    resp = request_with_retry(url)
    if resp is None or resp.status_code != 200:
        return {"status": "ERROR"}
    data = resp.json()
    return {
        "status": data.get("status", "UNKNOWN"),
        "pano_id": data.get("pano_id"),
        "lat": data.get("location", {}).get("lat"),
        "lon": data.get("location", {}).get("lng"),
    }


def gsv_image(lat, lon, api_key, size_px):
    url = (
        f"https://maps.googleapis.com/maps/api/streetview"
        f"?size={size_px}x{size_px}&location={lat},{lon}&key={api_key}"
    )
    time.sleep(REQUEST_DELAY)
    resp = request_with_retry(url)
    if resp is None or resp.status_code != 200:
        return None
    return resp.content


def collect_valid_panos(patch_id, bbox, args, seen_pano_ids, global_accepted=None, polygon=None):
    """
    Query GSV metadata for candidate points inside bbox.
    Returns list of dicts: {query_lat, query_lon, pano_id, lat, lon}
    Respects min_spacing and max_images limits.
    """
    bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = bbox
    seed = hash(patch_id) & 0xFFFFFFFF
    candidates = generate_random_points_in_bbox(
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat,
        args.candidate_points, seed=seed,
    )

    accepted = []
    for q_lat, q_lon in candidates:
        if len(accepted) >= args.max_images:
            break
        meta = gsv_metadata(q_lat, q_lon, args.api_key)
        if meta["status"] != "OK":
            continue
        pano_id = meta.get("pano_id")
        p_lat, p_lon = meta.get("lat"), meta.get("lon")
        if p_lat is None or p_lon is None:
            continue
        if pano_id and pano_id in seen_pano_ids:
            continue
        if not (bbox_min_lon <= p_lon <= bbox_max_lon and bbox_min_lat <= p_lat <= bbox_max_lat):
            continue
        if polygon is not None and not polygon.contains(Point(p_lon, p_lat)):
            continue
        if min_dist_to_accepted(p_lat, p_lon, accepted) < args.min_spacing:
            continue
        accepted.append({
            "query_lat": q_lat,
            "query_lon": q_lon,
            "pano_id": pano_id or "",
            "lat": p_lat,
            "lon": p_lon,
        })
    return accepted


def check_api_key(api_key):
    # Glasgow city centre as test location
    test_lat, test_lon = 55.8617, -4.2583
    print(f"Testing API key with metadata request at ({test_lat}, {test_lon}) ...")
    meta = gsv_metadata(test_lat, test_lon, api_key)
    status = meta.get("status")
    if status == "OK":
        print(f"  OK — pano_id={meta.get('pano_id')}, returned location=({meta.get('lat')}, {meta.get('lon')})")
        print("API key is valid.")
    elif status == "ZERO_RESULTS":
        print("  ZERO_RESULTS — API key works but no Street View at this location (unexpected for city centre).")
    elif status == "REQUEST_DENIED":
        print("  REQUEST_DENIED — API key is invalid or Street View Static API not enabled.")
    else:
        print(f"  Unexpected status: {status}")
    return status == "OK"


CSV_HEADER = [
    "image", "patch_id", "datazone", "lat", "lon",
    "query_lat", "query_lon", "google_pano_id",
]

SVI_METADATA_HEADER = [
    "image", "patch_id", "datazone", "lat", "lon",
    "bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat",
]


def main():
    args = parse_args()

    if not args.api_key:
        print("Error: API key required. Use --api-key or set GOOGLE_STREETVIEW_API_KEY")
        return

    if args.check_api_key:
        check_api_key(args.api_key)
        return

    patches_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(patches_df)} patches from {args.input_csv}")

    shp_path = "dataset/glasgow_datazone/glasgow_datazone.shp"
    gdf = gpd.read_file(shp_path).set_index("DataZone")
    print(f"Loaded {len(gdf)} datazone polygons from {shp_path}")

    if args.patch_id:
        patches_df = patches_df[patches_df["patch_id"] == args.patch_id]
        if patches_df.empty:
            print(f"Error: patch_id '{args.patch_id}' not found in CSV")
            return
        print(f"Filtered to patch_id={args.patch_id}")

    if args.max_patches:
        patches_df = patches_df.head(args.max_patches)

    if args.dry_run or args.preview_csv:
        preview_csv = args.preview_csv
        preview_header = [
            "patch_id", "datazone", "pano_index",
            "pano_lat", "pano_lon", "query_lat", "query_lon", "google_pano_id",
        ]

        # --- resume: derive completed patches from existing CSV ---
        completed_patches = set()
        if preview_csv and os.path.exists(preview_csv) and os.path.getsize(preview_csv) > 0:
            with open(preview_csv, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    completed_patches.add(row["patch_id"])

        patches_to_run = patches_df[~patches_df["patch_id"].isin(completed_patches)]
        csv_needs_header = (
            not preview_csv
            or not os.path.exists(preview_csv)
            or os.path.getsize(preview_csv) == 0
        )

        print(f"\n{'='*60}")
        print(f"PREVIEW — querying metadata only (free, no quota consumed)")
        print(f"max_images={args.max_images}, min_spacing={args.min_spacing}m, candidates={args.candidate_points}")
        if preview_csv:
            if completed_patches:
                print(f"Resuming: {len(completed_patches)} patches already done, "
                      f"{len(patches_to_run)} remaining")
            print(f"Output CSV: {preview_csv}")
        print(f"{'='*60}\n")

        total_valid = 0
        out_file = None
        writer = None
        if preview_csv:
            os.makedirs(os.path.dirname(preview_csv) or ".", exist_ok=True)
            out_file = open(preview_csv, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(out_file, fieldnames=preview_header)
            if csv_needs_header:
                writer.writeheader()
                out_file.flush()

        try:
            for i, (_, row) in enumerate(patches_to_run.iterrows()):
                pid = row["patch_id"]
                dz = row["datazone"]
                bbox = (row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"])
                print(f"[{i+1}/{len(patches_to_run)}] {pid}  "
                      f"bbox=[{bbox[1]:.5f},{bbox[0]:.5f} → {bbox[3]:.5f},{bbox[2]:.5f}]")

                polygon = gdf.geometry.get(dz)
                valid = collect_valid_panos(pid, bbox, args, seen_pano_ids=set(), polygon=polygon)

                if valid:
                    for j, p in enumerate(valid):
                        print(f"  [{j+1:2d}] pano_id={p['pano_id'] or 'N/A':>26}  "
                              f"pano=({p['lat']:.6f}, {p['lon']:.6f})  "
                              f"query=({p['query_lat']:.6f}, {p['query_lon']:.6f})")
                        if writer:
                            writer.writerow({
                                "patch_id": pid,
                                "datazone": dz,
                                "pano_index": j,
                                "pano_lat": p["lat"],
                                "pano_lon": p["lon"],
                                "query_lat": p["query_lat"],
                                "query_lon": p["query_lon"],
                                "google_pano_id": p["pano_id"],
                            })
                            out_file.flush()
                else:
                    print("  (no Street View coverage found)")

                print(f"  → {len(valid)} valid panos\n")
                total_valid += len(valid)
        finally:
            if out_file:
                out_file.close()

        print(f"Preview complete: {total_valid} valid panos across {len(patches_to_run)} patches.")
        if preview_csv:
            print(f"Saved → {preview_csv}")
        return

    # --- Normal run ---
    output_dir = args.output_dir
    image_dir = os.path.join(output_dir, "images")
    metadata_csv_path = os.path.join(output_dir, "metadata.csv")
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    svi_metadata_path = args.svi_metadata
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(svi_metadata_path) or ".", exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    completed_patches = set(checkpoint.get("completed_patches", [])) if checkpoint else set()
    print(f"Resuming: {len(completed_patches)} patches already completed")

    seen_pano_ids = load_seen_pano_ids(metadata_csv_path)
    print(f"Loaded {len(seen_pano_ids)} existing pano IDs")

    csv_needs_header = not os.path.exists(metadata_csv_path) or os.path.getsize(metadata_csv_path) == 0
    svi_needs_header = not os.path.exists(svi_metadata_path) or os.path.getsize(svi_metadata_path) == 0

    patches_to_process = patches_df[~patches_df["patch_id"].isin(completed_patches)]
    print(f"\nProcessing {len(patches_to_process)} patches, max_images={args.max_images}, "
          f"min_spacing={args.min_spacing}m\n")

    total_images = 0
    patches_done_this_run = 0

    try:
        with (
            open(metadata_csv_path, "a", newline="", encoding="utf-8") as csvfile,
            open(svi_metadata_path, "a", newline="", encoding="utf-8") as svi_file,
        ):
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADER)
            svi_writer = csv.DictWriter(svi_file, fieldnames=SVI_METADATA_HEADER)
            if csv_needs_header:
                writer.writeheader()
                csvfile.flush()
            if svi_needs_header:
                svi_writer.writeheader()
                svi_file.flush()

            for _, row in patches_to_process.iterrows():
                pid = row["patch_id"]
                dz = row["datazone"]
                bbox = (row["bbox_min_lon"], row["bbox_min_lat"], row["bbox_max_lon"], row["bbox_max_lat"])

                print(f"[{patches_done_this_run+1}/{len(patches_to_process)}] {pid} ...", end=" ", flush=True)

                polygon = gdf.geometry.get(dz)
                valid_panos = collect_valid_panos(pid, bbox, args, seen_pano_ids, polygon=polygon)

                img_count = 0
                for i, pano in enumerate(valid_panos):
                    img_data = gsv_image(pano["lat"], pano["lon"], args.api_key, args.image_size)
                    if img_data is None:
                        continue
                    image_name = f"{pid}_{i}.jpg"
                    with open(os.path.join(image_dir, image_name), "wb") as f:
                        f.write(img_data)
                    writer.writerow({
                        "image": image_name,
                        "patch_id": pid,
                        "datazone": dz,
                        "lat": pano["lat"],
                        "lon": pano["lon"],
                        "query_lat": pano["query_lat"],
                        "query_lon": pano["query_lon"],
                        "google_pano_id": pano["pano_id"],
                    })
                    csvfile.flush()
                    svi_writer.writerow({
                        "image": image_name,
                        "patch_id": pid,
                        "datazone": dz,
                        "lat": pano["lat"],
                        "lon": pano["lon"],
                        "bbox_min_lon": bbox[0],
                        "bbox_min_lat": bbox[1],
                        "bbox_max_lon": bbox[2],
                        "bbox_max_lat": bbox[3],
                    })
                    svi_file.flush()
                    if pano["pano_id"]:
                        seen_pano_ids.add(pano["pano_id"])
                    img_count += 1

                print(f"{img_count} images")
                total_images += img_count
                completed_patches.add(pid)
                patches_done_this_run += 1

                if patches_done_this_run % CHECKPOINT_EVERY_PATCHES == 0:
                    atomic_write_json(checkpoint_path, {
                        "updated_at": _utc_now_iso(),
                        "completed_patches": list(completed_patches),
                    })

    finally:
        atomic_write_json(checkpoint_path, {
            "updated_at": _utc_now_iso(),
            "completed_patches": list(completed_patches),
        })

    print(f"\nDone. {patches_done_this_run} patches processed, {total_images} images downloaded.")
    print(f"Output: {output_dir}")
    print(f"SVI metadata: {svi_metadata_path}")


if __name__ == "__main__":
    main()
