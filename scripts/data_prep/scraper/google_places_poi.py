"""
Extract Google Places API POI data for each Glasgow datazone.

For each datazone patch (from satellite_metadata.csv):
  1. Read the patch bounding box (bbox_min_lon/lat, bbox_max_lon/lat).
  2. Compute the search centre as the bbox midpoint and radius as the
     half-diagonal distance in metres.
  3. Query Google Places Nearby Search (legacy) with that centre + radius,
     paginating up to max_pages × 20 results.
  4. Filter results to only places whose lat/lon fall inside the bbox.

Output: dataset/poi_dataset/datazone_poi_google.csv
  746 rows, one per datazone. Columns:
  datazone, name, total, food_drink, retail, health, education,
  finance, transport, leisure, services, tourism, other

API cost estimate: ~746–2238 Nearby Search requests.
Google Places API pricing: https://developers.google.com/maps/billing/gmp-billing

Usage:
  export GOOGLE_MAPS_API_KEY=AIza...
  python scripts/data_prep/scraper/google_places_poi.py

  # override defaults:
  python scripts/data_prep/scraper/google_places_poi.py \
      --api-key AIza... \
      --satellite-metadata dataset/satellite_dataset/satellite_metadata.csv \
      --output dataset/poi_dataset/datazone_poi_google.csv \
      --max-pages 2 \
      --delay 0.1
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from math import cos, radians, sqrt
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import Point, MultiPolygon

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Each Google place type maps to one category bucket.
# Only the first matching type in a result's types list is used.
TYPE_TO_CATEGORY: dict[str, str] = {
    # food & drink
    "restaurant": "food_drink",
    "cafe": "food_drink",
    "bar": "food_drink",
    "bakery": "food_drink",
    "meal_takeaway": "food_drink",
    "meal_delivery": "food_drink",
    "food": "food_drink",
    "night_club": "food_drink",
    # retail
    "supermarket": "retail",
    "grocery_or_supermarket": "retail",
    "convenience_store": "retail",
    "shopping_mall": "retail",
    "clothing_store": "retail",
    "book_store": "retail",
    "home_goods_store": "retail",
    "hardware_store": "retail",
    "shoe_store": "retail",
    "electronics_store": "retail",
    "department_store": "retail",
    "bicycle_store": "retail",
    "florist": "retail",
    "jewelry_store": "retail",
    "liquor_store": "retail",
    "furniture_store": "retail",
    "pet_store": "retail",
    "store": "retail",
    # health
    "hospital": "health",
    "pharmacy": "health",
    "doctor": "health",
    "dentist": "health",
    "physiotherapist": "health",
    "health": "health",
    "veterinary_care": "health",
    # education
    "school": "education",
    "primary_school": "education",
    "secondary_school": "education",
    "university": "education",
    "library": "education",
    # finance
    "bank": "finance",
    "atm": "finance",
    "insurance_agency": "finance",
    "accounting": "finance",
    # transport
    "bus_station": "transport",
    "subway_station": "transport",
    "train_station": "transport",
    "transit_station": "transport",
    "taxi_stand": "transport",
    "light_rail_station": "transport",
    # leisure
    "park": "leisure",
    "gym": "leisure",
    "stadium": "leisure",
    "sports_complex": "leisure",
    "movie_theater": "leisure",
    "casino": "leisure",
    "amusement_park": "leisure",
    "bowling_alley": "leisure",
    "spa": "leisure",
    "swimming_pool": "leisure",
    "golf_course": "leisure",
    # services
    "post_office": "services",
    "police": "services",
    "local_government_office": "services",
    "fire_station": "services",
    "embassy": "services",
    "courthouse": "services",
    "city_hall": "services",
    "place_of_worship": "services",
    # tourism
    "tourist_attraction": "tourism",
    "museum": "tourism",
    "lodging": "tourism",
    "hotel": "tourism",
    "art_gallery": "tourism",
    "zoo": "tourism",
    "aquarium": "tourism",
}

CATEGORIES = [
    "food_drink", "retail", "health", "education",
    "finance", "transport", "leisure", "services", "tourism",
]
FIELDNAMES = ["patch_id", "datazone", "name", "total"] + CATEGORIES + ["other"]


def bbox_centre_radius(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> tuple[float, float, int]:
    """Return (lat, lon, radius_metres) derived from a bounding box.

    Radius = half the diagonal length. Clamped to [200, 50000] m.
    """
    lat = (min_lat + max_lat) / 2
    lon = (min_lon + max_lon) / 2
    dlat_m = (max_lat - min_lat) * 111_320
    dlon_m = (max_lon - min_lon) * 111_320 * cos(radians(lat))
    radius = int(min(max(sqrt(dlat_m ** 2 + dlon_m ** 2) / 2, 200), 50_000))
    return lat, lon, radius


def load_patch_metadata(path: Path) -> dict[str, dict]:
    """Load satellite_metadata.csv → {patch_id: row dict} for all 856 patches."""
    patches: dict[str, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patch_id"]
            patches[pid] = {
                "datazone": row["datazone"],
                "bbox_min_lon": float(row["bbox_min_lon"]),
                "bbox_min_lat": float(row["bbox_min_lat"]),
                "bbox_max_lon": float(row["bbox_max_lon"]),
                "bbox_max_lat": float(row["bbox_max_lat"]),
            }
    return patches


def nearby_search_page(
    lat: float, lon: float, radius: int, api_key: str, page_token: str | None
) -> dict:
    if page_token:
        params = {"pagetoken": page_token, "key": api_key}
    else:
        params = {"location": f"{lat},{lon}", "radius": radius, "key": api_key}
    resp = requests.get(NEARBY_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_places(
    lat: float, lon: float, radius: int, api_key: str, max_pages: int
) -> list[dict]:
    places: list[dict] = []
    seen: set[str] = set()
    page_token: str | None = None

    for page in range(max_pages):
        if page > 0:
            if not page_token:
                break
            time.sleep(2)  # Google requires a short delay before using next_page_token

        data = nearby_search_page(lat, lon, radius, api_key, page_token)
        status = data.get("status", "")

        if status == "OVER_QUERY_LIMIT":
            log.warning("Rate limit hit, sleeping 10 s then retrying")
            time.sleep(10)
            data = nearby_search_page(lat, lon, radius, api_key, page_token)
            status = data.get("status", "")

        if status not in ("OK", "ZERO_RESULTS"):
            log.error("API error: %s — %s", status, data.get("error_message", "(no message)"))
            break

        for place in data.get("results", []):
            pid = place.get("place_id", "")
            if pid and pid not in seen:
                seen.add(pid)
                places.append(place)

        page_token = data.get("next_page_token")

    return places


def filter_within_bbox(
    places: list[dict],
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
) -> list[dict]:
    """Return only places whose location falls inside the bounding box."""
    inside = []
    for place in places:
        loc = place.get("geometry", {}).get("location", {})
        plat, plon = loc.get("lat"), loc.get("lng")
        if plat is None or plon is None:
            continue
        if min_lat <= plat <= max_lat and min_lon <= plon <= max_lon:
            inside.append(place)
    return inside


def categorise(places: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    counts["other"] = 0
    counts["total"] = len(places)

    for place in places:
        assigned = False
        for ptype in place.get("types", []):
            cat = TYPE_TO_CATEGORY.get(ptype)
            if cat:
                counts[cat] += 1
                assigned = True
                break
        if not assigned:
            counts["other"] += 1

    return counts


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["patch_id"] for row in reader}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Google Places POI for Glasgow datazones")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GOOGLE_MAPS_API_KEY", ""),
        help="Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)",
    )
    parser.add_argument(
        "--satellite-metadata",
        default="dataset/satellite_dataset/satellite_metadata.csv",
        help="Path to satellite_metadata.csv with per-patch bounding boxes",
    )
    parser.add_argument(
        "--shapefile",
        default="dataset/glasgow_datazone/glasgow_datazone.shp",
        help="Path to Glasgow datazone shapefile (used only for datazone names)",
    )
    parser.add_argument(
        "--output",
        default="dataset/poi_dataset/datazone_poi_google.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-pages", type=int, default=3, choices=[1, 2, 3],
        help="Max pagination pages per datazone (20 results/page)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Seconds to sleep between datazones (rate limiting)",
    )
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Error: provide --api-key or set GOOGLE_MAPS_API_KEY")

    patch_meta = load_patch_metadata(Path(args.satellite_metadata))
    log.info("Loaded %d datazones from satellite metadata", len(patch_meta))

    # Load shapefile for names only (optional)
    dz_names: dict[str, str] = {}
    shp_path = Path(args.shapefile)
    if shp_path.exists():
        gdf = gpd.read_file(shp_path)
        for col in ("DataZone", "Data_Zone", "datazone"):
            if col in gdf.columns:
                gdf = gdf.rename(columns={col: "DataZone"})
                break
        name_col = "Name" if "Name" in gdf.columns else None
        for _, row in gdf.iterrows():
            dz_names[str(row["DataZone"])] = str(row[name_col]) if name_col else ""
    else:
        log.warning("Shapefile not found; datazone names will be empty")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(output_path)
    log.info("Loaded %d/%d already-completed datazones", len(done), len(patch_meta))

    write_header = not output_path.exists() or len(done) == 0
    with open(output_path, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, (patch_id, patch) in enumerate(patch_meta.items()):
            if patch_id in done:
                continue

            dz_id = patch["datazone"]
            name = dz_names.get(dz_id, "")
            min_lon = patch["bbox_min_lon"]
            min_lat = patch["bbox_min_lat"]
            max_lon = patch["bbox_max_lon"]
            max_lat = patch["bbox_max_lat"]
            lat, lon, radius = bbox_centre_radius(min_lon, min_lat, max_lon, max_lat)

            log.info("[%d/%d] %s / %s (%s) radius=%dm", i + 1, len(patch_meta), patch_id, dz_id, name, radius)

            try:
                candidates = fetch_places(lat, lon, radius, args.api_key, args.max_pages)
                places = filter_within_bbox(candidates, min_lon, min_lat, max_lon, max_lat)
                log.info("  → %d/%d places inside bbox", len(places), len(candidates))
                counts = categorise(places)
                writer.writerow({"patch_id": patch_id, "datazone": dz_id, "name": name, **counts})
                fout.flush()
                log.info("  → %d places (after bbox filter)", counts["total"])
            except requests.RequestException as exc:
                log.error("  Request failed for %s: %s — skipping", patch_id, exc)
                time.sleep(5)
                continue

            time.sleep(args.delay)

    log.info("Done. Output: %s", output_path)


if __name__ == "__main__":
    main()
