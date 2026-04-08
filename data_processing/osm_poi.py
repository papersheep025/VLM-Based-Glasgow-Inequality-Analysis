from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point


DEFAULT_SHAPEFILE = Path("glasgow_datazone") / "glasgow_datazone.shp"
DEFAULT_OUTPUT_DIR = Path("outputs") / "osm_poi"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_TAG_KEYS = (
    "amenity",
    "shop",
    "leisure",
    "tourism",
    "office",
    "healthcare",
    "public_transport",
    "craft",
    "sport",
    "historic",
    "emergency",
    "railway",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OpenStreetMap POIs inside Glasgow datazones.")
    parser.add_argument("--shapefile", type=Path, default=DEFAULT_SHAPEFILE, help="Path to the Glasgow datazone shapefile.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write POI outputs.")
    parser.add_argument("--overpass-url", type=str, default=DEFAULT_OVERPASS_URL, help="Overpass API endpoint.")
    parser.add_argument("--timeout", type=int, default=180, help="Overpass timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Number of request retries.")
    parser.add_argument("--sleep-seconds", type=float, default=10.0, help="Sleep between retries.")
    parser.add_argument(
        "--tag-keys",
        nargs="*",
        default=list(DEFAULT_TAG_KEYS),
        help="OSM tag keys to query, e.g. amenity shop leisure tourism.",
    )
    return parser.parse_args()


def normalize_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def first_existing_column(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise KeyError(f"Could not find any of these columns: {', '.join(candidates)}")


def load_datazones(shapefile: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shapefile)
    code_column = first_existing_column(gdf.columns, ("Data_Zone", "DataZone", "datazone"))
    gdf = gdf.copy()
    gdf["datazone"] = gdf[code_column].map(normalize_code)
    gdf = gdf.dropna(subset=["datazone"]).reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf[["datazone", "geometry"]]


def bbox_from_gdf(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(miny), float(minx), float(maxy), float(maxx)


def build_overpass_query(tag_key: str, bbox: tuple[float, float, float, float], timeout: int) -> str:
    south, west, north, east = bbox
    return f"""
[out:json][timeout:{timeout}];
(
  nwr["{tag_key}"]({south},{west},{north},{east});
);
out center tags;
""".strip()


def fetch_overpass(url: str, query: str, retries: int, sleep_seconds: float) -> dict[str, Any]:
    last_error: Exception | None = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "glasgow-vlm-osm-poi/1.0",
    }
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, data={"data": query}, headers=headers, timeout=600)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"Overpass request failed after {retries + 1} attempts: {last_error}")


def element_to_record(element: dict[str, Any], tag_key: str) -> dict[str, Any] | None:
    tags = element.get("tags", {})
    if not isinstance(tags, dict):
        tags = {}

    osm_type = element.get("type")
    osm_id = element.get("id")
    if osm_type is None or osm_id is None:
        return None

    lat = element.get("lat")
    lon = element.get("lon")
    if lat is None or lon is None:
        center = element.get("center", {})
        if isinstance(center, dict):
            lat = center.get("lat")
            lon = center.get("lon")
    if lat is None or lon is None:
        return None

    tag_value = tags.get(tag_key)
    primary_name = tags.get("name")

    return {
        "osm_type": str(osm_type),
        "osm_id": int(osm_id),
        "osm_url": f"https://www.openstreetmap.org/{osm_type}/{osm_id}",
        "poi_name": primary_name,
        "poi_type": tag_key,
        "poi_subtype": tag_value,
        "lat": float(lat),
        "lon": float(lon),
        "tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
    }


def collect_pois(overpass_url: str, bbox: tuple[float, float, float, float], tag_keys: list[str], timeout: int, retries: int, sleep_seconds: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for tag_key in tag_keys:
        print(f"Querying Overpass for tag key: {tag_key}")
        query = build_overpass_query(tag_key, bbox, timeout)
        payload = fetch_overpass(overpass_url, query, retries, sleep_seconds)
        for element in payload.get("elements", []):
            if not isinstance(element, dict):
                continue
            record = element_to_record(element, tag_key)
            if record is None:
                continue
            dedupe_key = (record["osm_type"], record["osm_id"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(record)
        print(f"Collected {len(records)} unique POIs so far")
    return pd.DataFrame(records)


def assign_datazones(datazones: gpd.GeoDataFrame, pois: pd.DataFrame) -> gpd.GeoDataFrame:
    if pois.empty:
        empty = gpd.GeoDataFrame(pois.copy(), geometry=gpd.GeoSeries([], crs="EPSG:4326"))
        return empty

    poi_gdf = gpd.GeoDataFrame(
        pois.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(pois["lon"], pois["lat"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(poi_gdf, datazones, how="left", predicate="intersects")
    if "datazone_right" in joined.columns:
        joined = joined.rename(columns={"datazone_right": "datazone"})
    if "datazone_left" in joined.columns:
        joined = joined.drop(columns=["datazone_left"])
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined


def write_outputs(output_dir: Path, joined: gpd.GeoDataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    poi_csv = output_dir / "osm_poi.csv"
    summary_csv = output_dir / "osm_poi_by_datazone.csv"

    detailed = pd.DataFrame(joined.drop(columns="geometry", errors="ignore"))
    detailed.to_csv(poi_csv, index=False, encoding="utf-8")

    if detailed.empty:
        summary = pd.DataFrame(columns=["datazone", "poi_count"])
    else:
        summary = (
            detailed.groupby("datazone")
            .agg(poi_count=("osm_id", "count"))
            .reset_index()
            .sort_values("datazone")
        )
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    print(f"Saved POI table to {poi_csv}")
    print(f"Saved datazone summary to {summary_csv}")
    print(f"Total POIs: {len(detailed)}")
    print(f"Matched datazones: {detailed['datazone'].nunique() if not detailed.empty else 0}")


def main() -> None:
    args = parse_args()
    datazones = load_datazones(args.shapefile)
    bbox = bbox_from_gdf(datazones)

    pois = collect_pois(
        overpass_url=args.overpass_url,
        bbox=bbox,
        tag_keys=[str(key) for key in args.tag_keys],
        timeout=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )

    joined = assign_datazones(datazones, pois)
    write_outputs(args.output_dir, joined)


if __name__ == "__main__":
    main()
