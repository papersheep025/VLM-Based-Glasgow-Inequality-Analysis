#!/usr/bin/env python3
"""Assign Glasgow datazones to OSM POIs by point-in-polygon matching.

This script reads:
  - dataset/osm_poi/osm_poi.csv
  - dataset/glasgow_datazone/glasgow_datazone.shp + .dbf

And writes:
  - dataset/osm_poi/datazone_poi.csv

It uses only the Python standard library so it can run in minimal
environments without geopandas/fiona/shapely.
"""

from __future__ import annotations

import csv
import math
import pathlib
import struct
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
POI_CSV = ROOT / "dataset" / "osm_poi" / "osm_poi.csv"
SHP_PATH = ROOT / "dataset" / "glasgow_datazone" / "glasgow_datazone.shp"
DBF_PATH = ROOT / "dataset" / "glasgow_datazone" / "glasgow_datazone.dbf"
OUT_CSV = ROOT / "dataset" / "osm_poi" / "datazone_poi.csv"


@dataclass(frozen=True)
class PolygonFeature:
    datazone: str
    rings: Tuple[Tuple[Tuple[float, float], ...], ...]
    bbox: Tuple[float, float, float, float]


def read_dbf_records(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("rb") as f:
        header = f.read(32)
        if len(header) != 32:
            raise ValueError("DBF header is truncated")
        nrecords = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]

        fields: list[tuple[str, str, int, int]] = []
        desc_bytes = f.read(header_len - 32)
        if len(desc_bytes) != header_len - 32:
            raise ValueError("Unexpected end of DBF field descriptors")
        for offset in range(0, len(desc_bytes), 32):
            block = desc_bytes[offset : offset + 32]
            if not block:
                break
            if block[0] == 0x0D:
                break
            name = block[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            ftype = chr(block[11])
            flen = block[16]
            dec = block[17]
            fields.append((name, ftype, flen, dec))

        records: list[dict[str, str]] = []
        for _ in range(nrecords):
            rec = f.read(record_len)
            if len(rec) != record_len:
                raise ValueError("Unexpected end of DBF records")
            if rec[:1] == b"*":
                continue

            offset = 1
            row: dict[str, str] = {}
            for name, ftype, flen, _dec in fields:
                raw = rec[offset : offset + flen]
                offset += flen
                if ftype == "C":
                    value = raw.decode("utf-8", errors="ignore").rstrip(" \x00")
                elif ftype in {"N", "F"}:
                    value = raw.decode("ascii", errors="ignore").strip()
                else:
                    value = raw.decode("utf-8", errors="ignore").strip()
                row[name] = value
            records.append(row)
    return records


def read_shp_features(path: pathlib.Path, datazones: Sequence[str]) -> list[PolygonFeature]:
    features: list[PolygonFeature] = []
    with path.open("rb") as f:
        header = f.read(100)
        if len(header) != 100:
            raise ValueError("SHP header is truncated")
        shape_type = struct.unpack("<i", header[32:36])[0]
        if shape_type != 5:
            raise ValueError(f"Expected polygon shapefile (type 5), got {shape_type}")

        idx = 0
        while True:
            rec_header = f.read(8)
            if not rec_header:
                break
            if len(rec_header) != 8:
                raise ValueError("Truncated SHP record header")
            _rec_num, content_len_words = struct.unpack(">2i", rec_header)
            content = f.read(content_len_words * 2)
            if len(content) != content_len_words * 2:
                raise ValueError("Truncated SHP record content")

            rec_shape_type = struct.unpack("<i", content[:4])[0]
            if rec_shape_type == 0:
                idx += 1
                continue
            if rec_shape_type != 5:
                raise ValueError(f"Unexpected shape type {rec_shape_type} in record {idx}")

            bbox = struct.unpack("<4d", content[4:36])
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + 4 * num_parts
            parts = struct.unpack(f"<{num_parts}i", content[parts_offset:points_offset])
            points_raw = struct.unpack(f"<{num_points * 2}d", content[points_offset:points_offset + 16 * num_points])
            points = list(zip(points_raw[0::2], points_raw[1::2]))

            rings: list[tuple[tuple[float, float], ...]] = []
            for i, start in enumerate(parts):
                end = parts[i + 1] if i + 1 < num_parts else num_points
                ring = tuple(points[start:end])
                if ring:
                    rings.append(ring)

            datazone = datazones[idx] if idx < len(datazones) else ""
            features.append(PolygonFeature(datazone=datazone, rings=tuple(rings), bbox=bbox))
            idx += 1

    return features


def point_on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float = 1e-12) -> bool:
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > eps:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= eps


def point_in_ring(px: float, py: float, ring: Sequence[Tuple[float, float]]) -> bool:
    if len(ring) < 3:
        return False
    inside = False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if point_on_segment(px, py, x1, y1, x2, y2):
            return True
        intersects = (y1 > py) != (y2 > py)
        if intersects:
            x_int = (x2 - x1) * (py - y1) / (y2 - y1 + 0.0) + x1
            if x_int == px:
                return True
            if x_int > px:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def point_in_polygon(px: float, py: float, rings: Sequence[Sequence[Tuple[float, float]]]) -> bool:
    inside = False
    for ring in rings:
        if point_in_ring(px, py, ring):
            inside = not inside
    return inside


def bbox_contains(bbox: Tuple[float, float, float, float], px: float, py: float) -> bool:
    xmin, ymin, xmax, ymax = bbox
    return xmin <= px <= xmax and ymin <= py <= ymax


def main() -> None:
    with POI_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        poi_rows = list(reader)

    dbf_rows = read_dbf_records(DBF_PATH)
    datazones = [row.get("DataZone", "").strip() for row in dbf_rows]
    features = read_shp_features(SHP_PATH, datazones)

    matched = 0
    unmatched = 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datazone", "poi_type", "poi_subtype", "lat", "lon"])

        for row in poi_rows:
            poi_type = row.get("poi_type", "")
            poi_subtype = row.get("poi_subtype", "")
            lat = row.get("lat", "")
            lon = row.get("lon", "")
            dz = ""

            try:
                px = float(lon)
                py = float(lat)
            except (TypeError, ValueError):
                px = py = math.nan

            if not math.isnan(px) and not math.isnan(py):
                for feature in features:
                    if not bbox_contains(feature.bbox, px, py):
                        continue
                    if point_in_polygon(px, py, feature.rings):
                        dz = feature.datazone
                        break

            if dz:
                matched += 1
                writer.writerow([dz, poi_type, poi_subtype, lat, lon])
            else:
                unmatched += 1

    print(f"wrote {OUT_CSV}")
    print(f"matched: {matched}")
    print(f"unmatched: {unmatched}")


if __name__ == "__main__":
    main()
