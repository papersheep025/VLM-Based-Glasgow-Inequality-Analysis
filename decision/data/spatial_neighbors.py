"""
Queen-contiguity neighbours for Glasgow datazones.

Uses a 10-metre buffer on the projected (OSGB36) geometries to close
floating-point boundary gaps between adjacent polygons.

Returns {datazone: [neighbour_dz, ...]} for all 746 datazones.
Datazones with no touching neighbours map to an empty list.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

DEFAULT_SHP = "dataset/glasgow_datazone/glasgow_datazone.shp"


def build_neighbors(
    shp_path: str | Path = DEFAULT_SHP,
    buffer_m: float = 10.0,
) -> dict[str, list[str]]:
    gdf = gpd.read_file(shp_path)[["DataZone", "geometry"]]
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=27700)

    left = gdf.rename(columns={"DataZone": "dz_left"}).copy()
    left["geometry"] = left.geometry.buffer(buffer_m)

    right = gdf.rename(columns={"DataZone": "dz_right"})
    joined = left.sjoin(right[["dz_right", "geometry"]], how="left", predicate="intersects")
    joined = joined[joined["dz_left"] != joined["dz_right"]]

    neighbors: dict[str, list[str]] = (
        joined.groupby("dz_left")["dz_right"].apply(list).to_dict()
    )
    for dz in gdf["DataZone"]:
        neighbors.setdefault(dz, [])
    return neighbors
