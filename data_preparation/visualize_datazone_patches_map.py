"""Generate an interactive Folium map of satellite patches with shapefile boundaries."""

import argparse
import base64
import io
import math
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds


def tiff_to_image_overlay(tiff_path: str, name: str, max_px: int = 2048, opacity: float = 0.6) -> folium.raster_layers.ImageOverlay:
    with rasterio.open(tiff_path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

        scale = min(max_px / src.width, max_px / src.height)
        out_w = math.ceil(src.width * scale)
        out_h = math.ceil(src.height * scale)

        if src.crs.to_epsg() == 4326:
            data = src.read(
                out_shape=(src.count, out_h, out_w),
                resampling=Resampling.lanczos,
            )
        else:
            dst_crs = "EPSG:4326"
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )
            scale2 = min(max_px / width, max_px / height)
            out_w2 = math.ceil(width * scale2)
            out_h2 = math.ceil(height * scale2)
            data = np.zeros((src.count, out_h2, out_w2), dtype=np.uint8)
            new_transform = transform * transform.scale(width / out_w2, height / out_h2)
            for i in range(src.count):
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=data[i],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.lanczos,
                )

    arr = np.moveaxis(data[:3], 0, -1)
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64}"

    west, south, east, north = bounds_wgs84
    return folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=[[south, west], [north, east]],
        name=name,
        opacity=opacity,
        show=False,
    )


def file_url(path_str):
    return Path(path_str).resolve().as_uri()


def sat_popup_html(row):
    sat_uri = file_url(row["satellite_patch"])
    ntl_uri = file_url(row["ntl_patch"])
    return (
        f"<b>{row['datazone']}</b> &nbsp; SV count: {row['sv_count']}<br>"
        f"<table style='margin-top:6px'><tr>"
        f"<td style='padding-right:6px;text-align:center'>"
        f"<a href='{sat_uri}' target='_blank'>"
        f"<img src='{sat_uri}' width='180' style='display:block'></a>"
        f"<small>Satellite</small></td>"
        f"<td style='text-align:center'>"
        f"<a href='{ntl_uri}' target='_blank'>"
        f"<img src='{ntl_uri}' width='180' style='display:block'></a>"
        f"<small>NTL</small></td>"
        f"</tr></table>"
    )


def sv_popup_html(row, project_root):
    abs_path = project_root / row["image_path"]
    img_uri = abs_path.resolve().as_uri()
    img_name = Path(row["image_path"]).name
    return (
        f"<b>{img_name}</b><br>DZ: {row['datazone']}<br>"
        f"<a href='{img_uri}' target='_blank'>"
        f"<img src='{img_uri}' width='280' style='margin-top:6px;display:block'></a>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--satellite-csv",
        default="dataset/satellite_dataset/satellite_metadata.csv",
    )
    parser.add_argument(
        "--streetview-csv",
        default="dataset/streetview_metadata.csv",
    )
    parser.add_argument(
        "--shapefile",
        default="dataset/glasgow_datazone/glasgow_datazone.shp",
    )
    parser.add_argument(
        "--output",
        default="outputs/datazone_patches_map.html",
    )
    parser.add_argument(
        "--satellite-tiff",
        default="dataset/TIFF/glasgow/glasgow.tif",
    )
    parser.add_argument(
        "--ntl-tiff",
        default="dataset/TIFF/glasgow_ntl/glasgow_ntl.tif",
    )
    parser.add_argument(
        "--tiff-max-px",
        type=int,
        default=8192,
        help="Max pixel dimension when downsampling TIFFs",
    )
    args = parser.parse_args()

    project_root = Path.cwd()

    df = pd.read_csv(args.satellite_csv)
    gdf = gpd.read_file(args.shapefile)

    patch_dzs = set(df["datazone"].tolist())
    gdf["has_patch"] = gdf["DataZone"].isin(patch_dzs)
    gdf["has_streetview"] = (
        gdf["DataZone"]
        .map(df.set_index("datazone")["has_streetview"].to_dict())
        .fillna(False)
        .astype(bool)
    )
    gdf["sv_count"] = (
        gdf["DataZone"]
        .map(df.set_index("datazone")["sv_count"].to_dict())
        .fillna(0)
        .astype(int)
    )

    center_lat = df["centroid_lat"].mean()
    center_lon = df["centroid_lon"].mean()
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron"
    )

    def style_func(feature):
        has_sv = feature["properties"]["has_streetview"]
        has_p = feature["properties"]["has_patch"]
        if has_sv:
            return {"fillColor": "#DE966395", "color": "#E2C314", "weight": 1.5, "fillOpacity": 0.2}
        elif has_p:
            return {"fillColor": "#DE966395", "color": "#C9AF19", "weight": 1.5, "fillOpacity": 0.2}
        return {"fillColor": "#DE966395", "color": "#C9AF19", "weight": 0.8, "fillOpacity": 0.1}

    folium.GeoJson(
        gdf[["DataZone", "Name", "has_patch", "has_streetview", "sv_count", "geometry"]].to_json(),
        name="Datazone Boundaries",
        style_function=style_func,
        tooltip=folium.GeoJsonTooltip(
            fields=["DataZone", "Name", "sv_count", "has_streetview"],
            aliases=["Datazone:", "Name:", "SV Count:", "Has Streetview:"],
            localize=True,
        ),
    ).add_to(m)

    sat_tiff_path = project_root / args.satellite_tiff
    ntl_tiff_path = project_root / args.ntl_tiff
    if sat_tiff_path.exists():
        print("Loading satellite TIFF…")
        tiff_to_image_overlay(str(sat_tiff_path), "Satellite TIFF", args.tiff_max_px, opacity=0.7).add_to(m)
    if ntl_tiff_path.exists():
        print("Loading NTL TIFF…")
        tiff_to_image_overlay(str(ntl_tiff_path), "NTL TIFF", args.tiff_max_px, opacity=0.7).add_to(m)

    sat_rect_group = folium.FeatureGroup(name="Satellite Patch Bounds", show=True)
    sat_center_group = folium.FeatureGroup(name="Satellite Patch Centers", show=True)
    for _, row in df.iterrows():
        color = "#2068B6"
        folium.Rectangle(
            bounds=[
                [row["bbox_min_lat"], row["bbox_min_lon"]],
                [row["bbox_max_lat"], row["bbox_max_lon"]],
            ],
            color=color,
            fill=True,
            fill_opacity=0.15,
            weight=1.5,
            dash_array="5",
        ).add_to(sat_rect_group)

        folium.CircleMarker(
            location=[row["centroid_lat"], row["centroid_lon"]],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.9,
            weight=1,
            popup=folium.Popup(sat_popup_html(row), max_width=420),
            tooltip=row["datazone"],
        ).add_to(sat_center_group)

    sat_rect_group.add_to(m)
    sat_center_group.add_to(m)

    sv_meta = pd.read_csv(args.streetview_csv)
    sv_group = folium.FeatureGroup(name="Street View Locations", show=False)
    for _, sv_row in sv_meta.iterrows():
        folium.CircleMarker(
            location=[sv_row["lat"], sv_row["lon"]],
            radius=1.5,
            color="#FF7B00",
            fill=True,
            fill_opacity=0.6,
            weight=0.5,
            popup=folium.Popup(sv_popup_html(sv_row, project_root), max_width=320),
            tooltip=sv_row["datazone"],
        ).add_to(sv_group)
    sv_group.add_to(m)
    n_sv_points = len(sv_meta)

    n_sv = int(df["has_streetview"].sum())
    n_nosv = int((~df["has_streetview"]).sum())
    legend_html = """
    <div style="position:fixed;bottom:50px;left:50px;z-index:1000;background:white;
                padding:12px;border:2px solid grey;border-radius:5px;font-size:13px;line-height:1.6;">
    <b>Datazone Satellite Patches</b><br>
    <span style="color:#3388ff;">&#9632;</span> Polygon: has streetview ({} zones)<br>
    <span style="color:#ff4444;">&#9632;</span> Polygon: no streetview ({} zones)<br>
    <span style="border:1px dashed #0066cc;padding:0 4px;">&#9634;</span> Satellite patch bbox<br>
    <span style="color:#0066cc;">&#9679;</span> Patch center → click for satellite/NTL images<br>
    <span style="color:#ff8800;">&#9679;</span> Street view location ({} points) → click for image<br>
    <span style="opacity:0.6;">&#9632;</span> Satellite TIFF / NTL TIFF (toggle in layer control)<br>
    Total datazones: {}
    </div>
    """.format(n_sv, n_nosv, n_sv_points, len(df))
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    m.save(args.output)
    print(f"Saved to {args.output}")
    print(
        f"Datazones in shp: {len(gdf)}, "
        f"with patch: {gdf['has_patch'].sum()}, "
        f"with SV: {gdf['has_streetview'].sum()}"
    )


if __name__ == "__main__":
    main()
