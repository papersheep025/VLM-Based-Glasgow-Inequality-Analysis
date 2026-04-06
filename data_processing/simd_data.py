from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd


DEFAULT_SHAPEFILE = Path("glasgow_datazone") / "glasgow_datazone.shp"
DEFAULT_SIMD_CSV = Path("SIMD") / "simd2020_withgeog" / "simd2020_withinds.csv"
DEFAULT_OUTPUT_CSV = Path("outputs") / "SIMD_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SIMD rows for the Glasgow datazones defined in the shapefile."
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=DEFAULT_SHAPEFILE,
        help="Path to the Glasgow datazone shapefile.",
    )
    parser.add_argument(
        "--simd-csv",
        type=Path,
        default=DEFAULT_SIMD_CSV,
        help="Path to the SIMD CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination CSV path.",
    )
    return parser.parse_args()


def normalize_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def first_existing_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise KeyError(f"Could not find any of these columns: {', '.join(candidates)}")


def load_datazone_codes(shapefile: Path) -> list[str]:
    gdf = gpd.read_file(shapefile)
    code_column = first_existing_column(gdf.columns, ("Data_Zone", "DataZone", "datazone"))
    codes = [
        code
        for code in (normalize_code(value) for value in gdf[code_column].tolist())
        if code is not None
    ]
    # Keep the original shapefile order, but remove duplicate codes if any are present.
    seen: set[str] = set()
    ordered_codes: list[str] = []
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        ordered_codes.append(code)
    return ordered_codes


def load_simd(simd_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(simd_csv, encoding="utf-8-sig")
    code_column = first_existing_column(df.columns, ("Data_Zone", "datazone", "DataZone"))
    df = df.copy()
    df[code_column] = df[code_column].map(normalize_code)
    df = df.dropna(subset=[code_column])
    if code_column != "Data_Zone":
        df = df.rename(columns={code_column: "Data_Zone"})
    return df


def build_output(shapefile: Path, simd_csv: Path) -> pd.DataFrame:
    datazone_codes = load_datazone_codes(shapefile)
    simd = load_simd(simd_csv)

    filtered = simd[simd["Data_Zone"].isin(datazone_codes)].copy()
    filtered.insert(0, "datazone", filtered["Data_Zone"])
    filtered = filtered.drop_duplicates(subset=["Data_Zone"], keep="first")

    # Preserve the shapefile order so downstream joins stay stable.
    order = pd.Categorical(filtered["Data_Zone"], categories=datazone_codes, ordered=True)
    filtered = filtered.assign(_order=order).sort_values("_order").drop(columns="_order")
    return filtered.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output = build_output(args.shapefile, args.simd_csv)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8")

    total = len(output)
    print(f"Saved {total} SIMD rows to {args.output_csv}")

    shapefile_codes = set(load_datazone_codes(args.shapefile))
    simd_codes = set(load_simd(args.simd_csv)["Data_Zone"].tolist())
    matched = shapefile_codes & simd_codes
    missing = shapefile_codes - simd_codes
    extra = simd_codes - shapefile_codes
    print(f"Matched: {len(matched)}")
    print(f"Missing from SIMD: {len(missing)}")
    print(f"SIMD rows outside shapefile: {len(extra)}")


if __name__ == "__main__":
    main()
