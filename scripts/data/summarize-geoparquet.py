"""
Print column types and summary statistics for a GeoParquet dataset.

Examples:

    python scripts/data/summarize-geoparquet.py data/ebird/processed_nc_2020_2023/checklists.geoparquet
    python scripts/data/summarize-geoparquet.py data/ebird/processed_nc_2020_2023/checklists.geoparquet --top 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize variables, types, and basic statistics in a GeoParquet file."
    )
    parser.add_argument("input", help="Input GeoParquet path.")
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of most common values to print for non-numeric columns. Defaults to 5.",
    )
    return parser.parse_args()


def format_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_dataset_header(gdf: gpd.GeoDataFrame, path: Path) -> None:
    crs = None
    if gdf.crs is not None:
        crs = gdf.crs.to_string() if hasattr(gdf.crs, "to_string") else str(gdf.crs)

    print(f"File: {path}")
    print(f"Rows: {len(gdf):,}")
    print(f"Columns: {len(gdf.columns):,}")
    print(f"CRS: {crs}")
    if gdf.geometry.name in gdf.columns:
        geom_types = gdf.geometry.geom_type.value_counts(dropna=False)
        print("Geometry types:")
        for geom_type, count in geom_types.items():
            print(f"  {geom_type}: {count:,}")
    print()


def summarize_numeric(series: pd.Series) -> list[tuple[str, object]]:
    stats = series.describe(percentiles=[0.25, 0.5, 0.75])
    keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    return [(key, stats.get(key)) for key in keys]


def summarize_datetime(series: pd.Series) -> list[tuple[str, object]]:
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    return [
        ("min", non_missing.min()),
        ("max", non_missing.max()),
    ]


def summarize_values(series: pd.Series, top: int) -> list[tuple[str, object]]:
    counts = series.value_counts(dropna=False).head(top)
    return [(format_value(value), f"{count:,}") for value, count in counts.items()]


def print_column_summary(gdf: gpd.GeoDataFrame, column: str, top: int) -> None:
    series = gdf[column]
    missing = int(series.isna().sum())
    unique = series.nunique(dropna=True)

    print(f"## {column}")
    print(f"type: {series.dtype}")
    print(f"non-null: {series.notna().sum():,}")
    print(f"missing: {missing:,} ({missing / len(series):.1%})")
    print(f"unique non-null: {unique:,}")

    if column == gdf.geometry.name:
        bounds = gdf.total_bounds
        print(f"bounds: xmin={bounds[0]:.6g}, ymin={bounds[1]:.6g}, xmax={bounds[2]:.6g}, ymax={bounds[3]:.6g}")
    elif is_bool_dtype(series):
        print("values:")
        for value, count in summarize_values(series, top):
            print(f"  {value}: {count}")
    elif is_numeric_dtype(series):
        print("stats:")
        for key, value in summarize_numeric(series):
            print(f"  {key}: {format_value(value)}")
    elif is_datetime64_any_dtype(series):
        print("stats:")
        for key, value in summarize_datetime(series):
            print(f"  {key}: {format_value(value)}")
    else:
        print("top values:")
        for value, count in summarize_values(series, top):
            print(f"  {value}: {count}")
    print()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    gdf = gpd.read_parquet(path)
    print_dataset_header(gdf, path)
    for column in gdf.columns:
        print_column_summary(gdf, column, args.top)


if __name__ == "__main__":
    main()
