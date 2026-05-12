"""
Combine eBird CSV outputs into one GeoJSON point dataset.

The input CSVs should be outputs from scripts/data/ebird-historic-species.py,
which include WGS84 latitude/longitude columns named lat and lng.

Examples:

    python scripts/data/combine-ebird-geojson.py --inputs data/wood_thrush_nc_2020.csv data/wood_thrush_nc_2021.csv data/wood_thrush_nc_2022.csv data/wood_thrush_nc_2023.csv --crs EPSG:3857 --output data/wood_thrush_nc_2020_2023.geojson
    python scripts/data/combine-ebird-geojson.py --inputs data/wood_thrush_nc_2020.csv data/wood_thrush_nc_2021.csv data/wood_thrush_nc_2022.csv data/wood_thrush_nc_2023.csv --boundary data/boundaries/nc_state_boundary.gpkg --crs EPSG:3857 --output data/wood_thrush_nc_2020_2023.geojson
    python scripts/data/combine-ebird-geojson.py --inputs data/wood_thrush_nc_2020.csv data/wood_thrush_nc_2021.csv --south 35.51948 --north 36.07629 --west -78.99507 --east -78.25368 --crs EPSG:4326 --output data/wood_thrush_wake_2020_2021.geojson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


SOURCE_CRS = "EPSG:4326"
LAT_COLUMN = "lat"
LON_COLUMN = "lng"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine eBird CSV files into one GeoJSON point dataset."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more eBird CSV files to combine.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--crs",
        required=True,
        help="Target CRS for the output, e.g. EPSG:4326, EPSG:3857, or EPSG:5070.",
    )
    parser.add_argument(
        "--drop-missing-coordinates",
        action="store_true",
        help="Drop records without valid lat/lng instead of failing.",
    )
    parser.add_argument(
        "--boundary",
        help="Optional boundary vector path readable by GeoPandas. Keeps points intersecting the dissolved boundary.",
    )
    parser.add_argument(
        "--south",
        type=float,
        help="Optional WGS84 bbox south coordinate.",
    )
    parser.add_argument(
        "--north",
        type=float,
        help="Optional WGS84 bbox north coordinate.",
    )
    parser.add_argument(
        "--west",
        type=float,
        help="Optional WGS84 bbox west coordinate.",
    )
    parser.add_argument(
        "--east",
        type=float,
        help="Optional WGS84 bbox east coordinate.",
    )
    return parser.parse_args()


def read_ebird_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    df = pd.read_csv(path)
    missing_columns = {LAT_COLUMN, LON_COLUMN}.difference(df.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {columns}")

    df["sourceFile"] = str(path)
    return df


def combine_inputs(paths: list[Path], drop_missing_coordinates: bool) -> gpd.GeoDataFrame:
    frames = [read_ebird_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)

    combined[LAT_COLUMN] = pd.to_numeric(combined[LAT_COLUMN], errors="coerce")
    combined[LON_COLUMN] = pd.to_numeric(combined[LON_COLUMN], errors="coerce")

    missing_coordinates = combined[LAT_COLUMN].isna() | combined[LON_COLUMN].isna()
    if missing_coordinates.any():
        count = int(missing_coordinates.sum())
        if not drop_missing_coordinates:
            raise ValueError(
                f"{count} record(s) have missing or invalid lat/lng. "
                "Use --drop-missing-coordinates to omit them."
            )
        combined = combined.loc[~missing_coordinates].copy()

    return gpd.GeoDataFrame(
        combined,
        geometry=gpd.points_from_xy(combined[LON_COLUMN], combined[LAT_COLUMN]),
        crs=SOURCE_CRS,
    )


def validate_bbox(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    bbox_values = (args.south, args.north, args.west, args.east)
    if all(value is None for value in bbox_values):
        return None
    if any(value is None for value in bbox_values):
        raise ValueError("--south, --north, --west, and --east must be provided together.")
    if args.south >= args.north:
        raise ValueError("--south must be less than --north.")
    if args.west >= args.east:
        raise ValueError("--west must be less than --east.")
    return args.south, args.north, args.west, args.east


def filter_to_bbox(
    gdf: gpd.GeoDataFrame, bbox_values: tuple[float, float, float, float] | None
) -> gpd.GeoDataFrame:
    if bbox_values is None:
        return gdf

    south, north, west, east = bbox_values
    bbox_geom = box(west, south, east, north)
    filtered = gdf.loc[gdf.geometry.intersects(bbox_geom)].copy()
    print(f"Bbox filter kept {len(filtered):,} of {len(gdf):,} observations")
    return filtered


def filter_to_boundary(gdf: gpd.GeoDataFrame, boundary_path: str | None) -> gpd.GeoDataFrame:
    if boundary_path is None:
        return gdf

    path = Path(boundary_path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {path}")

    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")

    boundary = boundary.to_crs(gdf.crs)
    if hasattr(boundary.geometry, "union_all"):
        boundary_geom = boundary.geometry.union_all()
    else:
        boundary_geom = boundary.geometry.unary_union
    filtered = gdf.loc[gdf.geometry.intersects(boundary_geom)].copy()
    print(f"Boundary filter kept {len(filtered):,} of {len(gdf):,} observations")
    return filtered


def write_geojson(gdf: gpd.GeoDataFrame, output_path: Path, target_crs: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected = gdf.to_crs(target_crs)
    projected.to_file(output_path, driver="GeoJSON")
    print(f"Wrote {len(projected):,} observations to {output_path}")
    print(f"Output CRS: {projected.crs}")


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)

    gdf = combine_inputs(input_paths, args.drop_missing_coordinates)
    gdf = filter_to_bbox(gdf, validate_bbox(args))
    gdf = filter_to_boundary(gdf, args.boundary)
    write_geojson(gdf, output_path, args.crs)


if __name__ == "__main__":
    main()
