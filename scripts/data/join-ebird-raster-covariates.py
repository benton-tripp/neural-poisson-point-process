"""
Sample raster covariates at eBird observation points and write a GeoJSON.

The raster can be a multi-band stack. Band descriptions are converted into
stable snake_case column names. For the current North Carolina covariate stack,
the TCC yearly bands are kept as separate columns and a canopy_median column is
added by default.

Examples:

    python scripts/data/join-ebird-raster-covariates.py --points data/wood_thrush_nc_2020_2023.geojson --raster data/nc_covariate_stack.tif --output data/wood_thrush_nc_2020_2023_covariates.geojson
    python scripts/data/join-ebird-raster-covariates.py --points data/wood_thrush_nc_2020_2023.geojson --raster data/nc_covariate_stack.tif --output data/wood_thrush_nc_2020_2023_covariates.geojson --output-crs EPSG:5070
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


DEFAULT_CANOPY_PREFIX = "tcc_"
DEFAULT_CANOPY_AGGREGATE = "canopy_median"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join multi-band raster covariates to point observations."
    )
    parser.add_argument(
        "--points",
        required=True,
        help="Input point dataset readable by GeoPandas, such as eBird GeoJSON.",
    )
    parser.add_argument(
        "--raster",
        required=True,
        help="Input raster stack path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--output-crs",
        help="Optional CRS for the output GeoJSON. Defaults to the input point CRS.",
    )
    parser.add_argument(
        "--canopy-prefix",
        default=DEFAULT_CANOPY_PREFIX,
        help=f"Column prefix used to identify yearly canopy bands. Defaults to {DEFAULT_CANOPY_PREFIX}.",
    )
    parser.add_argument(
        "--canopy-aggregate-column",
        default=DEFAULT_CANOPY_AGGREGATE,
        help=f"Output column for median canopy across yearly bands. Defaults to {DEFAULT_CANOPY_AGGREGATE}.",
    )
    parser.add_argument(
        "--no-canopy-aggregate",
        action="store_true",
        help="Do not add the median canopy aggregate column.",
    )
    parser.add_argument(
        "--drop-missing-covariates",
        action="store_true",
        help="Drop points where all sampled raster bands are missing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "band"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    return unique


def raster_band_names(src: rasterio.DatasetReader) -> list[str]:
    raw_names = []
    for index, description in enumerate(src.descriptions, start=1):
        name = description if description else f"band_{index}"
        raw_names.append(snake_case(name))
    return unique_names(raw_names)


def load_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Point file does not exist: {path}")

    points = gpd.read_file(path)
    if points.empty:
        raise ValueError(f"Point file has no records: {path}")
    if points.crs is None:
        raise ValueError(f"Point file has no CRS: {path}")

    points = points[points.geometry.notna()].copy()
    points = points[~points.geometry.is_empty].copy()
    points = points[points.geometry.geom_type == "Point"].copy()
    if points.empty:
        raise ValueError("Point file has no valid point geometries.")
    return points


def sample_raster(points: gpd.GeoDataFrame, raster_path: Path) -> tuple[gpd.GeoDataFrame, list[str]]:
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file does not exist: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        band_names = raster_band_names(src)
        raster_points = points.to_crs(src.crs)
        coordinates = [(geom.x, geom.y) for geom in raster_points.geometry]
        sampled = np.asarray(list(src.sample(coordinates, masked=True)), dtype=np.float64)

        for band_index, nodata in enumerate(src.nodatavals):
            if nodata is not None:
                sampled[:, band_index] = np.where(
                    sampled[:, band_index] == nodata,
                    np.nan,
                    sampled[:, band_index],
                )

        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)

    output = points.copy()
    for column, values in zip(band_names, sampled.T):
        output[column] = values
    return output, band_names


def add_canopy_aggregate(
    points: gpd.GeoDataFrame,
    band_names: list[str],
    canopy_prefix: str,
    aggregate_column: str,
) -> gpd.GeoDataFrame:
    canopy_columns = [name for name in band_names if name.startswith(canopy_prefix)]
    if not canopy_columns:
        print(f"No canopy columns matched prefix {canopy_prefix!r}; skipping aggregate.")
        return points

    points = points.copy()
    points[aggregate_column] = points[canopy_columns].median(axis=1, skipna=True)
    print(
        f"Added {aggregate_column} as row median across "
        f"{', '.join(canopy_columns)}"
    )
    return points


def write_output(points: gpd.GeoDataFrame, output_path: Path, output_crs: str | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_crs is not None:
        points = points.to_crs(output_crs)
    points.to_file(output_path, driver="GeoJSON")
    print(f"Wrote {len(points):,} observations to {output_path}")
    print(f"Output CRS: {points.crs}")


def main() -> None:
    args = parse_args()
    points_path = Path(args.points)
    raster_path = Path(args.raster)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        print(f"Using existing output: {output_path}")
        return

    points = load_points(points_path)
    joined, band_names = sample_raster(points, raster_path)
    if not args.no_canopy_aggregate:
        joined = add_canopy_aggregate(
            joined,
            band_names,
            canopy_prefix=args.canopy_prefix,
            aggregate_column=args.canopy_aggregate_column,
        )

    if args.drop_missing_covariates:
        before = len(joined)
        joined = joined.loc[~joined[band_names].isna().all(axis=1)].copy()
        print(f"Dropped {before - len(joined):,} observations with all covariates missing.")

    write_output(joined, output_path, args.output_crs)


if __name__ == "__main__":
    main()
