"""
Stack rasters into a common CRS, grid, and multi-band GeoTIFF.

By default, the output grid uses the coarsest resolution among the inputs
after projecting them to the requested CRS. Use nearest resampling for
categorical rasters and bilinear/cubic/average for continuous rasters.

When a boundary is provided, the stack can also be restricted to the shared
valid-data footprint of all output bands. This avoids inventing values where
source rasters have different support, especially along coastlines and boundary
edges.

Examples:

    python scripts/data/stack-rasters.py --inputs data/nc_tcc_2020_2023.tif data/nc_usgs30m_match_tcc.tif data/nc_hydro_distance_match_tcc.tif --crs EPSG:3857 --boundary data/boundaries/nc_state_boundary.gpkg --output data/nc_covariate_stack.tif
    python scripts/data/stack-rasters.py --inputs data/nc_tcc_2020_2023.tif data/nc_usgs30m_match_tcc.tif data/nc_hydro_distance_match_tcc.tif --crs EPSG:3857 --resampling bilinear --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --output data/nc_covariate_stack.tif
    python scripts/data/stack-rasters.py --inputs data/nc_tcc_2020_2023.tif data/nc_usgs30m_match_tcc.tif data/nc_hydro_distance_match_tcc.tif --crs EPSG:3857 --boundary data/boundaries/nc_state_boundary.gpkg --resampling nearest bilinear bilinear --mask-tcc-above 100 --output data/nc_covariate_stack.tif --overwrite
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box


SOURCE_BBOX_CRS = "EPSG:4326"
OUTPUT_DTYPE = "float32"
OUTPUT_NODATA = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack rasters into one multi-band GeoTIFF on a shared grid."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input raster paths. Multi-band rasters are expanded into multiple output bands.",
    )
    parser.add_argument("--output", required=True, help="Output multi-band GeoTIFF path.")
    parser.add_argument(
        "--crs",
        required=True,
        help="Target CRS for the stack, e.g. EPSG:3857, EPSG:5070, or EPSG:4326.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="Optional output pixel size in target CRS units. Defaults to the coarsest input resolution.",
    )
    parser.add_argument(
        "--resampling",
        nargs="+",
        choices=("nearest", "bilinear", "cubic", "average"),
        default=["bilinear"],
        help=(
            "Resampling method. Provide one value for all rasters or one per input. "
            "Use nearest for categorical rasters. Defaults to bilinear."
        ),
    )
    parser.add_argument(
        "--extent",
        choices=("intersection", "union"),
        default="intersection",
        help="Use the shared intersection or full union of input extents. Defaults to intersection.",
    )
    parser.add_argument(
        "--boundary",
        help="Optional boundary vector path readable by GeoPandas. Masks output outside the dissolved boundary.",
    )
    parser.add_argument(
        "--mask-tcc-above",
        type=float,
        help=(
            "For bands whose description starts with TCC, set values above this "
            "threshold to output nodata after reprojection. USFS TCC valid canopy "
            "values are 0-100; 254 is a non-processing mask and 255 is background."
        ),
    )
    parser.add_argument(
        "--valid-footprint",
        choices=("boundary", "intersection"),
        default="boundary",
        help=(
            "Output support mask. 'boundary' keeps every boundary cell and "
            "preserves interior nodata. 'intersection' keeps only cells where "
            "all output bands have valid data after reprojection and TCC "
            "masking. Defaults to boundary."
        ),
    )
    parser.add_argument("--south", type=float, help="Optional WGS84 bbox south coordinate.")
    parser.add_argument("--north", type=float, help="Optional WGS84 bbox north coordinate.")
    parser.add_argument("--west", type=float, help="Optional WGS84 bbox west coordinate.")
    parser.add_argument("--east", type=float, help="Optional WGS84 bbox east coordinate.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def resampling_from_name(name: str) -> Resampling:
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }[name]


def validate_resampling(methods: list[str], input_count: int) -> list[Resampling]:
    if len(methods) == 1:
        methods = methods * input_count
    elif len(methods) != input_count:
        raise ValueError(
            "--resampling must have either one value or exactly one value per input raster."
        )
    return [resampling_from_name(method) for method in methods]


def validate_bbox(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    values = (args.south, args.north, args.west, args.east)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("--south, --north, --west, and --east must be provided together.")
    if args.south >= args.north:
        raise ValueError("--south must be less than --north.")
    if args.west >= args.east:
        raise ValueError("--west must be less than --east.")
    return args.south, args.north, args.west, args.east


def combine_bounds(
    bounds: list[tuple[float, float, float, float]], extent_mode: str
) -> tuple[float, float, float, float]:
    if extent_mode == "union":
        return (
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )

    west = max(bound[0] for bound in bounds)
    south = max(bound[1] for bound in bounds)
    east = min(bound[2] for bound in bounds)
    north = min(bound[3] for bound in bounds)
    if west >= east or south >= north:
        raise ValueError("Input rasters do not have an overlapping extent in the target CRS.")
    return west, south, east, north


def intersect_bounds(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    west = max(first[0], second[0])
    south = max(first[1], second[1])
    east = min(first[2], second[2])
    north = min(first[3], second[3])
    if west >= east or south >= north:
        raise ValueError("The requested bbox/boundary does not overlap the raster stack extent.")
    return west, south, east, north


def raster_bounds_and_resolution(path: Path, target_crs: str) -> tuple[tuple[float, float, float, float], float]:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Input raster has no CRS: {path}")

        target_bounds = transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)
        transform, _, _ = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        resolution = max(abs(transform.a), abs(transform.e))
    return target_bounds, resolution


def load_boundary(
    boundary_path: str | None, target_crs: str
) -> tuple[list[object], tuple[float, float, float, float]] | None:
    if boundary_path is None:
        return None

    path = Path(boundary_path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {path}")

    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")

    boundary = boundary[boundary.geometry.notna()].copy()
    boundary = boundary.to_crs(target_crs)
    geometries = [geom for geom in boundary.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError(f"Boundary file has no valid geometries: {path}")

    return geometries, tuple(boundary.total_bounds)


def output_grid(
    input_paths: list[Path],
    target_crs: str,
    resolution: float | None,
    extent_mode: str,
    bbox_values: tuple[float, float, float, float] | None,
    boundary_info: tuple[list[object], tuple[float, float, float, float]] | None,
) -> tuple[rasterio.Affine, int, int, tuple[float, float, float, float], float]:
    bounds = []
    resolutions = []
    for path in input_paths:
        raster_bounds, raster_resolution = raster_bounds_and_resolution(path, target_crs)
        bounds.append(raster_bounds)
        resolutions.append(raster_resolution)

    stack_bounds = combine_bounds(bounds, extent_mode)
    if bbox_values is not None:
        south, north, west, east = bbox_values
        bbox_bounds = transform_bounds(SOURCE_BBOX_CRS, target_crs, west, south, east, north)
        stack_bounds = intersect_bounds(stack_bounds, bbox_bounds)
    if boundary_info is not None:
        stack_bounds = intersect_bounds(stack_bounds, boundary_info[1])

    pixel_size = resolution if resolution is not None else max(resolutions)
    if pixel_size <= 0:
        raise ValueError("--resolution must be greater than 0.")

    west, south, east, north = stack_bounds
    width = math.ceil((east - west) / pixel_size)
    height = math.ceil((north - south) / pixel_size)
    if width <= 0 or height <= 0:
        raise ValueError("Output grid has no cells.")

    adjusted_east = west + width * pixel_size
    adjusted_south = north - height * pixel_size
    adjusted_bounds = (west, adjusted_south, adjusted_east, north)
    transform = from_bounds(*adjusted_bounds, width=width, height=height)
    return transform, width, height, adjusted_bounds, pixel_size


def nodata_mask(values: np.ndarray) -> np.ndarray:
    return np.isnan(values) | (values == OUTPUT_NODATA)


def reproject_band_to_grid(
    input_path: Path,
    source_band: int,
    height: int,
    width: int,
    transform: rasterio.Affine,
    target_crs: str,
    resampling: Resampling,
) -> tuple[np.ndarray, str]:
    with rasterio.open(input_path) as src:
        destination = np.full((height, width), OUTPUT_NODATA, dtype=OUTPUT_DTYPE)
        reproject(
            source=rasterio.band(src, source_band),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=OUTPUT_NODATA,
            resampling=resampling,
        )

        description = src.descriptions[source_band - 1]
        if not description:
            if src.count == 1:
                description = input_path.stem
            else:
                description = f"{input_path.stem}_band_{source_band}"
    return destination, description


def stack_rasters(args: argparse.Namespace) -> Path:
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Using existing output: {output_path}")
        return output_path

    input_paths = [Path(path) for path in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input raster does not exist: {path}")

    resampling_methods = validate_resampling(args.resampling, len(input_paths))
    bbox_values = validate_bbox(args)
    boundary_info = load_boundary(args.boundary, args.crs)
    transform, width, height, bounds, pixel_size = output_grid(
        input_paths=input_paths,
        target_crs=args.crs,
        resolution=args.resolution,
        extent_mode=args.extent,
        bbox_values=bbox_values,
        boundary_info=boundary_info,
    )

    output_band_count = 0
    for path in input_paths:
        with rasterio.open(path) as src:
            output_band_count += src.count
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": output_band_count,
        "dtype": OUTPUT_DTYPE,
        "crs": args.crs,
        "transform": transform,
        "nodata": OUTPUT_NODATA,
        "compress": "lzw",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    print(
        f"Stack grid: {width:,} x {height:,} at {pixel_size:g} target CRS units "
        f"({args.crs})"
    )
    print(f"Stack bounds: {bounds}")

    boundary_mask = None
    if boundary_info is not None:
        boundary_mask = geometry_mask(
            boundary_info[0],
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )

    output_bands: list[tuple[np.ndarray, str]] = []
    valid_footprint = boundary_mask.copy() if boundary_mask is not None else np.ones((height, width), dtype=bool)
    for input_path, resampling in zip(input_paths, resampling_methods):
        with rasterio.open(input_path) as src:
            source_count = src.count
        for source_band in range(1, source_count + 1):
            destination, description = reproject_band_to_grid(
                input_path=input_path,
                source_band=source_band,
                height=height,
                width=width,
                transform=transform,
                target_crs=args.crs,
                resampling=resampling,
            )
            is_tcc = description.strip().lower().startswith("tcc")
            if args.mask_tcc_above is not None and is_tcc:
                destination[destination > args.mask_tcc_above] = OUTPUT_NODATA

            if boundary_mask is not None:
                destination[~boundary_mask] = OUTPUT_NODATA

            if args.valid_footprint == "intersection":
                valid_footprint &= ~nodata_mask(destination)

            output_bands.append((destination, description))

    if args.valid_footprint == "intersection":
        kept = int(valid_footprint.sum())
        total_domain = int(boundary_mask.sum()) if boundary_mask is not None else height * width
        print(f"Shared valid footprint kept {kept:,} of {total_domain:,} domain cells.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        for output_band, (destination, description) in enumerate(output_bands, start=1):
            if args.valid_footprint == "intersection":
                destination = destination.copy()
                destination[~valid_footprint] = OUTPUT_NODATA
            dst.write(destination, output_band)
            dst.set_band_description(output_band, description)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = stack_rasters(args)
    print(f"Saved raster stack to {output_path}")


if __name__ == "__main__":
    main()
