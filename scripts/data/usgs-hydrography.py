"""
Download USGS Small-scale hydrography data and create distance-to-water rasters.

This script downloads the USGS 1:1,000,000-scale hydrography FileGDB archive
`hydrusm010g.gdb_nt00897.tar.gz`, extracts its Waterbody and Coastline layers,
rasterizes them in a metric CRS, and writes a two-band GeoTIFF:

    band 1: distance to nearest waterbody, meters
    band 2: distance to nearest coastline, meters

Examples, run from the project root:

    python scripts/data/usgs-hydrography.py --south 35.51948 --north 36.07629 --west -78.99507 --east -78.25368 --resolution 500 --output data/wake_hydro_distance_500m.tif

    python scripts/data/usgs-hydrography.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --resolution 100 --output data/nc_hydro_distance_100m.tif
    python scripts/data/usgs-hydrography.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --output data/nc_hydro_distance_1km.tif --boundary data/boundaries/nc_state_boundary.gpkg
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import requests
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from raster_masking import mask_raster_with_boundary


DEFAULT_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/Small-scale/data/"
    "Hydrography/hydrusm010g.gdb_nt00897.tar.gz"
)
DEFAULT_DOWNLOAD_DIR = Path("data/hydrography/source")
DEFAULT_EXTRACT_DIR = Path("data/hydrography/extracted")
DEFAULT_OUTPUT = Path("data/hydrography/nc_hydro_distance_1km.tif")
DEFAULT_CRS = "EPSG:5070"
DEFAULT_RESOLUTION = 1000
DEFAULT_SEARCH_BUFFER = 500_000
NODATA = -9999.0


@dataclass(frozen=True)
class Wgs84BBox:
    south: float
    north: float
    west: float
    east: float


@dataclass(frozen=True)
class ProjectedGrid:
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    transform: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create distance-to-waterbody and distance-to-coastline rasters."
    )
    parser.add_argument("--south", type=float, required=True, help="WGS84 south latitude.")
    parser.add_argument("--north", type=float, required=True, help="WGS84 north latitude.")
    parser.add_argument("--west", type=float, required=True, help="WGS84 west longitude.")
    parser.add_argument("--east", type=float, required=True, help="WGS84 east longitude.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output GeoTIFF path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--boundary",
        help="Optional vector boundary used to mask the final GeoTIFF output.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=f"Directory for downloaded zip files. Defaults to {DEFAULT_DOWNLOAD_DIR}.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        help=f"Directory for extracted hydrography files. Defaults to {DEFAULT_EXTRACT_DIR}.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help="Output raster resolution in projected CRS meters. Defaults to 1000.",
    )
    parser.add_argument(
        "--crs",
        default=DEFAULT_CRS,
        help="Output projected CRS. Defaults to EPSG:5070.",
    )
    parser.add_argument(
        "--waterbody-filter-field",
        default="Feature",
        help="Field used with --exclude-waterbody-values. Defaults to Feature.",
    )
    parser.add_argument(
        "--exclude-waterbody-values",
        nargs="*",
        default=["Lake Dry"],
        help=(
            "Values to exclude from the waterbody filter field. "
            "Use an empty value list to disable. Defaults to Lake Dry if present."
        ),
    )
    parser.add_argument(
        "--search-buffer",
        type=float,
        default=DEFAULT_SEARCH_BUFFER,
        help=(
            "Meters to buffer around the requested bbox while computing distances. "
            f"Defaults to {DEFAULT_SEARCH_BUFFER:g}."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Download timeout in seconds. Defaults to 300.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and overwrite extracted/source files.",
    )
    return parser.parse_args()


def validate_bbox(bbox: Wgs84BBox) -> None:
    if not -90 <= bbox.south <= 90:
        raise ValueError("--south must be between -90 and 90.")
    if not -90 <= bbox.north <= 90:
        raise ValueError("--north must be between -90 and 90.")
    if not -180 <= bbox.west <= 180:
        raise ValueError("--west must be between -180 and 180.")
    if not -180 <= bbox.east <= 180:
        raise ValueError("--east must be between -180 and 180.")
    if bbox.south >= bbox.north:
        raise ValueError("--south must be less than --north.")
    if bbox.west >= bbox.east:
        raise ValueError("--west must be less than --east.")


def archive_stem(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".zip"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def download_archive(url: str, download_dir: Path, timeout: float, overwrite: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    if not filename:
        raise ValueError("Could not determine filename from URL.")

    output_path = download_dir / filename
    if output_path.exists() and not overwrite:
        print(f"Using existing download: {output_path}")
        return output_path

    print(f"Downloading {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def extract_archive(archive_path: Path, extract_dir: Path, overwrite: bool) -> Path:
    target_dir = extract_dir / archive_stem(archive_path)
    marker = target_dir / ".extracted"
    if marker.exists() and not overwrite:
        print(f"Using existing extraction: {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} to {target_dir}")
    if archive_path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(target_dir)
    elif archive_path.name.lower().endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, mode="r:gz") as archive:
            archive.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    marker.write_text("ok", encoding="utf-8")
    return target_dir


def find_vector_source(extracted_dir: Path, layer_name: str) -> tuple[Path, str | None]:
    gpkg_files = list(extracted_dir.rglob("*.gpkg"))
    if gpkg_files:
        return gpkg_files[0], layer_name

    gdb_dirs = list(extracted_dir.rglob("*.gdb"))
    if gdb_dirs:
        return gdb_dirs[0], layer_name

    layer_lower = layer_name.lower()
    shapefiles = [
        path
        for path in extracted_dir.rglob("*.shp")
        if path.stem.lower() == layer_lower or layer_lower in path.stem.lower()
    ]
    if not shapefiles:
        raise FileNotFoundError(f"Could not find layer or shapefile matching {layer_name}.")
    return shapefiles[0], None


def read_layer(extracted_dir: Path, layer_name: str) -> gpd.GeoDataFrame:
    source_path, layer = find_vector_source(extracted_dir, layer_name)
    print(f"Reading {layer_name} from {source_path}")
    if layer:
        return gpd.read_file(source_path, layer=layer)
    return gpd.read_file(source_path)


def project_bbox(bbox: Wgs84BBox, dst_crs: str) -> tuple[float, float, float, float]:
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox.west, bbox.south)
    xmax, ymax = transformer.transform(bbox.east, bbox.north)
    return min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)


def make_grid(bounds: tuple[float, float, float, float], resolution: float) -> ProjectedGrid:
    if resolution <= 0:
        raise ValueError("--resolution must be greater than 0.")

    xmin, ymin, xmax, ymax = bounds
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))
    if width <= 0 or height <= 0:
        raise ValueError("Computed output grid is empty.")

    transform = from_origin(xmin, ymax, resolution, resolution)
    return ProjectedGrid(bounds=bounds, width=width, height=height, transform=transform)


def make_buffered_grid(
    output_grid: ProjectedGrid,
    resolution: float,
    search_buffer: float,
) -> tuple[ProjectedGrid, int]:
    if search_buffer < 0:
        raise ValueError("--search-buffer must be 0 or greater.")

    buffer_pixels = int(np.ceil(search_buffer / resolution))
    xmin, ymin, xmax, ymax = output_grid.bounds
    buffered_bounds = (
        xmin - buffer_pixels * resolution,
        ymin - buffer_pixels * resolution,
        xmax + buffer_pixels * resolution,
        ymax + buffer_pixels * resolution,
    )
    buffered_grid = ProjectedGrid(
        bounds=buffered_bounds,
        width=output_grid.width + 2 * buffer_pixels,
        height=output_grid.height + 2 * buffer_pixels,
        transform=from_origin(
            buffered_bounds[0],
            buffered_bounds[3],
            resolution,
            resolution,
        ),
    )
    return buffered_grid, buffer_pixels


def clean_waterbodies(
    gdf: gpd.GeoDataFrame,
    field: str,
    excluded_values: list[str],
) -> gpd.GeoDataFrame:
    if not excluded_values or field not in gdf.columns:
        return gdf

    return gdf[~gdf[field].astype(str).isin(excluded_values)].copy()


def prepare_features(
    gdf: gpd.GeoDataFrame,
    output_crs: str,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.to_crs(output_crs)
    bbox_geom = box(*bounds)
    # Keep geometries that intersect the output bbox. Distances to features
    # outside the bbox are not represented unless the source extent includes
    # them in this cropped set.
    gdf = gdf[gdf.intersects(bbox_geom)].copy()
    return gdf


def distance_raster(
    gdf: gpd.GeoDataFrame,
    grid: ProjectedGrid,
    resolution: float,
    output_height: int,
    output_width: int,
    crop_offset_pixels: int,
) -> np.ndarray:
    if gdf.empty:
        return np.full((output_height, output_width), NODATA, dtype=np.float32)

    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    feature_mask = rasterize(
        shapes=shapes,
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    if not feature_mask.any():
        return np.full((output_height, output_width), NODATA, dtype=np.float32)

    distances = distance_transform_edt(~feature_mask) * resolution
    if crop_offset_pixels:
        distances = distances[
            crop_offset_pixels : crop_offset_pixels + output_height,
            crop_offset_pixels : crop_offset_pixels + output_width,
        ]
    return distances.astype(np.float32)


def write_distance_stack(
    output_path: Path,
    water_distance: np.ndarray,
    coast_distance: np.ndarray,
    grid: ProjectedGrid,
    crs: str,
) -> None:
    import rasterio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 2,
        "dtype": "float32",
        "crs": crs,
        "transform": grid.transform,
        "nodata": NODATA,
        "compress": "lzw",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(water_distance, 1)
        dst.write(coast_distance, 2)
        dst.set_band_description(1, "distance_to_waterbody_m")
        dst.set_band_description(2, "distance_to_coastline_m")


def build_hydro_distance_raster(args: argparse.Namespace) -> Path:
    bbox = Wgs84BBox(south=args.south, north=args.north, west=args.west, east=args.east)
    validate_bbox(bbox)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Using existing output: {output_path}")
        if args.boundary:
            print(f"Masking existing {output_path} to {args.boundary}")
            mask_raster_with_boundary(output_path, args.boundary)
        return output_path

    archive_path = download_archive(
        url=DEFAULT_URL,
        download_dir=Path(args.download_dir),
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    extracted_dir = extract_archive(archive_path, Path(args.extract_dir), args.overwrite)

    output_bounds = project_bbox(bbox, args.crs)
    grid = make_grid(output_bounds, args.resolution)
    processing_grid, crop_offset = make_buffered_grid(
        grid,
        resolution=args.resolution,
        search_buffer=args.search_buffer,
    )
    print(f"Output grid: {grid.width:,} x {grid.height:,} at {args.resolution:g} m")
    print(
        f"Processing grid: {processing_grid.width:,} x {processing_grid.height:,} "
        f"with {args.search_buffer:g} m search buffer"
    )

    waterbody = read_layer(extracted_dir, "Waterbody")
    coastline = read_layer(extracted_dir, "Coastline")

    waterbody = clean_waterbodies(
        waterbody,
        field=args.waterbody_filter_field,
        excluded_values=args.exclude_waterbody_values,
    )
    waterbody = prepare_features(waterbody, args.crs, processing_grid.bounds)
    coastline = prepare_features(coastline, args.crs, processing_grid.bounds)

    print(f"Waterbody features intersecting bbox: {len(waterbody):,}")
    print(f"Coastline features intersecting bbox: {len(coastline):,}")

    water_distance = distance_raster(
        waterbody,
        processing_grid,
        args.resolution,
        output_height=grid.height,
        output_width=grid.width,
        crop_offset_pixels=crop_offset,
    )
    coast_distance = distance_raster(
        coastline,
        processing_grid,
        args.resolution,
        output_height=grid.height,
        output_width=grid.width,
        crop_offset_pixels=crop_offset,
    )

    write_distance_stack(output_path, water_distance, coast_distance, grid, args.crs)
    if args.boundary:
        print(f"Masking {output_path} to {args.boundary}")
        mask_raster_with_boundary(output_path, args.boundary)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_hydro_distance_raster(args)
    print(f"Saved hydrography distance raster to {output_path}")


if __name__ == "__main__":
    main()
