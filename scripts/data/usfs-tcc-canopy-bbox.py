"""
Download USFS Science Tree Canopy Cover rasters for a WGS84 bbox and year(s).

The service provides annual 30 m percent tree canopy cover for CONUS from
1985 through 2023. Values are 0-100 percent canopy cover; 254 is the
non-processing mask and 255 is background.

Examples, run from the project root:

    python scripts/data/usfs-tcc-canopy-bbox.py --south 35.51948 --north 36.07629 --west -78.99507 --east -78.25368 --years 2023 --output data/wake_tcc_2023.tif
    python scripts/data/usfs-tcc-canopy-bbox.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --start-year 2020 --end-year 2023 --output data/nc_tcc_2020_2023.tif
    python scripts/data/usfs-tcc-canopy-bbox.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --years 2023 --output data/nc_tcc_2023.tif --boundary data/boundaries/nc_state_boundary.gpkg
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pyproj import Transformer
from raster_masking import mask_raster_with_boundary


TCC_IMAGE_SERVER_URL = (
    "https://imagery.geoplatform.gov/iipp/rest/services/"
    "Vegetation/USFS_EDW_Science_TCC_CONUS/ImageServer"
)
EXPORT_IMAGE_URL = f"{TCC_IMAGE_SERVER_URL}/exportImage"
MIN_YEAR = 1985
MAX_YEAR = 2023
DEFAULT_PIXEL_SIZE_METERS = 30
DEFAULT_MAX_TILE_SIZE = 4096
WEB_MERCATOR_WKID = 3857
WGS84_WKID = 4326
TCC_VALID_MIN = 0
TCC_VALID_MAX = 100
TCC_NODATA = 255


@dataclass(frozen=True)
class Wgs84BBox:
    south: float
    north: float
    west: float
    east: float


@dataclass(frozen=True)
class ProjectedBBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class RasterTile:
    row: int
    col: int
    bbox: ProjectedBBox
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download USFS Science TCC canopy cover rasters for a bbox."
    )
    parser.add_argument("--south", type=float, required=True, help="WGS84 south latitude.")
    parser.add_argument("--north", type=float, required=True, help="WGS84 north latitude.")
    parser.add_argument("--west", type=float, required=True, help="WGS84 west longitude.")
    parser.add_argument("--east", type=float, required=True, help="WGS84 east longitude.")

    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        "--years",
        type=int,
        nargs="+",
        help="One or more years to download, e.g. --years 2020 2021 2022.",
    )
    year_group.add_argument(
        "--start-year",
        type=int,
        help="First year in an inclusive year range. Requires --end-year.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        help="Last year in an inclusive year range when --start-year is used.",
    )
    parser.add_argument(
        "--output",
        help="Output GeoTIFF path. Defaults to a generated file under data/.",
    )
    parser.add_argument(
        "--boundary",
        help="Optional vector boundary used to mask the final GeoTIFF output.",
    )
    parser.add_argument(
        "--tile-dir",
        help="Directory for intermediate per-year and tile GeoTIFFs. Defaults to <output stem>_years.",
    )
    parser.add_argument(
        "--max-tile-size",
        type=int,
        default=DEFAULT_MAX_TILE_SIZE,
        help=(
            "Maximum tile width/height in pixels per ImageServer request. "
            f"Defaults to {DEFAULT_MAX_TILE_SIZE}."
        ),
    )
    parser.add_argument(
        "--keep-year-files",
        action="store_true",
        help="Keep per-year GeoTIFFs after creating a multiband stack.",
    )
    parser.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep intermediate request tile GeoTIFFs after mosaicking each year.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_METERS,
        help="Output pixel size in meters. Defaults to 30.",
    )
    parser.add_argument(
        "--raster-function",
        choices=("None", "NLCDTCC_noBkgrd"),
        default="None",
        help="Optional server raster function. Defaults to None.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Request timeout in seconds. Defaults to 300.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries for timeouts, connection errors, HTTP 429, and HTTP 5xx. Defaults to 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print request details without downloading.",
    )
    parser.add_argument(
        "--show-urls",
        action="store_true",
        help="With --dry-run, print full exportImage URLs for every tile.",
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


def parse_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        years = args.years
    else:
        if args.end_year is None:
            raise ValueError("--end-year is required when --start-year is used.")
        if args.start_year > args.end_year:
            raise ValueError("--start-year must be less than or equal to --end-year.")
        years = list(range(args.start_year, args.end_year + 1))

    invalid = [year for year in years if year < MIN_YEAR or year > MAX_YEAR]
    if invalid:
        raise ValueError(f"Years must be between {MIN_YEAR} and {MAX_YEAR}: {invalid}")

    return sorted(set(years))


def project_bbox_to_web_mercator(bbox: Wgs84BBox) -> ProjectedBBox:
    transformer = Transformer.from_crs(WGS84_WKID, WEB_MERCATOR_WKID, always_xy=True)
    xmin, ymin = transformer.transform(bbox.west, bbox.south)
    xmax, ymax = transformer.transform(bbox.east, bbox.north)
    return ProjectedBBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def image_size(projected_bbox: ProjectedBBox, pixel_size: float) -> tuple[int, int]:
    if pixel_size <= 0:
        raise ValueError("--pixel-size must be greater than 0.")

    width = math.ceil((projected_bbox.xmax - projected_bbox.xmin) / pixel_size)
    height = math.ceil((projected_bbox.ymax - projected_bbox.ymin) / pixel_size)
    if width <= 0 or height <= 0:
        raise ValueError("Computed image size is empty.")

    return width, height


def plan_tiles(
    projected_bbox: ProjectedBBox,
    width: int,
    height: int,
    max_tile_size: int,
) -> list[RasterTile]:
    if max_tile_size <= 0:
        raise ValueError("--max-tile-size must be greater than 0.")

    x_resolution = (projected_bbox.xmax - projected_bbox.xmin) / width
    y_resolution = (projected_bbox.ymax - projected_bbox.ymin) / height
    rows = math.ceil(height / max_tile_size)
    cols = math.ceil(width / max_tile_size)

    tiles = []
    for row in range(rows):
        y_start = row * max_tile_size
        y_stop = min((row + 1) * max_tile_size, height)
        tile_height = y_stop - y_start

        # ArcGIS bbox uses ymin/ymax, while raster rows are counted from the top.
        tile_ymax = projected_bbox.ymax - y_start * y_resolution
        tile_ymin = projected_bbox.ymax - y_stop * y_resolution

        for col in range(cols):
            x_start = col * max_tile_size
            x_stop = min((col + 1) * max_tile_size, width)
            tile_width = x_stop - x_start
            tile_xmin = projected_bbox.xmin + x_start * x_resolution
            tile_xmax = projected_bbox.xmin + x_stop * x_resolution

            tiles.append(
                RasterTile(
                    row=row,
                    col=col,
                    bbox=ProjectedBBox(
                        xmin=tile_xmin,
                        ymin=tile_ymin,
                        xmax=tile_xmax,
                        ymax=tile_ymax,
                    ),
                    width=tile_width,
                    height=tile_height,
                )
            )

    return tiles


def default_output_path(bbox: Wgs84BBox, years: list[int]) -> Path:
    year_label = str(years[0]) if len(years) == 1 else f"{years[0]}_{years[-1]}"
    bbox_label = (
        f"s{bbox.south:.5f}_n{bbox.north:.5f}_w{bbox.west:.5f}_e{bbox.east:.5f}"
        .replace("-", "m")
        .replace(".", "p")
    )
    return Path("data") / f"usfs_tcc_{year_label}_{bbox_label}.tif"


def build_export_params(
    projected_bbox: ProjectedBBox,
    width: int,
    height: int,
    year: int,
    raster_function: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "f": "image",
        "bbox": (
            f"{projected_bbox.xmin},{projected_bbox.ymin},"
            f"{projected_bbox.xmax},{projected_bbox.ymax}"
        ),
        "bboxSR": WEB_MERCATOR_WKID,
        "imageSR": WEB_MERCATOR_WKID,
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "U8",
        "interpolation": "RSP_NearestNeighbor",
        "mosaicRule": json.dumps(
            {
                "mosaicMethod": "esriMosaicAttribute",
                "where": f"beginyear = {year}",
                "sortField": "beginyear",
                "sortValue": str(year),
                "ascending": True,
            }
        ),
    }

    if raster_function != "None":
        params["renderingRule"] = json.dumps({"rasterFunction": raster_function})

    return params


def ensure_binary_raster_response(response: requests.Response, year: int) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"USFS TCC request failed for {year} with HTTP "
            f"{response.status_code}: {response.url}\n{response.text[:1000]}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type or "text/" in content_type:
        raise RuntimeError(
            f"USFS TCC returned a text response for {year} instead of a GeoTIFF:\n"
            f"{response.text[:1000]}"
        )


def download_year(
    year: int,
    params: dict[str, Any],
    output_path: Path,
    timeout: float,
    retries: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retryable_statuses = {429, 500, 502, 503, 504}
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(EXPORT_IMAGE_URL, params=params, timeout=timeout)
            if response.status_code in retryable_statuses and attempt < attempts:
                wait_seconds = min(2 ** (attempt - 1), 60)
                print(
                    f"{year}: HTTP {response.status_code}; "
                    f"retrying in {wait_seconds}s ({attempt}/{retries})"
                )
                time.sleep(wait_seconds)
                continue

            ensure_binary_raster_response(response, year)
            output_path.write_bytes(response.content)
            return output_path

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"USFS TCC request failed for {year} after {attempts} attempts: {exc}"
                ) from exc

            wait_seconds = min(2 ** (attempt - 1), 60)
            print(
                f"{year}: request timed out or connection failed; "
                f"retrying in {wait_seconds}s ({attempt}/{retries})"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"USFS TCC request failed for {year} unexpectedly.")


def stack_year_files(year_paths: list[Path], years: list[int], output_path: Path) -> None:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "Stacking multiple years requires rasterio. The per-year GeoTIFFs "
            "were downloaded; install rasterio or rerun for one year at a time."
        ) from exc

    datasets = [rasterio.open(path) for path in year_paths]
    try:
        profile = datasets[0].profile.copy()
        profile.update(count=len(datasets), compress="lzw", tiled=True, BIGTIFF="IF_SAFER")

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index, (dataset, year) in enumerate(zip(datasets, years), start=1):
                dst.write(dataset.read(1), band_index)
                dst.set_band_description(band_index, f"TCC {year}")
    finally:
        for dataset in datasets:
            dataset.close()


def mask_tcc_codes(raster_path: Path) -> None:
    """Set USFS TCC mask codes to nodata while preserving 0-100 canopy values."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Masking TCC codes requires rasterio.") from exc

    temp_path = raster_path.with_name(f"{raster_path.stem}_tcc_masked{raster_path.suffix}")
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        profile.update(nodata=TCC_NODATA)
        descriptions = src.descriptions

        with rasterio.open(temp_path, "w", **profile) as dst:
            total_masked = 0
            for band_index in range(1, src.count + 1):
                data = src.read(band_index)
                invalid = (data < TCC_VALID_MIN) | (data > TCC_VALID_MAX)
                total_masked += int(invalid.sum())
                data = data.copy()
                data[invalid] = TCC_NODATA
                dst.write(data, band_index)
                description = descriptions[band_index - 1]
                if description:
                    dst.set_band_description(band_index, description)

    temp_path.replace(raster_path)
    print(
        f"Masked {total_masked:,} TCC cells outside "
        f"{TCC_VALID_MIN}-{TCC_VALID_MAX} to nodata ({TCC_NODATA})"
    )


def mosaic_tile_files(tile_paths: list[Path], output_path: Path) -> None:
    try:
        import rasterio
        from rasterio.merge import merge
    except ImportError as exc:
        raise RuntimeError(
            "Mosaicking tiled downloads requires rasterio. Install rasterio "
            "or rerun with a smaller bbox that fits in one request."
        ) from exc

    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()


def cleanup_year_files(year_paths: list[Path], year_dir: Path) -> None:
    for path in year_paths:
        path.unlink(missing_ok=True)
    try:
        year_dir.rmdir()
    except OSError:
        pass


def cleanup_tile_files(tile_paths: list[Path], tile_dir: Path) -> None:
    for path in tile_paths:
        path.unlink(missing_ok=True)

    try:
        tile_dir.rmdir()
    except OSError:
        pass


def download_year_tiled(
    year: int,
    tiles: list[RasterTile],
    year_output_path: Path,
    base_tile_dir: Path,
    raster_function: str,
    timeout: float,
    retries: int,
    keep_tiles: bool,
) -> Path:
    if len(tiles) == 1:
        tile = tiles[0]
        params = build_export_params(
            tile.bbox, tile.width, tile.height, year, raster_function
        )
        download_year(year, params, year_output_path, timeout, retries)
        return year_output_path

    tile_dir = base_tile_dir / f"{year}_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths = []

    for index, tile in enumerate(tiles, start=1):
        tile_path = tile_dir / (
            f"{year_output_path.stem}_r{tile.row:03d}_c{tile.col:03d}.tif"
        )
        params = build_export_params(
            tile.bbox, tile.width, tile.height, year, raster_function
        )
        print(
            f"Downloading {year} tile {index}/{len(tiles)} "
            f"({tile.width} x {tile.height})"
        )
        download_year(year, params, tile_path, timeout, retries)
        tile_paths.append(tile_path)

    print(f"Mosaicking {len(tile_paths)} tiles for {year} into {year_output_path}")
    mosaic_tile_files(tile_paths, year_output_path)

    if not keep_tiles:
        cleanup_tile_files(tile_paths, tile_dir)

    return year_output_path


def download_canopy(args: argparse.Namespace) -> Path:
    bbox = Wgs84BBox(south=args.south, north=args.north, west=args.west, east=args.east)
    validate_bbox(bbox)
    years = parse_years(args)
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.retries < 0:
        raise ValueError("--retries must be 0 or greater.")

    projected_bbox = project_bbox_to_web_mercator(bbox)
    width, height = image_size(projected_bbox, args.pixel_size)
    tiles = plan_tiles(projected_bbox, width, height, args.max_tile_size)
    output_path = Path(args.output) if args.output else default_output_path(bbox, years)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Requesting {len(years)} year(s) at {width:,} x {height:,} pixels "
        f"({width * height:,} pixels per year), "
        f"{len(tiles)} tile request(s) per year"
    )

    if args.dry_run:
        print(
            f"Tile size limit: {args.max_tile_size:,} pixels; "
            f"total requests: {len(years) * len(tiles):,}"
        )
        for year in years:
            print(f"{year}: {len(tiles)} tile request(s)")
            for index, tile in enumerate(tiles, start=1):
                print(
                    f"  tile {index}/{len(tiles)} "
                    f"row={tile.row} col={tile.col} "
                    f"size={tile.width}x{tile.height} "
                    f"bbox={tile.bbox.xmin:.2f},{tile.bbox.ymin:.2f},"
                    f"{tile.bbox.xmax:.2f},{tile.bbox.ymax:.2f}"
                )
                if args.show_urls:
                    params = build_export_params(
                        tile.bbox, tile.width, tile.height, year, args.raster_function
                    )
                    request = requests.Request("GET", EXPORT_IMAGE_URL, params=params)
                    print(f"    {request.prepare().url}")
        return output_path

    year_dir = Path(args.tile_dir) if args.tile_dir else output_path.with_suffix("").with_name(
        f"{output_path.stem}_years"
    )

    if len(years) == 1:
        download_year_tiled(
            year=years[0],
            tiles=tiles,
            year_output_path=output_path,
            base_tile_dir=year_dir,
            raster_function=args.raster_function,
            timeout=args.timeout,
            retries=args.retries,
            keep_tiles=args.keep_tiles,
        )
        if args.boundary:
            print(f"Masking {output_path} to {args.boundary}")
            mask_raster_with_boundary(output_path, args.boundary)
        mask_tcc_codes(output_path)
        return output_path

    year_paths = []
    for year in years:
        year_path = year_dir / f"{output_path.stem}_{year}.tif"
        print(f"Downloading {year} to {year_path}")
        download_year_tiled(
            year=year,
            tiles=tiles,
            year_output_path=year_path,
            base_tile_dir=year_dir,
            raster_function=args.raster_function,
            timeout=args.timeout,
            retries=args.retries,
            keep_tiles=args.keep_tiles,
        )
        year_paths.append(year_path)

    print(f"Stacking {len(year_paths)} years into {output_path}")
    stack_year_files(year_paths, years, output_path)

    if args.boundary:
        print(f"Masking {output_path} to {args.boundary}")
        mask_raster_with_boundary(output_path, args.boundary)

    mask_tcc_codes(output_path)

    if not args.keep_year_files:
        cleanup_year_files(year_paths, year_dir)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = download_canopy(args)
    if args.dry_run:
        print(f"Dry run complete. Output path would be {output_path}")
    else:
        print(f"Saved canopy raster to {output_path}")


if __name__ == "__main__":
    main()
