"""
Download OpenTopography USGS 3DEP DEM data for a WGS84 bounding box.

Requires an OpenTopography API key in .env:

    OPEN_TOPOGRAPHY_API_KEY=your_token_here

Examples:

    python scripts/data/opentopography-dem-bbox.py --south 40.234 --north 40.288 --west -105.673 --east -105.583
    python scripts/data/opentopography-dem-bbox.py --south 35.51948 --north 36.07629 --west -78.99507 --east -78.25368 --output data/wake_county_usgs30m.tif
    python scripts/data/opentopography-dem-bbox.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --output data/nc_usgs30m.tif
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from typing import Any

import requests
from dotenv import load_dotenv
from pyproj import Geod


OPEN_TOPOGRAPHY_DEM_URL = "https://portal.opentopography.org/API/usgsdem"
DEFAULT_DATASET = "USGS30m"
DEFAULT_OUTPUT_FORMAT = "GTiff"
OUTPUT_EXTENSIONS = {
    "GTiff": ".tif",
    "AAIGrid": ".asc",
    "HFA": ".img",
}
MAX_TILE_AREA_KM2 = 200_000


@dataclass(frozen=True)
class BBox:
    south: float
    north: float
    west: float
    east: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenTopography USGS 3DEP DEM data for a WGS84 bbox."
    )
    parser.add_argument(
        "--dataset",
        choices=("USGS30m", "USGS10m", "USGS1m"),
        default=DEFAULT_DATASET,
        help="OpenTopography USGS DEM dataset. Defaults to USGS30m.",
    )
    parser.add_argument(
        "--south",
        type=float,
        required=True,
        help="WGS84 bounding box south latitude.",
    )
    parser.add_argument(
        "--north",
        type=float,
        required=True,
        help="WGS84 bounding box north latitude.",
    )
    parser.add_argument(
        "--west",
        type=float,
        required=True,
        help="WGS84 bounding box west longitude.",
    )
    parser.add_argument(
        "--east",
        type=float,
        required=True,
        help="WGS84 bounding box east longitude.",
    )
    parser.add_argument(
        "--output-format",
        choices=("GTiff", "AAIGrid", "HFA"),
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format. Defaults to GTiff.",
    )
    parser.add_argument(
        "--output",
        help="Output file path. Defaults to a generated file under data/.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Request timeout in seconds. Defaults to 300.",
    )
    parser.add_argument(
        "--max-tile-area-km2",
        type=float,
        default=MAX_TILE_AREA_KM2,
        help=(
            "Maximum estimated area per API request before tiling. "
            f"Defaults to {MAX_TILE_AREA_KM2:,.0f} km2."
        ),
    )
    parser.add_argument(
        "--tile-dir",
        help="Directory for temporary tile downloads. Defaults to <output stem>_tiles.",
    )
    parser.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep downloaded tile files after mosaicking.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated area and tile bboxes without downloading.",
    )
    return parser.parse_args()


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPEN_TOPOGRAPHY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPEN_TOPOGRAPHY_API_KEY is not set. Add it to .env or your environment."
        )
    return api_key


def validate_bbox(south: float, north: float, west: float, east: float) -> None:
    if not -90 <= south <= 90:
        raise ValueError("--south must be between -90 and 90.")
    if not -90 <= north <= 90:
        raise ValueError("--north must be between -90 and 90.")
    if not -180 <= west <= 180:
        raise ValueError("--west must be between -180 and 180.")
    if not -180 <= east <= 180:
        raise ValueError("--east must be between -180 and 180.")
    if south >= north:
        raise ValueError("--south must be less than --north.")
    if west >= east:
        raise ValueError("--west must be less than --east.")


def bbox_area_km2(bbox: BBox) -> float:
    geod = Geod(ellps="WGS84")
    lon = [bbox.west, bbox.east, bbox.east, bbox.west, bbox.west]
    lat = [bbox.south, bbox.south, bbox.north, bbox.north, bbox.south]
    area_m2, _ = geod.polygon_area_perimeter(lon, lat)
    return abs(area_m2) / 1_000_000


def split_bbox(bbox: BBox, max_tile_area_km2: float) -> list[BBox]:
    if max_tile_area_km2 <= 0:
        raise ValueError("--max-tile-area-km2 must be greater than 0.")

    rows = 1
    cols = 1
    mid_lat_radians = math.radians((bbox.south + bbox.north) / 2)

    while True:
        lat_step = (bbox.north - bbox.south) / rows
        lon_step = (bbox.east - bbox.west) / cols
        largest_tile = max(
            bbox_area_km2(
                BBox(
                    south=bbox.south + row * lat_step,
                    north=bbox.south + (row + 1) * lat_step,
                    west=bbox.west + col * lon_step,
                    east=bbox.west + (col + 1) * lon_step,
                )
            )
            for row in range(rows)
            for col in range(cols)
        )

        if largest_tile <= max_tile_area_km2:
            break

        lon_span_kmish = (bbox.east - bbox.west) * max(math.cos(mid_lat_radians), 0.1) / cols
        lat_span_kmish = (bbox.north - bbox.south) / rows
        if lon_span_kmish >= lat_span_kmish:
            cols += 1
        else:
            rows += 1

    tiles = []
    lat_step = (bbox.north - bbox.south) / rows
    lon_step = (bbox.east - bbox.west) / cols
    for row in range(rows):
        for col in range(cols):
            tiles.append(
                BBox(
                    south=bbox.south + row * lat_step,
                    north=bbox.south + (row + 1) * lat_step,
                    west=bbox.west + col * lon_step,
                    east=bbox.west + (col + 1) * lon_step,
                )
            )

    return tiles


def redact_api_key_from_url(url: str) -> str:
    parts = urlsplit(url)
    query = [
        (key, "REDACTED" if key.lower() == "api_key" else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def build_default_output_path(
    dataset: str,
    south: float,
    north: float,
    west: float,
    east: float,
    output_format: str,
) -> Path:
    ext = OUTPUT_EXTENSIONS[output_format]
    bbox_label = (
        f"s{south:.5f}_n{north:.5f}_w{west:.5f}_e{east:.5f}"
        .replace("-", "m")
        .replace(".", "p")
    )
    return Path("data") / f"opentopography_{dataset.lower()}_{bbox_label}{ext}"


def build_query(
    dataset: str,
    bbox: BBox,
    output_format: str,
    api_key: str,
) -> dict[str, Any]:
    return {
        "datasetName": dataset,
        "south": bbox.south,
        "north": bbox.north,
        "west": bbox.west,
        "east": bbox.east,
        "outputFormat": output_format,
        "API_Key": api_key,
    }


def download_single_dem(
    query: dict[str, Any],
    output_path: Path,
    timeout: float,
) -> Path:
    response = requests.get(OPEN_TOPOGRAPHY_DEM_URL, params=query, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"OpenTopography request failed with HTTP {response.status_code}: "
            f"{redact_api_key_from_url(response.url)}\n{response.text}"
        ) from exc

    content_type = response.headers.get("Content-Type", "")
    if (
        "application/json" in content_type
        or "application/xml" in content_type
        or "text/" in content_type
    ):
        body_preview = response.text[:500]
        raise RuntimeError(
            "OpenTopography returned a text response instead of a DEM file. "
            f"Response preview:\n{body_preview}"
        )

    output_path.write_bytes(response.content)
    return output_path


def mosaic_geotiff_tiles(tile_paths: list[Path], output_path: Path) -> None:
    try:
        import rasterio
        from rasterio.merge import merge
    except ImportError as exc:
        raise RuntimeError(
            "Tiled GeoTIFF mosaicking requires rasterio. Install project "
            "requirements, then rerun this command."
        ) from exc

    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()


def cleanup_tiles(tile_paths: list[Path], tile_dir: Path) -> None:
    for path in tile_paths:
        path.unlink(missing_ok=True)

    try:
        tile_dir.rmdir()
    except OSError:
        pass


def output_path_from_args(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)

    return build_default_output_path(
        args.dataset,
        args.south,
        args.north,
        args.west,
        args.east,
        args.output_format,
    )


def download_dem(args: argparse.Namespace) -> Path:
    validate_bbox(args.south, args.north, args.west, args.east)
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.max_tile_area_km2 <= 0:
        raise ValueError("--max-tile-area-km2 must be greater than 0.")

    api_key = "REDACTED" if args.dry_run else get_api_key()
    bbox = BBox(
        south=args.south,
        north=args.north,
        west=args.west,
        east=args.east,
    )
    output_path = output_path_from_args(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    area_km2 = bbox_area_km2(bbox)
    tiles = split_bbox(bbox, args.max_tile_area_km2)

    if len(tiles) == 1:
        query = build_query(args.dataset, bbox, args.output_format, api_key)
        if args.dry_run:
            print(f"Would download 1 tile covering approximately {area_km2:,.0f} km2")
            request = requests.Request("GET", OPEN_TOPOGRAPHY_DEM_URL, params=query)
            print(redact_api_key_from_url(request.prepare().url))
            return output_path

        print(f"Downloading 1 tile covering approximately {area_km2:,.0f} km2")
        return download_single_dem(query, output_path, args.timeout)

    if args.output_format != "GTiff":
        raise ValueError("Automatic mosaicking is only supported with --output-format GTiff.")

    tile_dir = Path(args.tile_dir) if args.tile_dir else output_path.with_suffix("").with_name(
        f"{output_path.stem}_tiles"
    )
    tile_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            f"Requested bbox is approximately {area_km2:,.0f} km2; "
            f"would download {len(tiles)} tiles under "
            f"{args.max_tile_area_km2:,.0f} km2 each"
        )
        for index, tile in enumerate(tiles, start=1):
            print(
                f"Tile {index}/{len(tiles)}: "
                f"south={tile.south:.6f}, north={tile.north:.6f}, "
                f"west={tile.west:.6f}, east={tile.east:.6f}, "
                f"area={bbox_area_km2(tile):,.0f} km2"
            )
        return output_path

    print(
        f"Requested bbox is approximately {area_km2:,.0f} km2; "
        f"downloading {len(tiles)} tiles under {args.max_tile_area_km2:,.0f} km2 each"
    )

    tile_paths = []
    for index, tile in enumerate(tiles, start=1):
        tile_path = tile_dir / f"{output_path.stem}_tile_{index:03d}.tif"
        query = build_query(args.dataset, tile, args.output_format, api_key)
        print(
            f"Downloading tile {index}/{len(tiles)}: "
            f"south={tile.south:.6f}, north={tile.north:.6f}, "
            f"west={tile.west:.6f}, east={tile.east:.6f}"
        )
        download_single_dem(query, tile_path, args.timeout)
        tile_paths.append(tile_path)

    print(f"Mosaicking {len(tile_paths)} tiles into {output_path}")
    mosaic_geotiff_tiles(tile_paths, output_path)

    if not args.keep_tiles:
        cleanup_tiles(tile_paths, tile_dir)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = download_dem(args)
    if args.dry_run:
        print(f"Dry run complete. Output path would be {output_path}")
    else:
        print(f"Saved DEM to {output_path}")


if __name__ == "__main__":
    main()
