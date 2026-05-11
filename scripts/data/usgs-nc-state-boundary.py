"""
Download and extract the USGS North Carolina state boundary.

The script writes a dissolved boundary GeoPackage suitable for masking rasters.

Run from the project root:

    python scripts/data/usgs-nc-state-boundary.py
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import requests


BOUNDARY_URL = (
    "https://prd-tnm.s3.amazonaws.com/StagedProducts/GovtUnit/Shape/"
    "GOVTUNIT_North_Carolina_State_Shape.zip"
)
DEFAULT_DOWNLOAD_DIR = Path("data/boundaries/source")
DEFAULT_EXTRACT_DIR = Path("data/boundaries/extracted")
DEFAULT_OUTPUT = Path("data/boundaries/nc_state_boundary.gpkg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NC state boundary.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output boundary path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help=f"Directory for downloaded zip files. Defaults to {DEFAULT_DOWNLOAD_DIR}.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(DEFAULT_EXTRACT_DIR),
        help=f"Directory for extracted files. Defaults to {DEFAULT_EXTRACT_DIR}.",
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
        help="Redownload, re-extract, and overwrite the output.",
    )
    return parser.parse_args()


def download_zip(download_dir: Path, timeout: float, overwrite: bool) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(BOUNDARY_URL).path).name
    output_path = download_dir / filename

    if output_path.exists() and not overwrite:
        print(f"Using existing download: {output_path}")
        return output_path

    print(f"Downloading {BOUNDARY_URL}")
    response = requests.get(BOUNDARY_URL, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def extract_zip(zip_path: Path, extract_dir: Path, overwrite: bool) -> Path:
    target_dir = extract_dir / zip_path.stem
    marker = target_dir / ".extracted"

    if marker.exists() and not overwrite:
        print(f"Using existing extraction: {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} to {target_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_dir)
    marker.write_text("ok", encoding="utf-8")
    return target_dir


def find_boundary_shapefile(extracted_dir: Path) -> Path:
    shapefiles = sorted(extracted_dir.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found under {extracted_dir}.")

    polygon_matches = []
    for path in shapefiles:
        try:
            gdf = gpd.read_file(path, rows=1)
        except Exception:
            continue
        geom_types = set(gdf.geometry.geom_type.dropna())
        if any("Polygon" in geom_type for geom_type in geom_types):
            polygon_matches.append(path)

    if not polygon_matches:
        raise FileNotFoundError(f"No polygon shapefile found under {extracted_dir}.")

    state_matches = [
        path for path in polygon_matches if "state" in path.stem.lower()
    ]
    return state_matches[0] if state_matches else polygon_matches[0]


def write_boundary(extracted_dir: Path, output_path: Path, overwrite: bool) -> Path:
    if output_path.exists() and not overwrite:
        print(f"Using existing output: {output_path}")
        return output_path

    source_path = find_boundary_shapefile(extracted_dir)
    print(f"Reading boundary from {source_path}")
    boundary = gpd.read_file(source_path)
    boundary = boundary[boundary.geometry.notna()].copy()
    boundary = boundary.to_crs("EPSG:4326")
    dissolved = gpd.GeoDataFrame(
        {"name": ["North Carolina"]},
        geometry=[boundary.geometry.union_all()],
        crs=boundary.crs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dissolved.to_file(output_path, driver="GPKG")
    return output_path


def main() -> None:
    args = parse_args()
    zip_path = download_zip(Path(args.download_dir), args.timeout, args.overwrite)
    extracted_dir = extract_zip(zip_path, Path(args.extract_dir), args.overwrite)
    output_path = write_boundary(extracted_dir, Path(args.output), args.overwrite)
    print(f"Saved NC state boundary to {output_path}")


if __name__ == "__main__":
    main()
