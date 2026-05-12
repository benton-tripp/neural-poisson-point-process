"""
Reproject a raster to exactly match another raster's grid.

Run from the project root:

    python scripts/data/reproject-raster-to-template.py --input data/nc_usgs30m.tif --template data/nc_tcc_2020_2023.tif --output data/nc_usgs30m_match_tcc.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproject a raster to a template grid.")
    parser.add_argument("--input", required=True, help="Input raster path.")
    parser.add_argument("--template", required=True, help="Template raster path.")
    parser.add_argument("--output", required=True, help="Output raster path.")
    parser.add_argument(
        "--resampling",
        choices=("nearest", "bilinear", "cubic"),
        default="bilinear",
        help="Resampling method. Defaults to bilinear.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def default_nodata(dtype: str):
    if np.issubdtype(np.dtype(dtype), np.floating):
        return -9999.0
    if np.issubdtype(np.dtype(dtype), np.unsignedinteger):
        return np.iinfo(dtype).max
    return -9999


def reproject_to_template(
    input_path: Path,
    template_path: Path,
    output_path: Path,
    resampling_name: str,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        print(f"Using existing output: {output_path}")
        return output_path

    resampling = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }[resampling_name]

    with rasterio.open(input_path) as src, rasterio.open(template_path) as template:
        nodata = src.nodata
        if nodata is None:
            nodata = default_nodata(src.dtypes[0])

        profile = src.profile.copy()
        profile.update(
            {
                "crs": template.crs,
                "transform": template.transform,
                "width": template.width,
                "height": template.height,
                "nodata": nodata,
                "compress": "lzw",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index in range(1, src.count + 1):
                destination = np.full(
                    (template.height, template.width),
                    nodata,
                    dtype=src.dtypes[band_index - 1],
                )
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=template.transform,
                    dst_crs=template.crs,
                    dst_nodata=nodata,
                    resampling=resampling,
                )
                dst.write(destination, band_index)
                description = src.descriptions[band_index - 1]
                if description:
                    dst.set_band_description(band_index, description)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = reproject_to_template(
        input_path=Path(args.input),
        template_path=Path(args.template),
        output_path=Path(args.output),
        resampling_name=args.resampling,
        overwrite=args.overwrite,
    )
    print(f"Saved reprojected raster to {output_path}")


if __name__ == "__main__":
    main()
