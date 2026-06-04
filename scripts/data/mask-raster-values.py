"""
Mask raster values outside a valid range.

This is useful for rasters whose source uses ordinary numeric codes for masks.
For example, USFS TCC uses 0-100 for valid percent canopy, 254 for a
non-processing mask, and 255 for background/nodata.

Examples:

    python scripts/data/mask-raster-values.py --input data/nc_tcc_2020_2023.tif --output data/nc_tcc_2020_2023.tif --valid-min 0 --valid-max 100 --nodata 255 --overwrite
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set raster values outside a valid range to nodata."
    )
    parser.add_argument("--input", required=True, help="Input raster path.")
    parser.add_argument("--output", required=True, help="Output raster path.")
    parser.add_argument("--valid-min", type=float, help="Minimum valid value, inclusive.")
    parser.add_argument("--valid-max", type=float, help="Maximum valid value, inclusive.")
    parser.add_argument(
        "--nodata",
        type=float,
        help="Output nodata value. Defaults to the input nodata value if present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def default_nodata(dtype: str) -> float | int:
    dtype_obj = np.dtype(dtype)
    if np.issubdtype(dtype_obj, np.floating):
        return -9999.0
    if np.issubdtype(dtype_obj, np.unsignedinteger):
        return np.iinfo(dtype_obj).max
    return -9999


def output_temp_path(output_path: Path) -> Path:
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{output_path.stem}_masked_",
        suffix=output_path.suffix,
        dir=output_path.parent,
    )
    os.close(temp_fd)
    return Path(temp_name)


def mask_raster_values(
    input_path: Path,
    output_path: Path,
    valid_min: float | None,
    valid_max: float | None,
    nodata_arg: float | None,
    overwrite: bool,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input raster does not exist: {input_path}")
    if output_path.exists() and input_path.resolve() != output_path.resolve() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_temp_path(output_path)

    try:
        with rasterio.open(input_path) as src:
            nodata = nodata_arg if nodata_arg is not None else src.nodata
            if nodata is None:
                nodata = default_nodata(src.dtypes[0])

            dtype = np.dtype(src.dtypes[0])
            nodata_value = dtype.type(nodata).item()
            profile = src.profile.copy()
            profile.update(nodata=nodata_value)
            descriptions = src.descriptions

            with rasterio.open(temp_path, "w", **profile) as dst:
                total_masked = 0
                for band_index in range(1, src.count + 1):
                    data = src.read(band_index)
                    invalid = np.zeros(data.shape, dtype=bool)
                    if src.nodata is not None:
                        invalid |= data == src.nodata
                    if valid_min is not None:
                        invalid |= data < valid_min
                    if valid_max is not None:
                        invalid |= data > valid_max

                    total_masked += int(invalid.sum())
                    data = data.copy()
                    data[invalid] = nodata_value
                    dst.write(data, band_index)

                    description = descriptions[band_index - 1]
                    if description:
                        dst.set_band_description(band_index, description)

        if input_path.resolve() == output_path.resolve():
            os.replace(temp_path, output_path)
        else:
            if output_path.exists() and overwrite:
                output_path.unlink()
            os.replace(temp_path, output_path)
        print(f"Masked {total_masked:,} cells outside valid range in {output_path}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    args = parse_args()
    mask_raster_values(
        input_path=Path(args.input),
        output_path=Path(args.output),
        valid_min=args.valid_min,
        valid_max=args.valid_max,
        nodata_arg=args.nodata,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
