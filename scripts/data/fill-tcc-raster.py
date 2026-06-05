"""
Fill USFS TCC nodata values using hydrography and local canopy context.

The intended workflow is:

1. Build or update the hydro distance raster on the TCC grid.
2. Download/mask TCC so valid canopy is 0-100 and mask/background is nodata.
3. Fill TCC nodata:
   - cells within a water/coast distance threshold become 0 percent canopy
   - remaining nodata cells are filled from local valid canopy means
   - any stubborn cells fall back to nearest valid canopy

Examples:

    python scripts/data/fill-tcc-raster.py --tcc data/nc_tcc_2020_2023.tif --hydro data/nc_hydro_distance_match_tcc.tif --output data/nc_tcc_2020_2023_filled.tif --water-distance-threshold 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


DEFAULT_WATER_DISTANCE_THRESHOLD = 30.0
DEFAULT_MAX_MEAN_RADIUS = 32
DEFAULT_NODATA = 255
VALID_MIN = 0
VALID_MAX = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill TCC nodata using hydrography and local canopy means."
    )
    parser.add_argument("--tcc", required=True, help="Input multi-band TCC raster.")
    parser.add_argument(
        "--hydro",
        required=True,
        help="Hydro distance raster with waterbody and coastline distance bands on the TCC grid.",
    )
    parser.add_argument("--output", required=True, help="Output filled TCC raster.")
    parser.add_argument(
        "--water-distance-threshold",
        type=float,
        default=DEFAULT_WATER_DISTANCE_THRESHOLD,
        help=(
            "Cells with missing TCC and waterbody or coastline distance at or below "
            f"this threshold are filled as 0 canopy. Defaults to {DEFAULT_WATER_DISTANCE_THRESHOLD:g} m."
        ),
    )
    parser.add_argument(
        "--max-mean-radius",
        type=int,
        default=DEFAULT_MAX_MEAN_RADIUS,
        help=(
            "Maximum square-window radius, in cells, for local mean filling. "
            f"Defaults to {DEFAULT_MAX_MEAN_RADIUS}."
        ),
    )
    parser.add_argument(
        "--nodata",
        type=int,
        default=DEFAULT_NODATA,
        help=f"Output nodata value. Defaults to {DEFAULT_NODATA}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    return parser.parse_args()


def check_grid_match(tcc: rasterio.DatasetReader, hydro: rasterio.DatasetReader) -> None:
    if tcc.crs != hydro.crs:
        raise ValueError(f"TCC and hydro CRS differ: {tcc.crs} != {hydro.crs}")
    if tcc.transform != hydro.transform:
        raise ValueError("TCC and hydro transforms differ. Rebuild hydro on the TCC template.")
    if tcc.width != hydro.width or tcc.height != hydro.height:
        raise ValueError("TCC and hydro dimensions differ. Rebuild hydro on the TCC template.")
    if hydro.count < 2:
        raise ValueError("Hydro raster must have waterbody and coastline distance bands.")


def valid_tcc(data: np.ndarray, nodata: float | int | None) -> np.ndarray:
    valid = (data >= VALID_MIN) & (data <= VALID_MAX)
    if nodata is not None:
        valid &= data != nodata
    return valid


def local_mean_fill(values: np.ndarray, missing: np.ndarray, max_radius: int) -> tuple[np.ndarray, np.ndarray]:
    filled = values.copy()
    remaining = missing.copy()
    valid = ~remaining & np.isfinite(filled)

    radius = 1
    while remaining.any() and radius <= max_radius:
        size = 2 * radius + 1
        value_sum = ndimage.uniform_filter(
            np.where(valid, filled, 0.0),
            size=size,
            mode="constant",
            cval=0.0,
        )
        valid_count = ndimage.uniform_filter(
            valid.astype(np.float32),
            size=size,
            mode="constant",
            cval=0.0,
        )
        can_fill = remaining & (valid_count > 0)
        if can_fill.any():
            filled[can_fill] = value_sum[can_fill] / valid_count[can_fill]
            remaining[can_fill] = False
            valid[can_fill] = True
        radius *= 2

    return filled, remaining


def nearest_fill(values: np.ndarray, missing: np.ndarray) -> np.ndarray:
    if not missing.any():
        return values
    valid = ~missing & np.isfinite(values)
    if not valid.any():
        raise ValueError("Cannot nearest-fill TCC band because it has no valid cells.")

    _, nearest_indices = ndimage.distance_transform_edt(
        ~valid,
        return_distances=True,
        return_indices=True,
    )
    filled = values.copy()
    filled[missing] = values[nearest_indices[0][missing], nearest_indices[1][missing]]
    return filled


def fill_tcc_band(
    data: np.ndarray,
    source_nodata: float | int | None,
    water_or_coast: np.ndarray,
    max_mean_radius: int,
) -> tuple[np.ndarray, dict[str, int]]:
    valid = valid_tcc(data, source_nodata)
    filled = data.astype(np.float32)
    filled[~valid] = np.nan

    missing = ~valid
    water_fill = missing & water_or_coast
    filled[water_fill] = 0.0
    missing[water_fill] = False

    filled, remaining = local_mean_fill(filled, missing, max_mean_radius)
    mean_filled_count = int(missing.sum() - remaining.sum())
    filled = nearest_fill(filled, remaining)

    filled = np.clip(np.rint(filled), VALID_MIN, VALID_MAX).astype(np.uint8)
    return filled, {
        "initial_missing": int((~valid).sum()),
        "water_zero_filled": int(water_fill.sum()),
        "local_mean_filled": mean_filled_count,
        "nearest_filled": int(remaining.sum()),
    }


def fill_tcc_raster(
    tcc_path: Path,
    hydro_path: Path,
    output_path: Path,
    water_distance_threshold: float,
    max_mean_radius: int,
    output_nodata: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists. Use --overwrite to replace: {output_path}")
    if max_mean_radius < 1:
        raise ValueError("--max-mean-radius must be at least 1.")
    if water_distance_threshold < 0:
        raise ValueError("--water-distance-threshold must be nonnegative.")

    with rasterio.open(tcc_path) as tcc, rasterio.open(hydro_path) as hydro:
        check_grid_match(tcc, hydro)
        water_distance = hydro.read(1, masked=True).astype(np.float32).filled(np.nan)
        coast_distance = hydro.read(2, masked=True).astype(np.float32).filled(np.nan)
        water_or_coast = (
            (np.isfinite(water_distance) & (water_distance <= water_distance_threshold))
            | (np.isfinite(coast_distance) & (coast_distance <= water_distance_threshold))
        )

        profile = tcc.profile.copy()
        profile.update(dtype="uint8", nodata=output_nodata, compress="lzw", tiled=True, BIGTIFF="IF_SAFER")
        descriptions = tcc.descriptions
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **profile) as dst:
            for band_index in range(1, tcc.count + 1):
                filled, stats = fill_tcc_band(
                    tcc.read(band_index),
                    source_nodata=tcc.nodatavals[band_index - 1],
                    water_or_coast=water_or_coast,
                    max_mean_radius=max_mean_radius,
                )
                dst.write(filled, band_index)
                description = descriptions[band_index - 1]
                if description:
                    dst.set_band_description(band_index, description)
                print(
                    f"Band {band_index}: initial_missing={stats['initial_missing']:,}; "
                    f"water_zero_filled={stats['water_zero_filled']:,}; "
                    f"local_mean_filled={stats['local_mean_filled']:,}; "
                    f"nearest_filled={stats['nearest_filled']:,}"
                )

    print(f"Wrote filled TCC raster to {output_path}")


def main() -> None:
    args = parse_args()
    fill_tcc_raster(
        tcc_path=Path(args.tcc),
        hydro_path=Path(args.hydro),
        output_path=Path(args.output),
        water_distance_threshold=args.water_distance_threshold,
        max_mean_radius=args.max_mean_radius,
        output_nodata=args.nodata,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
