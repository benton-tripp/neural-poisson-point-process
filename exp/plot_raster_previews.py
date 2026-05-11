"""
Create quick-look PNG previews for raster covariates.

The script reads large rasters at reduced resolution and saves simple images
for canopy, elevation, distance to waterbody, and distance to coastline.

Run from the project root:

    python exp/plot_raster_previews.py
    python exp/plot_raster_previews.py --rasters hydro coastline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling


DEFAULT_CANOPY = "data/nc_tcc_2020_2023.tif"
DEFAULT_ELEVATION = "data/nc_usgs30m.tif"
DEFAULT_HYDRO = "data/nc_hydro_distance_100m.tif"
DEFAULT_OUTPUT_DIR = "images/raster_previews"
RASTER_CHOICES = ("canopy", "elevation", "hydro", "coastline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot quick raster previews.")
    parser.add_argument("--canopy", default=DEFAULT_CANOPY, help="Canopy GeoTIFF path.")
    parser.add_argument("--elevation", default=DEFAULT_ELEVATION, help="Elevation GeoTIFF path.")
    parser.add_argument(
        "--hydro",
        default=DEFAULT_HYDRO,
        help="Two-band hydro distance GeoTIFF path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1200,
        help="Maximum plotted width or height in pixels. Defaults to 1200.",
    )
    parser.add_argument(
        "--canopy-band",
        type=int,
        default=1,
        help="Band to preview from the canopy stack. Defaults to 1.",
    )
    parser.add_argument(
        "--rasters",
        nargs="+",
        choices=RASTER_CHOICES,
        default=list(RASTER_CHOICES),
        help="Raster previews to create. Defaults to all.",
    )
    return parser.parse_args()


def preview_shape(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if max_dim <= 0:
        raise ValueError("--max-dim must be greater than 0.")

    scale = min(1.0, max_dim / max(width, height))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def read_preview(path: Path, band: int, max_dim: int) -> tuple[np.ndarray, str | None]:
    with rasterio.open(path) as src:
        out_shape = preview_shape(src.width, src.height, max_dim)
        data = src.read(
            band,
            out_shape=out_shape,
            resampling=Resampling.nearest,
            masked=True,
        )
        description = src.descriptions[band - 1] if src.descriptions else None
        nodata = src.nodata

    if np.ma.is_masked(data):
        array = data.astype("float32").filled(np.nan)
    else:
        array = np.asarray(data, dtype="float32")

    if nodata is not None:
        array[array == nodata] = np.nan

    return array, description


def mask_canopy_codes(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result[(result == 254) | (result == 255)] = np.nan
    return result


def plot_raster(
    array: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    label: str,
    percentile_clip: tuple[float, float] | None = (2, 98),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"No finite values to plot for {title}.")

    vmin = vmax = None
    if percentile_clip is not None:
        vmin, vmax = np.nanpercentile(array, percentile_clip)
        if vmin == vmax:
            vmin = vmax = None

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    selected = set(args.rasters)

    if "canopy" in selected:
        canopy, canopy_description = read_preview(Path(args.canopy), args.canopy_band, args.max_dim)
        canopy = mask_canopy_codes(canopy)
        canopy_title = "Tree Canopy Cover"
        if canopy_description:
            canopy_title = f"{canopy_title} ({canopy_description})"
        plot_raster(
            canopy,
            canopy_title,
            output_dir / "canopy_preview.png",
            cmap="Greens",
            label="Percent canopy cover",
            percentile_clip=(0, 100),
        )

    if "elevation" in selected:
        elevation, _ = read_preview(Path(args.elevation), 1, args.max_dim)
        plot_raster(
            elevation,
            "Elevation",
            output_dir / "elevation_preview.png",
            cmap="terrain",
            label="Elevation",
        )

    if "hydro" in selected:
        water_distance, _ = read_preview(Path(args.hydro), 1, args.max_dim)
        plot_raster(
            water_distance / 1000,
            "Distance to Nearest Waterbody",
            output_dir / "waterbody_distance_preview.png",
            cmap="Blues_r",
            label="Distance (km)",
        )

    if "coastline" in selected:
        coastline_distance, _ = read_preview(Path(args.hydro), 2, args.max_dim)
        plot_raster(
            coastline_distance / 1000,
            "Distance to Nearest Coastline",
            output_dir / "coastline_distance_preview.png",
            cmap="magma",
            label="Distance (km)",
        )

    print(f"Saved selected raster previews to {output_dir}")


if __name__ == "__main__":
    main()
