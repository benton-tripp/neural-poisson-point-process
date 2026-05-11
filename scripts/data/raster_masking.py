"""Shared raster masking helpers for data acquisition scripts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def default_nodata(dtype: str):
    if np.issubdtype(np.dtype(dtype), np.floating):
        return -9999.0
    if np.dtype(dtype) == np.dtype("uint8"):
        return 255
    if np.issubdtype(np.dtype(dtype), np.unsignedinteger):
        return np.iinfo(dtype).max
    return -9999


def mask_raster_with_boundary(
    raster_path: str | Path,
    boundary_path: str | Path,
    crop: bool = True,
) -> Path:
    raster_path = Path(raster_path)
    boundary_path = Path(boundary_path)

    with rasterio.open(raster_path) as src:
        boundary = gpd.read_file(boundary_path)
        if boundary.empty:
            raise ValueError(f"Boundary dataset has no features: {boundary_path}")

        boundary = boundary[boundary.geometry.notna()].copy()
        boundary = boundary.to_crs(src.crs)
        geometries = [geom for geom in boundary.geometry if geom is not None and not geom.is_empty]
        if not geometries:
            raise ValueError(f"Boundary dataset has no valid geometries: {boundary_path}")

        nodata = src.nodata
        if nodata is None:
            nodata = default_nodata(src.dtypes[0])

        masked_data, masked_transform = mask(
            src,
            geometries,
            crop=crop,
            filled=True,
            nodata=nodata,
        )
        profile = src.profile.copy()
        profile.update(
            {
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "transform": masked_transform,
                "nodata": nodata,
            }
        )
        descriptions = src.descriptions

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f"{raster_path.stem}_masked_",
        suffix=raster_path.suffix,
        dir=raster_path.parent,
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(masked_data)
            for band_index, description in enumerate(descriptions, start=1):
                if description:
                    dst.set_band_description(band_index, description)

        os.replace(temp_path, raster_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return raster_path
