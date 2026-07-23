"""Fixed-grid raster normalization and logical VRT assembly."""

from __future__ import annotations

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject


RESAMPLING_METHODS = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
}

VRT_DTYPES = {
    "uint8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "float32": "Float32",
    "float64": "Float64",
}

AOI_MASK_RULES = {"center", "all_touched"}


def safe_band_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    if not slug:
        raise ValueError("Band id must contain at least one filename-safe character.")
    return slug


def safe_artifact_path(
    directory: Path,
    value: str,
    suffix: str,
    *,
    maximum_absolute_length: int = 259,
) -> Path:
    """Keep readable names unless a resolved path would cross Windows limits."""
    slug = safe_band_slug(value)
    candidate = directory / f"{slug}{suffix}"
    if len(str(candidate.resolve())) < maximum_absolute_length:
        return candidate
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    directory_length = len(str(directory.resolve()))
    target_length = maximum_absolute_length - 12
    slug_budget = target_length - directory_length - 1 - len(suffix)
    prefix_length = slug_budget - len(digest) - 2
    if prefix_length < 8:
        raise ValueError(
            f"Output directory is too long for a portable artifact path: {directory}"
        )
    shortened = f"{slug[:prefix_length]}__{digest}"
    return directory / f"{shortened}{suffix}"


def load_plan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Build plan does not exist: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid build plan JSON in {path}: {exc}") from exc
    if plan.get("schema_version") != 1:
        raise ValueError("Build plan must use schema_version 1.")
    grid = plan.get("grid", {})
    required = {
        "crs",
        "resolution_m",
        "tile_size_m",
        "snapped_bounds_m",
        "bounding_width_cells",
        "bounding_height_cells",
        "tiles",
    }
    missing = sorted(required - set(grid))
    if missing:
        raise ValueError(f"Build plan grid is missing: {', '.join(missing)}")
    aoi_mask_rule = grid.get("aoi_mask_rule", "center")
    if aoi_mask_rule not in AOI_MASK_RULES:
        raise ValueError(
            "Build plan grid aoi_mask_rule must be 'center' or 'all_touched'."
        )
    return plan


def aoi_mask_all_touched(plan: dict[str, Any]) -> bool:
    rule = plan.get("grid", {}).get("aoi_mask_rule", "center")
    if rule not in AOI_MASK_RULES:
        raise ValueError("AOI mask rule must be 'center' or 'all_touched'.")
    return rule == "all_touched"


def load_plan_aoi(plan: dict[str, Any]) -> Any:
    aoi = plan.get("aoi", {})
    path = Path(aoi.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(f"Build-plan AOI does not exist: {path}")
    frame = gpd.read_file(path, layer=aoi.get("layer"))
    if frame.empty or frame.crs is None:
        raise ValueError(f"Build-plan AOI is empty or has no CRS: {path}")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame = frame.to_crs(plan["grid"]["crs"])
    geometry = frame.geometry.make_valid().union_all()
    if geometry.is_empty:
        raise ValueError(f"Build-plan AOI has no valid geometry: {path}")
    return geometry


def tile_transform(tile: dict[str, Any], width: int, height: int) -> Any:
    return from_bounds(*tile["bounds_m"], width=width, height=height)


def write_cog(
    output_path: Path,
    values: np.ndarray,
    transform: Any,
    crs: str,
    nodata: float,
    description: str,
    tags: dict[str, str],
    overview_resampling: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        output_path.parent
        / (
            "."
            + hashlib.sha256(str(output_path).encode("utf-8")).hexdigest()[:16]
            + ".tmp.tif"
        )
    )
    temporary_overview = Path(f"{temporary}.ovr.tmp")
    if temporary.exists():
        temporary.unlink()
    if temporary_overview.exists():
        temporary_overview.unlink()
    with rasterio.open(
        temporary,
        "w",
        driver="COG",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="DEFLATE",
        blocksize=256,
        overview_resampling=overview_resampling,
        BIGTIFF="IF_SAFER",
    ) as destination:
        destination.write(values, 1)
        destination.set_band_description(1, description)
        destination.update_tags(**tags)
    temporary.replace(output_path)
    if temporary_overview.exists():
        temporary_overview.unlink()


def normalize_raster_to_tiles(
    plan: dict[str, Any],
    source_path: Path,
    output_dir: Path,
    band_id: str,
    source_band: int = 1,
    resampling: str = "bilinear",
    mask_outside_aoi: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Warp one source band to every intersecting fixed-grid tile."""
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unsupported resampling {resampling!r}; choose from {sorted(RESAMPLING_METHODS)}."
        )
    if not source_path.exists():
        raise FileNotFoundError(f"Source raster does not exist: {source_path}")

    grid = plan["grid"]
    resolution = float(grid["resolution_m"])
    tile_size = float(grid["tile_size_m"])
    tile_cells = int(round(tile_size / resolution))
    target_crs = grid["crs"]
    nodata = -9999.0
    aoi_geometry = load_plan_aoi(plan) if mask_outside_aoi else None
    aoi_mask_rule = (
        "all_touched" if aoi_mask_all_touched(plan) else "center"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_outputs: list[dict[str, Any]] = []
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError(f"Source raster has no CRS: {source_path}")
        if source_band < 1 or source_band > source.count:
            raise ValueError(
                f"Source band {source_band} is outside 1-{source.count}: {source_path}"
            )
        source_nodata = source.nodata
        for tile in grid["tiles"]:
            output_path = safe_artifact_path(
                output_dir,
                f"{band_id}__{tile['tile_id']}",
                ".tif",
            )
            if output_path.exists() and not overwrite:
                with rasterio.open(output_path) as existing:
                    if mask_outside_aoi:
                        existing_mask_rule = existing.tags().get(
                            "aoi_mask_rule", "center"
                        )
                        if existing_mask_rule != aoi_mask_rule:
                            raise ValueError(
                                f"Existing tile {output_path} uses AOI mask rule "
                                f"{existing_mask_rule!r}, but the build plan requires "
                                f"{aoi_mask_rule!r}. Rerun with --overwrite."
                            )
                    valid = existing.read_masks(1) > 0
                    valid_cells = int(valid.sum())
                tile_outputs.append(
                    {
                        "tile_id": tile["tile_id"],
                        "path": str(output_path.resolve()),
                        "valid_cells": valid_cells,
                        "coverage_fraction": valid_cells / (tile_cells * tile_cells),
                        "reused": True,
                    }
                )
                continue

            transform = tile_transform(tile, tile_cells, tile_cells)
            values = np.full((tile_cells, tile_cells), nodata, dtype=np.float32)
            reproject(
                source=rasterio.band(source, source_band),
                destination=values,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source_nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=nodata,
                resampling=RESAMPLING_METHODS[resampling],
                num_threads=2,
                init_dest_nodata=True,
            )

            if aoi_geometry is not None:
                inside = geometry_mask(
                    [aoi_geometry],
                    out_shape=values.shape,
                    transform=transform,
                    invert=True,
                    all_touched=aoi_mask_all_touched(plan),
                )
                values[~inside] = nodata
            valid = np.isfinite(values) & (values != nodata)
            valid_cells = int(valid.sum())
            if valid_cells == 0:
                continue

            write_cog(
                output_path=output_path,
                values=values,
                transform=transform,
                crs=target_crs,
                nodata=nodata,
                description=band_id,
                tags={
                    "band_id": band_id,
                    "source_path": str(source_path.resolve()),
                    "source_band": str(source_band),
                    "resampling": resampling,
                    "tile_id": tile["tile_id"],
                    "grid_build_id": plan["build_id"],
                    "aoi_mask_rule": aoi_mask_rule if mask_outside_aoi else "none",
                },
                overview_resampling="nearest" if resampling in {"nearest", "mode"} else "average",
            )
            tile_outputs.append(
                {
                    "tile_id": tile["tile_id"],
                    "path": str(output_path.resolve()),
                    "valid_cells": valid_cells,
                    "coverage_fraction": valid_cells / (tile_cells * tile_cells),
                    "reused": False,
                }
            )

    if not tile_outputs:
        raise ValueError(f"Source raster produced no valid cells in the build AOI: {source_path}")
    return {
        "schema_version": 1,
        "build_id": plan["build_id"],
        "band_id": band_id,
        "dtype": "float32",
        "nodata": nodata,
        "source_path": str(source_path.resolve()),
        "source_band": source_band,
        "resampling": resampling,
        "aoi_mask_rule": aoi_mask_rule if mask_outside_aoi else "none",
        "tile_count": len(tile_outputs),
        "valid_cells": sum(tile["valid_cells"] for tile in tile_outputs),
        "tiles": tile_outputs,
    }


def write_band_inventory(inventory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_band_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Band inventory does not exist: {path}")
    inventory = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "build_id", "band_id", "dtype", "nodata", "tiles"}
    missing = sorted(required - set(inventory))
    if missing:
        raise ValueError(f"Band inventory {path} is missing: {', '.join(missing)}")
    return inventory


def vrt_data_type(dtype: str) -> str:
    try:
        return VRT_DTYPES[np.dtype(dtype).name]
    except KeyError as exc:
        raise ValueError(f"Unsupported VRT dtype: {dtype}") from exc


def clipped_rectangles(
    tile_bounds: list[float],
    build_bounds: list[float],
    resolution: float,
) -> tuple[dict[str, int], dict[str, int]] | None:
    tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_bounds
    build_min_x, build_min_y, build_max_x, build_max_y = build_bounds
    min_x = max(tile_min_x, build_min_x)
    min_y = max(tile_min_y, build_min_y)
    max_x = min(tile_max_x, build_max_x)
    max_y = min(tile_max_y, build_max_y)
    if min_x >= max_x or min_y >= max_y:
        return None
    width = int(round((max_x - min_x) / resolution))
    height = int(round((max_y - min_y) / resolution))
    source_rect = {
        "xOff": int(round((min_x - tile_min_x) / resolution)),
        "yOff": int(round((tile_max_y - max_y) / resolution)),
        "xSize": width,
        "ySize": height,
    }
    destination_rect = {
        "xOff": int(round((min_x - build_min_x) / resolution)),
        "yOff": int(round((build_max_y - max_y) / resolution)),
        "xSize": width,
        "ySize": height,
    }
    return source_rect, destination_rect


def write_logical_vrt(
    plan: dict[str, Any],
    inventories: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not inventories:
        raise ValueError("At least one band inventory is required to create a VRT.")
    grid = plan["grid"]
    build_bounds = [float(value) for value in grid["snapped_bounds_m"]]
    width = int(grid["bounding_width_cells"])
    height = int(grid["bounding_height_cells"])
    resolution = float(grid["resolution_m"])
    crs = CRS.from_user_input(grid["crs"])
    tile_by_id = {tile["tile_id"]: tile for tile in grid["tiles"]}

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(width),
        rasterYSize=str(height),
    )
    ET.SubElement(root, "SRS", dataAxisToSRSAxisMapping="1,2").text = crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = (
        f"{build_bounds[0]}, {resolution}, 0.0, {build_bounds[3]}, 0.0, {-resolution}"
    )
    metadata = ET.SubElement(root, "Metadata")
    ET.SubElement(metadata, "MDI", key="build_id").text = str(plan["build_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output_path.resolve().parent
    for band_index, inventory in enumerate(inventories, start=1):
        if inventory["build_id"] != plan["build_id"]:
            raise ValueError(
                f"Band {inventory['band_id']} belongs to build {inventory['build_id']}, "
                f"not {plan['build_id']}."
            )
        vrt_band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType=vrt_data_type(inventory["dtype"]),
            band=str(band_index),
        )
        ET.SubElement(vrt_band, "Description").text = inventory["band_id"]
        ET.SubElement(vrt_band, "NoDataValue").text = str(inventory["nodata"])
        band_metadata = ET.SubElement(vrt_band, "Metadata")
        ET.SubElement(band_metadata, "MDI", key="band_id").text = inventory["band_id"]

        for tile_output in inventory["tiles"]:
            tile = tile_by_id.get(tile_output["tile_id"])
            if tile is None:
                raise ValueError(
                    f"Inventory tile {tile_output['tile_id']} is absent from build plan."
                )
            rectangles = clipped_rectangles(
                tile["bounds_m"], build_bounds, resolution
            )
            if rectangles is None:
                continue
            source_rect, destination_rect = rectangles
            tile_path = Path(tile_output["path"])
            if not tile_path.exists():
                raise FileNotFoundError(f"VRT source tile does not exist: {tile_path}")
            with rasterio.open(tile_path) as source:
                block_height, block_width = source.block_shapes[0]
                source_width = source.width
                source_height = source.height
                source_dtype = vrt_data_type(source.dtypes[0])
            simple_source = ET.SubElement(vrt_band, "SimpleSource")
            relative_path = Path(os.path.relpath(tile_path.resolve(), output_parent)).as_posix()
            ET.SubElement(
                simple_source,
                "SourceFilename",
                relativeToVRT="1",
            ).text = relative_path
            ET.SubElement(simple_source, "SourceBand").text = "1"
            ET.SubElement(
                simple_source,
                "SourceProperties",
                RasterXSize=str(source_width),
                RasterYSize=str(source_height),
                DataType=source_dtype,
                BlockXSize=str(block_width),
                BlockYSize=str(block_height),
            )
            ET.SubElement(
                simple_source,
                "SrcRect",
                **{key: str(value) for key, value in source_rect.items()},
            )
            ET.SubElement(
                simple_source,
                "DstRect",
                **{key: str(value) for key, value in destination_rect.items()},
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    tree.write(temporary, encoding="UTF-8", xml_declaration=True)
    temporary.replace(output_path)
    return output_path
