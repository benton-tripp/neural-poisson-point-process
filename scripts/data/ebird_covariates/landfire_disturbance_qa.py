"""Validation and mapped QA for annual LANDFIRE disturbance fractions."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS

from .landfire_disturbance_derive import expected_disturbance_band_ids


BAND_PATTERN = re.compile(
    r"^availability__landfire__annual_disturbance_fraction__mean__"
    r"r(?P<radius>\d+)__y(?P<year>\d{4})$"
)


def parse_disturbance_band_id(identifier: str) -> dict[str, int]:
    match = BAND_PATTERN.match(identifier)
    if match is None:
        raise ValueError(f"Unrecognized disturbance band ID: {identifier}")
    return {
        "radius_m": int(match.group("radius")),
        "year": int(match.group("year")),
    }


def _load_inventories(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path_value in summary.get("inventory_paths", []):
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Disturbance inventory does not exist: {path}")
        inventory = json.loads(path.read_text(encoding="utf-8"))
        identifier = inventory["band_id"]
        if identifier in result:
            raise ValueError(f"Duplicate disturbance inventory: {identifier}")
        result[identifier] = inventory
    return result


def _tile_path(inventory: dict[str, Any], tile_id: str) -> Path | None:
    paths = [
        Path(record["path"])
        for record in inventory.get("tiles", [])
        if record.get("tile_id") == tile_id
    ]
    if len(paths) > 1:
        raise ValueError(f"Duplicate tile records for {inventory['band_id']}.")
    return paths[0] if paths else None


def validate_landfire_disturbance_derivation(
    plan: dict[str, Any],
    summary: dict[str, Any],
    *,
    range_tolerance: float = 1e-5,
) -> dict[str, Any]:
    inventories = _load_inventories(summary)
    years = [int(value) for value in summary["disturbance_years"]]
    radii = [int(value) for value in summary["neighborhoods_m"]]
    expected_ids = expected_disturbance_band_ids(years, radii)
    tile_id = str(summary["tile_id"])
    issues: list[str] = []
    if list(inventories) != expected_ids:
        issues.append("Disturbance inventory IDs or ordering differ from schema.")
    if int(summary["band_count"]) != len(expected_ids):
        issues.append("Disturbance summary band count differs from schema.")

    grid = plan["grid"]
    tile_contracts = {tile["tile_id"]: tile for tile in grid["tiles"]}
    tile_contract = tile_contracts.get(tile_id)
    if tile_contract is None:
        issues.append(f"Disturbance tile {tile_id} is absent from build plan.")
    target_crs = CRS.from_user_input(grid["crs"])
    tile_size = int(grid["tile_width_cells"])
    resolution = float(grid["resolution_m"])
    expected_mask_rule = str(grid.get("aoi_mask_rule", "center"))
    active_key = (
        "active_cells_all_touched_rule"
        if expected_mask_rule == "all_touched"
        else "active_cells_center_rule"
    )
    active_aoi_cells = (
        int(tile_contract.get(active_key, tile_size * tile_size))
        if tile_contract is not None
        else 0
    )

    range_records: list[dict[str, Any]] = []
    empty_ids: list[str] = []
    cog_count = 0
    total_bytes = 0
    for identifier, inventory in inventories.items():
        try:
            parsed = parse_disturbance_band_id(identifier)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if parsed["year"] not in years or parsed["radius_m"] not in radii:
            issues.append(f"Disturbance band axis differs from summary: {identifier}")
        path = _tile_path(inventory, tile_id)
        if path is None:
            empty_ids.append(identifier)
            continue
        if not path.exists():
            issues.append(f"Disturbance COG does not exist: {path}")
            continue
        total_bytes += path.stat().st_size
        with rasterio.open(path) as dataset:
            cog_count += 1
            if dataset.driver != "GTiff" or not dataset.profile.get("tiled", False):
                issues.append(f"Disturbance file is not a tiled GeoTIFF: {path}")
            if dataset.crs != target_crs:
                issues.append(f"Disturbance file has wrong CRS: {path}")
            if (dataset.width, dataset.height) != (tile_size, tile_size):
                issues.append(f"Disturbance file has wrong dimensions: {path}")
            if not np.allclose(dataset.res, (resolution, resolution), atol=1e-9):
                issues.append(f"Disturbance file has wrong resolution: {path}")
            if dataset.count != 1 or dataset.dtypes[0] != "float32":
                issues.append(f"Disturbance file has wrong band contract: {path}")
            if dataset.descriptions[0] != identifier:
                issues.append(f"Disturbance description differs: {path}")
            if dataset.tags().get("aoi_mask_rule", "center") != expected_mask_rule:
                issues.append(f"Disturbance file has wrong AOI mask rule: {path}")
            if tile_contract is not None:
                actual_bounds = [
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                ]
                if not np.allclose(
                    actual_bounds,
                    tile_contract["bounds_m"],
                    rtol=0.0,
                    atol=1e-6,
                ):
                    issues.append(f"Disturbance file has wrong bounds: {path}")
            values = dataset.read(1, masked=True).compressed().astype(np.float64)
        if values.size == 0:
            issues.append(f"Physical disturbance COG has no valid cells: {path}")
            continue
        minimum = float(values.min())
        maximum = float(values.max())
        if minimum < -range_tolerance or maximum > 1 + range_tolerance:
            issues.append(
                f"Disturbance values outside [0, 1] for {identifier}: "
                f"[{minimum}, {maximum}]."
            )
        range_records.append(
            {
                "band_id": identifier,
                **parsed,
                "active_aoi_cells": active_aoi_cells,
                "valid_cells": int(values.size),
                "supported_aoi_fraction": (
                    float(values.size) / active_aoi_cells
                    if active_aoi_cells > 0
                    else None
                ),
                "minimum": minimum,
                "maximum": maximum,
                "mean": float(values.mean()),
                "nonzero_cells": int((values > 0).sum()),
            }
        )

    if cog_count != int(summary["derived_cog_count"]):
        issues.append("Disturbance summary COG count differs from validated files.")
    if empty_ids != list(summary.get("empty_band_ids", [])):
        issues.append("Disturbance empty-band inventory differs from summary.")

    logical_vrt = summary.get("logical_vrt")
    if logical_vrt:
        path = Path(logical_vrt)
        if not path.exists():
            issues.append(f"Disturbance logical VRT does not exist: {path}")
        else:
            with rasterio.open(path) as dataset:
                if dataset.count != len(expected_ids):
                    issues.append("Disturbance VRT has wrong band count.")
                if list(dataset.descriptions) != expected_ids:
                    issues.append("Disturbance VRT descriptions are out of order.")
                if dataset.crs != target_crs:
                    issues.append("Disturbance VRT has wrong CRS.")
                if (dataset.width, dataset.height) != (
                    int(grid["bounding_width_cells"]),
                    int(grid["bounding_height_cells"]),
                ):
                    issues.append("Disturbance VRT has wrong dimensions.")

    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": summary["build_id"],
        "source_id": "landfire",
        "component": "annual_disturbance",
        "tile_id": tile_id,
        "disturbance_years": years,
        "neighborhoods_m": radii,
        "band_count": len(inventories),
        "derived_cog_count": cog_count,
        "derived_cog_bytes": total_bytes,
        "derived_cog_mib": total_bytes / (1024**2),
        "empty_band_count": len(empty_ids),
        "empty_band_ids": empty_ids,
        "range_tolerance": range_tolerance,
        "band_statistics": range_records,
        "minimum_supported_aoi_fraction": min(
            (
                record["supported_aoi_fraction"]
                for record in range_records
                if record["supported_aoi_fraction"] is not None
            ),
            default=None,
        ),
        "maximum_disturbance_fraction": max(
            (record["maximum"] for record in range_records),
            default=None,
        ),
        "issues": issues,
        "all_checks_passed": not issues,
    }


def write_landfire_disturbance_validation(
    validation: dict[str, Any], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def plot_landfire_disturbance_preview(
    plan: dict[str, Any],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inventories = _load_inventories(summary)
    tile_id = str(summary["tile_id"])
    years = [int(value) for value in summary["disturbance_years"]]
    radii = [int(value) for value in summary["neighborhoods_m"]]
    radius = 1000 if 1000 in radii else min(radii)
    arrays: list[np.ma.MaskedArray] = []
    reference: dict[str, Any] | None = None
    for year in years:
        identifier = (
            "availability__landfire__annual_disturbance_fraction__mean__"
            f"r{radius}__y{year}"
        )
        path = _tile_path(inventories[identifier], tile_id)
        if path is None:
            raise ValueError(f"Disturbance preview band is all NoData: {identifier}")
        with rasterio.open(path) as dataset:
            arrays.append(dataset.read(1, masked=True))
            if reference is None:
                reference = {"bounds": dataset.bounds, "crs": dataset.crs}
    assert reference is not None
    positives = np.concatenate(
        [array.compressed()[array.compressed() > 0] for array in arrays]
    )
    vmax = (
        min(1.0, max(0.01, float(np.quantile(positives, 0.99))))
        if positives.size
        else 1.0
    )
    boundary = gpd.GeoSeries(
        [gpd.read_file(plan["aoi"]["path"]).to_crs(reference["crs"]).geometry.union_all()],
        crs=reference["crs"],
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    image = None
    extent = (
        reference["bounds"].left,
        reference["bounds"].right,
        reference["bounds"].bottom,
        reference["bounds"].top,
    )
    for axis, year, array in zip(axes.flat, years, arrays, strict=True):
        image = axis.imshow(
            array,
            extent=extent,
            origin="upper",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
        )
        boundary.boundary.plot(ax=axis, color="black", linewidth=0.8)
        axis.set_title(str(year))
        axis.set_axis_off()
    if image is not None:
        figure.colorbar(
            image,
            ax=list(axes.flat),
            shrink=0.72,
            label=f"Disturbed fraction within {radius / 1000:g} km",
        )
    figure.suptitle(
        f"LANDFIRE annual disturbance: {tile_id}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
