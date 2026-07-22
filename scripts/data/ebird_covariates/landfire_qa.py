"""Validation and visual QA for derived LANDFIRE vegetation covariates."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS

from .landfire_crosswalk import MODEL_CLASSES
from .landfire_derive import expected_band_ids


BAND_PATTERN = re.compile(
    r"^availability__landfire__(?P<variable>.+)__"
    r"(?P<statistic>mean|value)__r(?P<radius>\d+)__"
    r"(?P<release>lf\d{4})$"
)

EVT_COLORS = {
    "forest_tree": "#2f6b3b",
    "shrub": "#8a9a5b",
    "herbaceous": "#d7c95b",
    "riparian": "#57a8a8",
    "agriculture": "#d79b46",
    "developed": "#c9514b",
    "sparse_barren": "#a49b91",
    "open_water": "#3f77b5",
    "snow_ice": "#dcebf4",
}


def parse_band_id(identifier: str) -> dict[str, Any]:
    match = BAND_PATTERN.match(identifier)
    if match is None:
        raise ValueError(f"Unrecognized LANDFIRE band ID: {identifier}")
    parsed = match.groupdict()
    return {
        "variable": parsed["variable"],
        "statistic": parsed["statistic"],
        "radius_m": int(parsed["radius"]),
        "release": parsed["release"].upper(),
    }


def load_inventories(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventories: dict[str, dict[str, Any]] = {}
    for value in summary.get("inventory_paths", []):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"LANDFIRE inventory does not exist: {path}")
        inventory = json.loads(path.read_text(encoding="utf-8"))
        identifier = inventory["band_id"]
        if identifier in inventories:
            raise ValueError(f"Duplicate LANDFIRE inventory: {identifier}")
        inventories[identifier] = inventory
    return inventories


def tile_path(inventory: dict[str, Any], tile_id: str) -> Path | None:
    matches = [
        Path(tile["path"])
        for tile in inventory.get("tiles", [])
        if tile.get("tile_id") == tile_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Inventory {inventory['band_id']} has duplicate records for "
            f"tile {tile_id}."
        )
    return matches[0] if matches else None


def variable_range(variable: str, maximum_height_m: float) -> tuple[float, float]:
    if variable.endswith("_height_m_conditional"):
        return 0.0, maximum_height_m
    if (
        variable.startswith("evt_")
        or variable == "source_coverage_fraction"
        or variable.endswith("_cover_fraction_conditional")
    ):
        return 0.0, 1.0
    raise ValueError(f"No LANDFIRE range contract for variable: {variable}")


def validate_landfire_derivation(
    plan: dict[str, Any],
    summary: dict[str, Any],
    *,
    range_tolerance: float = 1e-5,
    fraction_sum_tolerance: float = 1e-4,
    maximum_height_m: float = 100.0,
) -> dict[str, Any]:
    inventories = load_inventories(summary)
    release = str(summary["release"])
    radii = [int(value) for value in summary["neighborhoods_m"]]
    expected_ids = expected_band_ids(release, radii)
    tile_id = str(summary["tile_id"])
    issues: list[str] = []

    inventory_ids = list(inventories)
    if inventory_ids != expected_ids:
        issues.append("LANDFIRE inventory IDs or ordering differ from the schema.")
    if int(summary["band_count"]) != len(expected_ids):
        issues.append(
            f"Summary band count {summary['band_count']} does not match "
            f"the {len(expected_ids)}-band schema."
        )

    grid = plan["grid"]
    tile_contracts = {tile["tile_id"]: tile for tile in grid["tiles"]}
    if tile_id not in tile_contracts:
        issues.append(f"Derived tile {tile_id} is absent from the build plan.")
        tile_contract = None
    else:
        tile_contract = tile_contracts[tile_id]
    target_crs = CRS.from_user_input(grid["crs"])
    tile_size = int(grid["tile_width_cells"])
    resolution = float(grid["resolution_m"])
    expected_mask_rule = str(grid.get("aoi_mask_rule", "center"))

    paths_by_band: dict[str, Path] = {}
    empty_ids: list[str] = []
    variable_accumulators: dict[str, dict[str, float]] = {}
    cog_count = 0
    total_bytes = 0
    for identifier, inventory in inventories.items():
        try:
            parsed = parse_band_id(identifier)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if parsed["release"] != release.upper():
            issues.append(f"Band release differs from summary: {identifier}")
        unexpected_tiles = {
            record.get("tile_id")
            for record in inventory.get("tiles", [])
            if record.get("tile_id") != tile_id
        }
        if unexpected_tiles:
            issues.append(f"Inventory {identifier} contains unexpected tiles.")
        try:
            path = tile_path(inventory, tile_id)
        except ValueError as exc:
            issues.append(str(exc))
            continue
        if path is None:
            empty_ids.append(identifier)
            continue
        paths_by_band[identifier] = path
        if not path.exists():
            issues.append(f"Derived COG does not exist: {path}")
            continue
        total_bytes += path.stat().st_size
        with rasterio.open(path) as dataset:
            cog_count += 1
            if dataset.driver != "GTiff" or not dataset.profile.get("tiled", False):
                issues.append(f"Derived file is not a tiled GeoTIFF: {path}")
            if dataset.crs != target_crs:
                issues.append(f"Derived file has the wrong CRS: {path}")
            if (dataset.width, dataset.height) != (tile_size, tile_size):
                issues.append(f"Derived file has the wrong dimensions: {path}")
            if tile_contract is not None:
                expected_bounds = tile_contract["bounds_m"]
                actual_bounds = [
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                ]
                if not np.allclose(
                    actual_bounds, expected_bounds, rtol=0.0, atol=1e-6
                ):
                    issues.append(f"Derived file has the wrong bounds: {path}")
            if not np.allclose(
                dataset.res,
                (resolution, resolution),
                rtol=0.0,
                atol=1e-9,
            ):
                issues.append(f"Derived file has the wrong resolution: {path}")
            if dataset.count != 1 or dataset.dtypes[0] != "float32":
                issues.append(f"Derived file has the wrong band contract: {path}")
            if dataset.descriptions[0] != identifier:
                issues.append(f"Derived file description does not match: {path}")
            actual_mask_rule = dataset.tags().get("aoi_mask_rule", "center")
            if actual_mask_rule != expected_mask_rule:
                issues.append(f"Derived file has the wrong AOI mask rule: {path}")
            values = dataset.read(1, masked=True).compressed().astype(np.float64)
        if values.size == 0:
            issues.append(f"Physical COG has no valid cells: {path}")
            continue
        expected_minimum, expected_maximum = variable_range(
            parsed["variable"], maximum_height_m
        )
        minimum = float(values.min())
        maximum = float(values.max())
        if (
            minimum < expected_minimum - range_tolerance
            or maximum > expected_maximum + range_tolerance
        ):
            issues.append(
                f"Values outside [{expected_minimum}, {expected_maximum}] for "
                f"{identifier}: [{minimum}, {maximum}]."
            )
        accumulator = variable_accumulators.setdefault(
            parsed["variable"],
            {
                "minimum": minimum,
                "maximum": maximum,
                "sum": 0.0,
                "valid_cells": 0.0,
                "files": 0.0,
            },
        )
        accumulator["minimum"] = min(accumulator["minimum"], minimum)
        accumulator["maximum"] = max(accumulator["maximum"], maximum)
        accumulator["sum"] += float(values.sum())
        accumulator["valid_cells"] += float(values.size)
        accumulator["files"] += 1.0

    if cog_count != int(summary["derived_cog_count"]):
        issues.append(
            f"Summary COG count {summary['derived_cog_count']} does not match "
            f"the {cog_count} validated files."
        )
    declared_empty_ids = list(summary.get("empty_band_ids", []))
    if empty_ids != declared_empty_ids:
        issues.append("Derived empty-band inventory differs from the summary.")
    if int(summary.get("empty_band_count", len(declared_empty_ids))) != len(empty_ids):
        issues.append("Derived empty-band count differs from the summary.")

    fraction_checks: list[dict[str, Any]] = []
    active_cell_key = (
        "active_cells_all_touched_rule"
        if expected_mask_rule == "all_touched"
        else "active_cells_center_rule"
    )
    active_aoi_cells = (
        int(
            tile_contract.get(
                active_cell_key,
                int(tile_contract.get("width", 0))
                * int(tile_contract.get("height", 0)),
            )
        )
        if tile_contract is not None
        else 0
    )
    for radius in radii:
        arrays: list[np.ma.MaskedArray] = []
        for model_class in MODEL_CLASSES:
            identifier = (
                f"availability__landfire__evt_{model_class}_fraction__mean__"
                f"r{radius}__{release.lower()}"
            )
            path = paths_by_band.get(identifier)
            if path is None or not path.exists():
                arrays = []
                break
            with rasterio.open(path) as dataset:
                arrays.append(dataset.read(1, masked=True))
        if not arrays:
            issues.append(f"Cannot calculate EVT class closure for r{radius}.")
            continue
        masks = [np.ma.getmaskarray(array) for array in arrays]
        if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
            issues.append(f"EVT class-fraction masks differ for r{radius}.")
        common = ~np.logical_or.reduce(masks)
        total = np.sum(
            np.stack([array.filled(0.0) for array in arrays]), axis=0
        )
        errors = np.abs(total[common] - 1.0)
        if errors.size == 0:
            issues.append(f"EVT class fractions have no common cells for r{radius}.")
            continue
        check = {
            "tile_id": tile_id,
            "radius_m": radius,
            "active_aoi_cells": active_aoi_cells,
            "valid_cells": int(errors.size),
            "supported_aoi_fraction": (
                float(errors.size) / active_aoi_cells
                if active_aoi_cells > 0
                else None
            ),
            "mean_absolute_error": float(errors.mean()),
            "maximum_absolute_error": float(errors.max()),
            "minimum_sum": float(total[common].min()),
            "maximum_sum": float(total[common].max()),
        }
        fraction_checks.append(check)
        if check["maximum_absolute_error"] > fraction_sum_tolerance:
            issues.append(
                f"EVT fractions do not sum to one for r{radius}: maximum "
                f"error {check['maximum_absolute_error']}."
            )

    logical_vrt = summary.get("logical_vrt")
    if logical_vrt:
        vrt_path = Path(logical_vrt)
        if not vrt_path.exists():
            issues.append(f"LANDFIRE logical VRT does not exist: {vrt_path}")
        else:
            with rasterio.open(vrt_path) as dataset:
                if dataset.count != len(expected_ids):
                    issues.append("LANDFIRE VRT has the wrong band count.")
                if list(dataset.descriptions) != expected_ids:
                    issues.append("LANDFIRE VRT band descriptions are out of order.")
                if dataset.crs != target_crs:
                    issues.append("LANDFIRE VRT has the wrong CRS.")
                if (dataset.width, dataset.height) != (
                    int(grid["bounding_width_cells"]),
                    int(grid["bounding_height_cells"]),
                ):
                    issues.append("LANDFIRE VRT has the wrong dimensions.")

    variable_ranges = {
        variable: {
            "files": int(values["files"]),
            "valid_cells": int(values["valid_cells"]),
            "minimum": values["minimum"],
            "maximum": values["maximum"],
            "mean": values["sum"] / values["valid_cells"],
        }
        for variable, values in sorted(variable_accumulators.items())
    }
    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": summary["build_id"],
        "source_id": "landfire",
        "release": release,
        "tile_id": tile_id,
        "band_count": len(inventories),
        "derived_cog_count": cog_count,
        "derived_cog_bytes": total_bytes,
        "derived_cog_mib": total_bytes / (1024**2),
        "empty_band_count": len(empty_ids),
        "empty_band_ids": empty_ids,
        "range_tolerance": range_tolerance,
        "fraction_sum_tolerance": fraction_sum_tolerance,
        "maximum_height_m": maximum_height_m,
        "variable_ranges": variable_ranges,
        "evt_fraction_sum_checks": fraction_checks,
        "maximum_evt_fraction_sum_error": max(
            (check["maximum_absolute_error"] for check in fraction_checks),
            default=None,
        ),
        "minimum_supported_aoi_fraction": min(
            (
                check["supported_aoi_fraction"]
                for check in fraction_checks
                if check["supported_aoi_fraction"] is not None
            ),
            default=None,
        ),
        "issues": issues,
        "all_checks_passed": not issues,
    }


def plot_landfire_tile_preview(
    plan: dict[str, Any],
    summary: dict[str, Any],
    output_path: Path,
) -> Path:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    inventories = load_inventories(summary)
    tile_id = str(summary["tile_id"])
    release = str(summary["release"])
    radii = [int(value) for value in summary["neighborhoods_m"]]
    local_radius = min(radii)
    context_radius = 1000 if 1000 in radii else local_radius

    class_arrays: list[np.ma.MaskedArray] = []
    reference: dict[str, Any] | None = None
    for model_class in MODEL_CLASSES:
        identifier = (
            f"availability__landfire__evt_{model_class}_fraction__mean__"
            f"r{local_radius}__{release.lower()}"
        )
        path = tile_path(inventories[identifier], tile_id)
        if path is None:
            raise ValueError(f"Preview band is all NoData: {identifier}")
        with rasterio.open(path) as dataset:
            class_arrays.append(dataset.read(1, masked=True))
            if reference is None:
                reference = {"bounds": dataset.bounds, "crs": dataset.crs}
    assert reference is not None
    stacked = np.ma.stack(class_arrays)
    dominant = np.ma.array(
        np.argmax(stacked.filled(-np.inf), axis=0),
        mask=np.logical_or.reduce(
            [np.ma.getmaskarray(value) for value in class_arrays]
        ),
    )

    def read_optional(variable: str, radius: int) -> np.ma.MaskedArray:
        identifier = (
            f"availability__landfire__{variable}__mean__"
            f"r{radius}__{release.lower()}"
        )
        path = tile_path(inventories[identifier], tile_id)
        if path is None:
            return np.ma.masked_all(dominant.shape, dtype=np.float32)
        with rasterio.open(path) as dataset:
            return dataset.read(1, masked=True)

    tree_cover = read_optional(
        "dominant_tree_cover_fraction_conditional", context_radius
    )
    tree_height = read_optional(
        "dominant_tree_height_m_conditional", context_radius
    )
    coverage_identifier = (
        "availability__landfire__source_coverage_fraction__value__"
        f"r{local_radius}__{release.lower()}"
    )
    coverage_path = tile_path(inventories[coverage_identifier], tile_id)
    if coverage_path is None:
        coverage = np.ma.masked_all(dominant.shape, dtype=np.float32)
    else:
        with rasterio.open(coverage_path) as dataset:
            coverage = dataset.read(1, masked=True)

    bounds = reference["bounds"]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    boundary = gpd.read_file(
        plan["aoi"]["path"], layer=plan["aoi"].get("layer")
    ).to_crs(reference["crs"])
    colors = [EVT_COLORS[name] for name in MODEL_CLASSES]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].imshow(
        dominant,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(MODEL_CLASSES) - 0.5,
    )
    present = sorted(np.unique(dominant.compressed()).astype(int).tolist())
    axes[0, 0].legend(
        handles=[
            Patch(
                facecolor=colors[index],
                label=MODEL_CLASSES[index].replace("_", " "),
            )
            for index in present
        ],
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
    )
    axes[0, 0].set_title(f"Dominant EVT class, r={local_radius} m")
    panels = [
        (
            axes[0, 1],
            tree_cover,
            "Conditional tree cover",
            "YlGn",
            0.0,
            1.0,
            context_radius,
        ),
        (
            axes[1, 0],
            tree_height,
            "Conditional tree height (m)",
            "viridis",
            0.0,
            None,
            context_radius,
        ),
        (
            axes[1, 1],
            coverage,
            "Source coverage",
            "cividis",
            0.0,
            1.0,
            local_radius,
        ),
    ]
    for axis, values, title, color_map, minimum, maximum, radius in panels:
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            cmap=color_map,
            vmin=minimum,
            vmax=maximum,
        )
        figure.colorbar(image, ax=axis, shrink=0.78)
        axis.set_title(f"{title}, r={radius} m")
    for axis in axes.flat:
        boundary.boundary.plot(ax=axis, color="black", linewidth=1.0)
        axis.set_xlim(bounds.left, bounds.right)
        axis.set_ylim(bounds.bottom, bounds.top)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"LANDFIRE QA | {tile_id} | {release}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_landfire_derivation_validation(
    validation: dict[str, Any], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
