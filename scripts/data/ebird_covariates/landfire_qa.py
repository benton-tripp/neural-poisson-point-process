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


def validate_landfire_checklist_support(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    checklist_path: Path,
    release: str,
) -> tuple[dict[str, Any], Any]:
    """Measure one completed LANDFIRE release at processed checklist locations."""
    import pandas as pd
    import pyarrow.parquet as pq
    from pyproj import Transformer

    release = release.upper()
    grid = plan["grid"]
    plan_tiles = list(grid["tiles"])
    plan_tile_ids = [str(tile["tile_id"]) for tile in plan_tiles]
    release_units = [
        unit
        for unit in manifest.get("units", [])
        if unit.get("component") == "vegetation"
        and str(unit.get("release", "")).upper() == release
    ]
    units_by_tile = {str(unit["tile_id"]): unit for unit in release_units}
    missing_tiles = [tile_id for tile_id in plan_tile_ids if tile_id not in units_by_tile]
    unexpected_tiles = sorted(set(units_by_tile) - set(plan_tile_ids))
    incomplete_tiles = sorted(
        tile_id
        for tile_id, unit in units_by_tile.items()
        if unit.get("status") != "completed"
    )
    if missing_tiles or unexpected_tiles or incomplete_tiles:
        problems = []
        if missing_tiles:
            problems.append("missing tiles: " + ", ".join(missing_tiles))
        if unexpected_tiles:
            problems.append("unexpected tiles: " + ", ".join(unexpected_tiles))
        if incomplete_tiles:
            problems.append("incomplete tiles: " + ", ".join(incomplete_tiles))
        raise ValueError(
            f"LANDFIRE {release} is not a complete plan-tile release: "
            + "; ".join(problems)
        )

    summaries: dict[str, dict[str, Any]] = {}
    validations: dict[str, dict[str, Any]] = {}
    for tile_id in plan_tile_ids:
        unit = units_by_tile[tile_id]
        summary_path = Path(unit["summary_path"])
        validation_path = Path(unit["validation_path"])
        if not summary_path.exists():
            raise FileNotFoundError(f"LANDFIRE summary does not exist: {summary_path}")
        if not validation_path.exists():
            raise FileNotFoundError(
                f"LANDFIRE validation does not exist: {validation_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if str(summary.get("release", "")).upper() != release:
            raise ValueError(f"LANDFIRE summary release differs for {tile_id}.")
        if not validation.get("all_checks_passed", False):
            raise ValueError(f"LANDFIRE validation does not pass for {tile_id}.")
        summaries[tile_id] = summary
        validations[tile_id] = validation

    radii = [int(value) for value in summaries[plan_tile_ids[0]]["neighborhoods_m"]]
    for tile_id, summary in summaries.items():
        summary_radii = [int(value) for value in summary["neighborhoods_m"]]
        if summary_radii != radii:
            raise ValueError(f"LANDFIRE neighborhood schema differs for {tile_id}.")

    required_columns = {
        "sampling_event_identifier",
        "latitude",
        "longitude",
        "observation_date",
    }
    optional_columns = [
        "locality_id",
        "locality",
        "locality_type",
        "protocol_name",
        "effort_distance_km",
        "distance_to_coastline_m",
    ]
    available_columns = set(pq.ParquetFile(checklist_path).schema.names)
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(
            "Checklist Parquet is missing required columns: "
            + ", ".join(missing_columns)
        )
    columns = sorted(required_columns) + [
        column for column in optional_columns if column in available_columns
    ]
    checklists = pd.read_parquet(checklist_path, columns=columns)

    checklist_count = len(checklists)
    latitude = pd.to_numeric(checklists["latitude"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    longitude = pd.to_numeric(
        checklists["longitude"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    coordinate_valid = np.isfinite(latitude) & np.isfinite(longitude)
    projected_x = np.full(checklist_count, np.nan, dtype=np.float64)
    projected_y = np.full(checklist_count, np.nan, dtype=np.float64)
    valid_positions = np.flatnonzero(coordinate_valid)
    transformer = Transformer.from_crs(
        "EPSG:4326", grid["crs"], always_xy=True
    )
    transformed_x, transformed_y = transformer.transform(
        longitude[valid_positions], latitude[valid_positions], errcheck=False
    )
    transformed_valid = np.isfinite(transformed_x) & np.isfinite(transformed_y)
    transformed_positions = valid_positions[transformed_valid]
    projected_x[transformed_positions] = np.asarray(transformed_x)[transformed_valid]
    projected_y[transformed_positions] = np.asarray(transformed_y)[transformed_valid]
    projected_coordinate_valid = np.isfinite(projected_x) & np.isfinite(projected_y)

    tile_assignment = np.full(checklist_count, "", dtype=object)
    maximum_right = max(float(tile["bounds_m"][2]) for tile in plan_tiles)
    maximum_top = max(float(tile["bounds_m"][3]) for tile in plan_tiles)
    for tile in plan_tiles:
        left, bottom, right, top = [float(value) for value in tile["bounds_m"]]
        right_test = (
            projected_x <= right if right == maximum_right else projected_x < right
        )
        top_test = projected_y <= top if top == maximum_top else projected_y < top
        selected = (
            projected_coordinate_valid
            & (tile_assignment == "")
            & (projected_x >= left)
            & right_test
            & (projected_y >= bottom)
            & top_test
        )
        tile_assignment[selected] = str(tile["tile_id"])

    in_plan_tile = tile_assignment != ""
    in_tile_raster_extent = np.zeros(checklist_count, dtype=bool)
    support_by_radius = {
        radius: np.zeros(checklist_count, dtype=bool) for radius in radii
    }
    release_slug = release.lower()
    support_variable = "evt_forest_tree_fraction"
    for tile_id in plan_tile_ids:
        positions = np.flatnonzero(tile_assignment == tile_id)
        if positions.size == 0:
            continue
        logical_vrt = Path(summaries[tile_id]["logical_vrt"])
        if not logical_vrt.exists():
            raise FileNotFoundError(f"LANDFIRE logical VRT does not exist: {logical_vrt}")
        with rasterio.open(logical_vrt) as dataset:
            if dataset.crs is None:
                raise ValueError(f"LANDFIRE logical VRT has no CRS: {logical_vrt}")
            if dataset.crs != CRS.from_user_input(grid["crs"]):
                raise ValueError(f"LANDFIRE logical VRT has the wrong CRS: {logical_vrt}")
            description_to_index = {
                description: index + 1
                for index, description in enumerate(dataset.descriptions)
                if description is not None
            }
            required_bands = {}
            for radius in radii:
                identifier = (
                    f"availability__landfire__{support_variable}__mean__"
                    f"r{radius}__{release_slug}"
                )
                if identifier not in description_to_index:
                    raise ValueError(
                        f"LANDFIRE logical VRT is missing support band {identifier}."
                    )
                required_bands[radius] = description_to_index[identifier]
            rows, columns_index = rasterio.transform.rowcol(
                dataset.transform,
                projected_x[positions],
                projected_y[positions],
            )
            rows = np.asarray(rows, dtype=np.int64)
            columns_index = np.asarray(columns_index, dtype=np.int64)
            local_extent = (
                (rows >= 0)
                & (rows < dataset.height)
                & (columns_index >= 0)
                & (columns_index < dataset.width)
            )
            local_positions = positions[local_extent]
            in_tile_raster_extent[local_positions] = True
            for radius in radii:
                values = dataset.read(required_bands[radius], masked=True)
                value_mask = np.ma.getmaskarray(values)
                support_by_radius[radius][local_positions] = ~value_mask[
                    rows[local_extent], columns_index[local_extent]
                ]

    eligible = coordinate_valid & in_plan_tile & in_tile_raster_extent
    eligible_count = int(np.count_nonzero(eligible))
    support_records = []
    for radius in radii:
        supported_count = int(np.count_nonzero(eligible & support_by_radius[radius]))
        support_records.append(
            {
                "radius_m": radius,
                "eligible_checklists": eligible_count,
                "supported_checklists": supported_count,
                "unsupported_checklists": eligible_count - supported_count,
                "supported_fraction": (
                    supported_count / eligible_count if eligible_count else None
                ),
            }
        )

    tile_support_records = []
    for tile_id in plan_tile_ids:
        tile_eligible = eligible & (tile_assignment == tile_id)
        tile_eligible_count = int(np.count_nonzero(tile_eligible))
        for radius in radii:
            supported_count = int(
                np.count_nonzero(tile_eligible & support_by_radius[radius])
            )
            tile_support_records.append(
                {
                    "tile_id": tile_id,
                    "radius_m": radius,
                    "eligible_checklists": tile_eligible_count,
                    "supported_checklists": supported_count,
                    "unsupported_checklists": tile_eligible_count - supported_count,
                    "supported_fraction": (
                        supported_count / tile_eligible_count
                        if tile_eligible_count
                        else None
                    ),
                }
            )

    all_radius_support = np.logical_and.reduce(
        [support_by_radius[radius] for radius in radii]
    )
    unsupported_any = ~eligible | ~all_radius_support
    diagnostics = checklists.copy()
    diagnostics["landfire_release"] = release
    diagnostics["landfire_tile_id"] = tile_assignment
    diagnostics["coordinate_valid"] = coordinate_valid
    diagnostics["coordinate_in_plan_tile"] = in_plan_tile
    diagnostics["coordinate_in_tile_raster_extent"] = in_tile_raster_extent
    for radius in radii:
        diagnostics[f"landfire_supported_r{radius}"] = support_by_radius[radius]
    unsupported = diagnostics.loc[unsupported_any].copy()

    validation = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": manifest.get("build_id", plan.get("build_id")),
        "source_id": "landfire",
        "release": release,
        "checklist_path": str(checklist_path),
        "support_variable": support_variable,
        "release_tile_count": len(plan_tile_ids),
        "checklist_count": checklist_count,
        "coordinate_valid_count": int(np.count_nonzero(coordinate_valid)),
        "coordinate_in_plan_tile_count": int(
            np.count_nonzero(coordinate_valid & in_plan_tile)
        ),
        "coordinate_in_tile_raster_extent_count": int(
            np.count_nonzero(coordinate_valid & in_tile_raster_extent)
        ),
        "eligible_checklist_count": eligible_count,
        "unsupported_at_any_radius_count": int(np.count_nonzero(unsupported_any)),
        "support_by_radius": support_records,
        "support_by_tile_radius": tile_support_records,
    }
    return validation, unsupported


def write_landfire_checklist_support(
    validation: dict[str, Any],
    unsupported: Any,
    output_path: Path,
    unsupported_output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(output_path)

    unsupported_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = unsupported_output_path.with_suffix(
        unsupported_output_path.suffix + ".tmp"
    )
    unsupported.to_csv(temporary_csv, index=False)
    temporary_csv.replace(unsupported_output_path)


def compare_landfire_releases(
    plan: dict[str, Any],
    manifest: dict[str, Any],
    baseline_release: str,
    comparison_release: str,
    tile_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare two complete vegetation releases on the same plan grid."""
    baseline_release = baseline_release.upper()
    comparison_release = comparison_release.upper()
    if baseline_release == comparison_release:
        raise ValueError("Baseline and comparison LANDFIRE releases must differ.")

    plan_tile_ids = [str(tile["tile_id"]) for tile in plan["grid"]["tiles"]]
    release_content: dict[str, dict[str, dict[str, Any]]] = {}
    for release in (baseline_release, comparison_release):
        units = {
            str(unit["tile_id"]): unit
            for unit in manifest.get("units", [])
            if unit.get("component") == "vegetation"
            and str(unit.get("release", "")).upper() == release
        }
        missing = [tile_id for tile_id in plan_tile_ids if tile_id not in units]
        unexpected = sorted(set(units) - set(plan_tile_ids))
        incomplete = sorted(
            tile_id
            for tile_id, unit in units.items()
            if unit.get("status") != "completed"
        )
        if missing or unexpected or incomplete:
            problems = []
            if missing:
                problems.append("missing tiles: " + ", ".join(missing))
            if unexpected:
                problems.append("unexpected tiles: " + ", ".join(unexpected))
            if incomplete:
                problems.append("incomplete tiles: " + ", ".join(incomplete))
            raise ValueError(
                f"LANDFIRE {release} is not a complete plan-tile release: "
                + "; ".join(problems)
            )

        summaries: dict[str, dict[str, Any]] = {}
        validations: dict[str, dict[str, Any]] = {}
        for tile_id in plan_tile_ids:
            summary_path = Path(units[tile_id]["summary_path"])
            validation_path = Path(units[tile_id]["validation_path"])
            if not summary_path.exists() or not validation_path.exists():
                raise FileNotFoundError(
                    f"LANDFIRE {release} artifacts are missing for {tile_id}."
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if str(summary.get("release", "")).upper() != release:
                raise ValueError(f"LANDFIRE summary release differs for {tile_id}.")
            if not validation.get("all_checks_passed", False):
                raise ValueError(f"LANDFIRE validation does not pass for {tile_id}.")
            summaries[tile_id] = summary
            validations[tile_id] = validation
        release_content[release] = {
            "summaries": summaries,
            "validations": validations,
        }

    selected_tile_ids = list(tile_ids) if tile_ids else list(plan_tile_ids)
    invalid_tiles = sorted(set(selected_tile_ids) - set(plan_tile_ids))
    if invalid_tiles:
        raise ValueError(
            "Comparison tile IDs are outside the plan: " + ", ".join(invalid_tiles)
        )
    if not selected_tile_ids:
        raise ValueError("At least one LANDFIRE comparison tile is required.")

    support_differences = []
    for tile_id in plan_tile_ids:
        checks_by_release = {}
        for release in (baseline_release, comparison_release):
            checks = release_content[release]["validations"][tile_id][
                "evt_fraction_sum_checks"
            ]
            checks_by_release[release] = {
                int(check["radius_m"]): check for check in checks
            }
        if set(checks_by_release[baseline_release]) != set(
            checks_by_release[comparison_release]
        ):
            raise ValueError(f"LANDFIRE support radii differ for {tile_id}.")
        for radius in sorted(checks_by_release[baseline_release]):
            baseline_support = checks_by_release[baseline_release][radius][
                "supported_aoi_fraction"
            ]
            comparison_support = checks_by_release[comparison_release][radius][
                "supported_aoi_fraction"
            ]
            difference = (
                float(comparison_support) - float(baseline_support)
                if baseline_support is not None and comparison_support is not None
                else None
            )
            support_differences.append(
                {
                    "tile_id": tile_id,
                    "radius_m": radius,
                    "baseline_supported_aoi_fraction": baseline_support,
                    "comparison_supported_aoi_fraction": comparison_support,
                    "difference": difference,
                }
            )

    accumulators: dict[str, dict[str, Any]] = {}
    matched_band_count: int | None = None
    for tile_id in selected_tile_ids:
        paths = {
            release: Path(
                release_content[release]["summaries"][tile_id]["logical_vrt"]
            )
            for release in (baseline_release, comparison_release)
        }
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(f"LANDFIRE logical VRT does not exist: {path}")
        with rasterio.open(paths[baseline_release]) as baseline_dataset, rasterio.open(
            paths[comparison_release]
        ) as comparison_dataset:
            if (
                baseline_dataset.crs != comparison_dataset.crs
                or baseline_dataset.transform != comparison_dataset.transform
                or baseline_dataset.shape != comparison_dataset.shape
                or baseline_dataset.count != comparison_dataset.count
            ):
                raise ValueError(f"LANDFIRE release grids differ for {tile_id}.")
            baseline_descriptions = list(baseline_dataset.descriptions)
            comparison_descriptions = list(comparison_dataset.descriptions)
            if any(value is None for value in baseline_descriptions) or any(
                value is None for value in comparison_descriptions
            ):
                raise ValueError(f"LANDFIRE band descriptions are missing for {tile_id}.")
            normalized_baseline = [
                re.sub(r"__lf\d{4}$", "", str(value).lower())
                for value in baseline_descriptions
            ]
            normalized_comparison = [
                re.sub(r"__lf\d{4}$", "", str(value).lower())
                for value in comparison_descriptions
            ]
            if normalized_baseline != normalized_comparison:
                raise ValueError(f"LANDFIRE release schemas differ for {tile_id}.")
            if matched_band_count is None:
                matched_band_count = baseline_dataset.count
            elif matched_band_count != baseline_dataset.count:
                raise ValueError("LANDFIRE tile band counts differ within the release.")

            for band_index, normalized_id in enumerate(normalized_baseline, start=1):
                parsed = parse_band_id(str(baseline_descriptions[band_index - 1]))
                record = accumulators.setdefault(
                    normalized_id,
                    {
                        "band_id_without_release": normalized_id,
                        "variable": parsed["variable"],
                        "statistic": parsed["statistic"],
                        "radius_m": parsed["radius_m"],
                        "overlap_valid_cells": 0,
                        "mask_union_cells": 0,
                        "mask_mismatch_cells": 0,
                        "sum_baseline": 0.0,
                        "sum_comparison": 0.0,
                        "sum_baseline_squared": 0.0,
                        "sum_comparison_squared": 0.0,
                        "sum_cross_product": 0.0,
                        "sum_absolute_delta": 0.0,
                        "sum_squared_delta": 0.0,
                    },
                )
                baseline_values = baseline_dataset.read(band_index, masked=True)
                comparison_values = comparison_dataset.read(band_index, masked=True)
                baseline_mask = np.ma.getmaskarray(baseline_values)
                comparison_mask = np.ma.getmaskarray(comparison_values)
                valid = ~(baseline_mask | comparison_mask)
                record["mask_union_cells"] += int(
                    np.count_nonzero(~(baseline_mask & comparison_mask))
                )
                record["mask_mismatch_cells"] += int(
                    np.count_nonzero(baseline_mask ^ comparison_mask)
                )
                if not np.any(valid):
                    continue
                baseline_valid = np.asarray(
                    baseline_values.data[valid], dtype=np.float64
                )
                comparison_valid = np.asarray(
                    comparison_values.data[valid], dtype=np.float64
                )
                difference = comparison_valid - baseline_valid
                record["overlap_valid_cells"] += int(baseline_valid.size)
                record["sum_baseline"] += float(np.sum(baseline_valid))
                record["sum_comparison"] += float(np.sum(comparison_valid))
                record["sum_baseline_squared"] += float(
                    np.sum(np.square(baseline_valid))
                )
                record["sum_comparison_squared"] += float(
                    np.sum(np.square(comparison_valid))
                )
                record["sum_cross_product"] += float(
                    np.sum(baseline_valid * comparison_valid)
                )
                record["sum_absolute_delta"] += float(np.sum(np.abs(difference)))
                record["sum_squared_delta"] += float(np.sum(np.square(difference)))

    metrics = []
    for record in accumulators.values():
        count = int(record["overlap_valid_cells"])
        mask_union = int(record["mask_union_cells"])
        if count:
            baseline_mean = record["sum_baseline"] / count
            comparison_mean = record["sum_comparison"] / count
            mean_delta = comparison_mean - baseline_mean
            mean_absolute_delta = record["sum_absolute_delta"] / count
            root_mean_squared_delta = np.sqrt(record["sum_squared_delta"] / count)
            baseline_variation = (
                record["sum_baseline_squared"]
                - record["sum_baseline"] ** 2 / count
            )
            comparison_variation = (
                record["sum_comparison_squared"]
                - record["sum_comparison"] ** 2 / count
            )
            covariance = (
                record["sum_cross_product"]
                - record["sum_baseline"] * record["sum_comparison"] / count
            )
            pearson = (
                covariance / np.sqrt(baseline_variation * comparison_variation)
                if baseline_variation > 0 and comparison_variation > 0
                else None
            )
        else:
            baseline_mean = None
            comparison_mean = None
            mean_delta = None
            mean_absolute_delta = None
            root_mean_squared_delta = None
            pearson = None
        metrics.append(
            {
                "band_id_without_release": record["band_id_without_release"],
                "variable": record["variable"],
                "statistic": record["statistic"],
                "radius_m": record["radius_m"],
                "overlap_valid_cells": count,
                "baseline_mean": baseline_mean,
                "comparison_mean": comparison_mean,
                "mean_delta": mean_delta,
                "mean_absolute_delta": mean_absolute_delta,
                "root_mean_squared_delta": root_mean_squared_delta,
                "pearson": pearson,
                "mask_union_cells": mask_union,
                "mask_mismatch_cells": int(record["mask_mismatch_cells"]),
                "mask_mismatch_fraction": (
                    record["mask_mismatch_cells"] / mask_union
                    if mask_union
                    else None
                ),
            }
        )
    metrics.sort(key=lambda value: value["band_id_without_release"])
    support_values = [
        abs(record["difference"])
        for record in support_differences
        if record["difference"] is not None
    ]
    summary = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": manifest.get("build_id", plan.get("build_id")),
        "source_id": "landfire",
        "baseline_release": baseline_release,
        "comparison_release": comparison_release,
        "release_tile_count": len(plan_tile_ids),
        "comparison_tile_ids": selected_tile_ids,
        "comparison_tile_count": len(selected_tile_ids),
        "matched_band_count": matched_band_count,
        "support_comparison_count": len(support_differences),
        "maximum_absolute_support_difference": max(support_values, default=None),
        "nonzero_support_difference_count": sum(
            value > 0 for value in support_values
        ),
        "support_differences": support_differences,
        "structural_checks_passed": True,
    }
    return summary, metrics


def write_landfire_release_comparison(
    summary: dict[str, Any],
    metrics: list[dict[str, Any]],
    output_path: Path,
    metrics_output_path: Path,
) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(output_path)

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = metrics_output_path.with_suffix(metrics_output_path.suffix + ".tmp")
    fieldnames = list(metrics[0]) if metrics else []
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(metrics)
    temporary_csv.replace(metrics_output_path)
