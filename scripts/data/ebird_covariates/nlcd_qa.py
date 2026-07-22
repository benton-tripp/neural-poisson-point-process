"""Validation and visual QA for derived Annual NLCD covariates."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS


BAND_PATTERN = re.compile(
    r"^availability__annual_nlcd__(?P<variable>.+)__"
    r"(?P<statistic>mean|value)__r(?P<radius>\d+)__y(?P<year>\d{4})$"
)

NLCD_COLORS = {
    11: "#466b9f",
    12: "#d1def8",
    21: "#dec5c5",
    22: "#d99282",
    23: "#eb0000",
    24: "#ab0000",
    31: "#b3ac9f",
    41: "#68ab5f",
    42: "#1c5f2c",
    43: "#b5c58f",
    52: "#ccb879",
    71: "#dfdfc2",
    81: "#dcd939",
    82: "#ab6c28",
    90: "#b8d9eb",
    95: "#6c9fb8",
}


def parse_band_id(identifier: str) -> dict[str, Any]:
    match = BAND_PATTERN.match(identifier)
    if match is None:
        raise ValueError(f"Unrecognized Annual NLCD band ID: {identifier}")
    parsed = match.groupdict()
    return {
        "variable": parsed["variable"],
        "statistic": parsed["statistic"],
        "radius_m": int(parsed["radius"]),
        "year": int(parsed["year"]),
    }


def load_inventories(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventories: dict[str, dict[str, Any]] = {}
    for value in summary.get("inventory_paths", []):
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Annual NLCD inventory does not exist: {path}")
        inventory = json.loads(path.read_text(encoding="utf-8"))
        identifier = inventory["band_id"]
        if identifier in inventories:
            raise ValueError(f"Duplicate Annual NLCD inventory: {identifier}")
        inventories[identifier] = inventory
    return inventories


def tile_path(inventory: dict[str, Any], tile_id: str) -> Path:
    matches = [
        Path(tile["path"])
        for tile in inventory.get("tiles", [])
        if tile.get("tile_id") == tile_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Inventory {inventory['band_id']} has {len(matches)} records "
            f"for tile {tile_id}."
        )
    return matches[0]


def validate_nlcd_derivation(
    plan: dict[str, Any],
    summary: dict[str, Any],
    range_tolerance: float = 1e-5,
    fraction_sum_tolerance: float = 1e-4,
) -> dict[str, Any]:
    inventories = load_inventories(summary)
    selected_tiles = list(summary.get("tile_ids", []))
    expected_tiles = set(selected_tiles)
    aoi_mask_rule = str(plan["grid"].get("aoi_mask_rule", "center"))
    active_cell_key = (
        "active_cells_all_touched_rule"
        if aoi_mask_rule == "all_touched"
        else "active_cells_center_rule"
    )
    active_aoi_cells = {
        tile["tile_id"]: int(
            tile.get(
                active_cell_key,
                int(tile.get("width", 0)) * int(tile.get("height", 0)),
            )
        )
        for tile in plan["grid"]["tiles"]
    }
    tile_contracts = {
        tile["tile_id"]: tile for tile in plan["grid"]["tiles"]
    }
    issues: list[str] = []
    summary_mask_rule = str(summary.get("aoi_mask_rule", "center"))
    if summary_mask_rule != aoi_mask_rule:
        issues.append(
            f"Summary AOI mask rule {summary_mask_rule!r} does not match "
            f"build-plan rule {aoi_mask_rule!r}."
        )
    target_crs = CRS.from_user_input(plan["grid"]["crs"])
    tile_size = int(plan["grid"]["tile_width_cells"])
    expected_bands = int(summary["expected_band_count"])

    if len(inventories) != expected_bands:
        issues.append(
            f"Expected {expected_bands} inventories; found {len(inventories)}."
        )
    if int(summary["band_count"]) != expected_bands:
        issues.append(
            f"Summary band count {summary['band_count']} does not match "
            f"{expected_bands}."
        )

    paths_by_band_tile: dict[tuple[str, str], Path] = {}
    variable_accumulators: dict[str, dict[str, float]] = {}
    cog_count = 0
    total_bytes = 0
    for identifier, inventory in inventories.items():
        parsed = parse_band_id(identifier)
        inventory_tiles = {
            tile["tile_id"] for tile in inventory.get("tiles", [])
        }
        if inventory_tiles != expected_tiles:
            issues.append(
                f"Inventory {identifier} tiles differ from the summary tile set."
            )
        for tile_id in selected_tiles:
            try:
                path = tile_path(inventory, tile_id)
            except ValueError as exc:
                issues.append(str(exc))
                continue
            paths_by_band_tile[(identifier, tile_id)] = path
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
                expected_bounds = tile_contracts[tile_id]["bounds_m"]
                actual_bounds = [
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top,
                ]
                if not np.allclose(
                    actual_bounds,
                    expected_bounds,
                    rtol=0.0,
                    atol=1e-6,
                ):
                    issues.append(f"Derived file has the wrong tile bounds: {path}")
                expected_resolution = float(plan["grid"]["resolution_m"])
                if not np.allclose(
                    dataset.res,
                    (expected_resolution, expected_resolution),
                    rtol=0.0,
                    atol=1e-9,
                ):
                    issues.append(f"Derived file has the wrong resolution: {path}")
                if dataset.count != 1 or dataset.dtypes[0] != "float32":
                    issues.append(f"Derived file has the wrong band contract: {path}")
                if dataset.descriptions[0] != identifier:
                    issues.append(f"Derived file description does not match: {path}")
                values = dataset.read(1, masked=True).compressed().astype(np.float64)
            if values.size == 0:
                issues.append(f"Derived file has no valid cells: {path}")
                continue
            minimum = float(values.min())
            maximum = float(values.max())
            if minimum < -range_tolerance or maximum > 1.0 + range_tolerance:
                issues.append(
                    f"Derived values outside [0, 1] for {identifier} "
                    f"on {tile_id}: [{minimum}, {maximum}]."
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

    fraction_checks: list[dict[str, Any]] = []
    class_names = [
        record["name"] for record in summary["land_cover_classes"]
    ]
    for tile_id in selected_tiles:
        for year in summary["years"]:
            for radius in summary["neighborhoods_m"]:
                arrays: list[np.ma.MaskedArray] = []
                for class_name in class_names:
                    identifier = (
                        f"availability__annual_nlcd__{class_name}_fraction__"
                        f"mean__r{radius}__y{year}"
                    )
                    path = paths_by_band_tile.get((identifier, tile_id))
                    if path is None or not path.exists():
                        arrays = []
                        break
                    with rasterio.open(path) as dataset:
                        arrays.append(dataset.read(1, masked=True))
                if not arrays:
                    issues.append(
                        f"Cannot calculate class-fraction sum for {tile_id}, "
                        f"{year}, r{radius}."
                    )
                    continue
                masks = [np.ma.getmaskarray(array) for array in arrays]
                if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
                    issues.append(
                        f"Class-fraction masks differ for {tile_id}, "
                        f"{year}, r{radius}."
                    )
                common = ~np.logical_or.reduce(masks)
                total = np.sum(
                    np.stack([array.filled(0.0) for array in arrays]),
                    axis=0,
                )
                errors = np.abs(total[common] - 1.0)
                if errors.size == 0:
                    issues.append(
                        f"Class fractions have no common cells for {tile_id}, "
                        f"{year}, r{radius}."
                    )
                    continue
                check = {
                    "tile_id": tile_id,
                    "year": int(year),
                    "radius_m": int(radius),
                    "active_aoi_cells": active_aoi_cells[tile_id],
                    "valid_cells": int(errors.size),
                    "supported_aoi_fraction": (
                        float(errors.size) / active_aoi_cells[tile_id]
                    ),
                    "mean_absolute_error": float(errors.mean()),
                    "maximum_absolute_error": float(errors.max()),
                    "minimum_sum": float(total[common].min()),
                    "maximum_sum": float(total[common].max()),
                }
                fraction_checks.append(check)
                if check["maximum_absolute_error"] > fraction_sum_tolerance:
                    issues.append(
                        f"Class fractions do not sum to one for {tile_id}, "
                        f"{year}, r{radius}: maximum error "
                        f"{check['maximum_absolute_error']}."
                    )

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
        "source_id": "annual_nlcd",
        "release": summary["release"],
        "aoi_mask_rule": aoi_mask_rule,
        "summary_aoi_mask_rule": summary_mask_rule,
        "tile_ids": selected_tiles,
        "tile_count": len(selected_tiles),
        "band_count": len(inventories),
        "derived_cog_count": cog_count,
        "derived_cog_bytes": total_bytes,
        "derived_cog_mib": total_bytes / (1024**2),
        "range_tolerance": range_tolerance,
        "fraction_sum_tolerance": fraction_sum_tolerance,
        "variable_ranges": variable_ranges,
        "class_fraction_sum_checks": fraction_checks,
        "maximum_class_fraction_sum_error": max(
            (check["maximum_absolute_error"] for check in fraction_checks),
            default=None,
        ),
        "minimum_supported_aoi_fraction": min(
            (check["supported_aoi_fraction"] for check in fraction_checks),
            default=None,
        ),
        "issues": issues,
        "all_checks_passed": not issues,
    }


def plot_nlcd_tile_preview(
    plan: dict[str, Any],
    summary: dict[str, Any],
    tile_id: str,
    output_path: Path,
) -> Path:
    import geopandas as gpd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    inventories = load_inventories(summary)
    if tile_id not in summary.get("tile_ids", []):
        raise ValueError(f"Tile {tile_id} is absent from the derivation summary.")
    year = max(int(value) for value in summary["years"])
    radii = [int(value) for value in summary["neighborhoods_m"]]
    class_radius = min(radii)
    context_radius = 1000 if 1000 in radii else class_radius
    classes = list(summary["land_cover_classes"])

    class_arrays = []
    reference = None
    for record in classes:
        identifier = (
            f"availability__annual_nlcd__{record['name']}_fraction__"
            f"mean__r{class_radius}__y{year}"
        )
        with rasterio.open(tile_path(inventories[identifier], tile_id)) as dataset:
            class_arrays.append(dataset.read(1, masked=True))
            if reference is None:
                reference = {
                    "bounds": dataset.bounds,
                    "crs": dataset.crs,
                }
    stacked = np.ma.stack(class_arrays)
    dominant = np.ma.array(
        np.argmax(stacked.filled(-np.inf), axis=0),
        mask=np.logical_or.reduce([np.ma.getmaskarray(value) for value in class_arrays]),
    )

    def read_variable(variable: str, statistic: str) -> np.ma.MaskedArray:
        identifier = (
            f"availability__annual_nlcd__{variable}__{statistic}__"
            f"r{context_radius}__y{year}"
        )
        with rasterio.open(tile_path(inventories[identifier], tile_id)) as dataset:
            return dataset.read(1, masked=True)

    impervious = read_variable("impervious_fraction", "mean")
    change = read_variable("land_cover_change_fraction", "mean")
    diversity = read_variable("land_cover_shannon_diversity", "value")
    bounds = reference["bounds"]
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    boundary = gpd.read_file(
        plan["aoi"]["path"],
        layer=plan["aoi"].get("layer"),
    ).to_crs(reference["crs"])

    colors = [NLCD_COLORS[int(record["value"])] for record in classes]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    dominant_image = axes[0, 0].imshow(
        dominant,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        cmap=ListedColormap(colors),
        vmin=-0.5,
        vmax=len(classes) - 0.5,
    )
    del dominant_image
    present = sorted(np.unique(dominant.compressed()).astype(int).tolist())
    axes[0, 0].legend(
        handles=[
            Patch(
                facecolor=colors[index],
                label=classes[index]["name"].replace("_", " "),
            )
            for index in present
        ],
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
    )
    axes[0, 0].set_title(f"Dominant land cover, r={class_radius} m")

    panels = [
        (axes[0, 1], impervious, "Impervious fraction", "magma"),
        (axes[1, 0], change, "Annual land-cover change fraction", "YlOrRd"),
        (axes[1, 1], diversity, "Land-cover diversity", "viridis"),
    ]
    for axis, values, title, color_map in panels:
        image = axis.imshow(
            values,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            cmap=color_map,
            vmin=0.0,
            vmax=1.0,
        )
        figure.colorbar(image, ax=axis, shrink=0.78)
        axis.set_title(f"{title}, r={context_radius} m")

    for axis in axes.flat:
        boundary.boundary.plot(ax=axis, color="black", linewidth=1.0)
        axis.set_xlim(bounds.left, bounds.right)
        axis.set_ylim(bounds.bottom, bounds.top)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Annual NLCD QA | {tile_id} | {year}", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def write_nlcd_derivation_validation(
    validation: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def validate_nlcd_checklist_support(
    summary: dict[str, Any],
    checklist_path: Path,
) -> tuple[dict[str, Any], Any]:
    """Measure derived NLCD support at processed checklist coordinates."""
    import pandas as pd
    import pyarrow.parquet as pq
    from pyproj import Transformer

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
    observation_year = pd.to_datetime(
        checklists["observation_date"], errors="coerce"
    ).dt.year.astype("Int64")
    latitude = pd.to_numeric(checklists["latitude"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    longitude = pd.to_numeric(
        checklists["longitude"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    coordinate_valid = np.isfinite(latitude) & np.isfinite(longitude)

    years = [int(value) for value in summary["years"]]
    radii = [int(value) for value in summary["neighborhoods_m"]]
    class_name = str(summary["land_cover_classes"][0]["name"])
    logical_vrt = Path(summary["logical_vrt"])
    if not logical_vrt.exists():
        raise FileNotFoundError(f"Annual NLCD logical VRT does not exist: {logical_vrt}")

    checklist_count = len(checklists)
    rows = np.full(checklist_count, -1, dtype=np.int64)
    columns_index = np.full(checklist_count, -1, dtype=np.int64)
    support_by_radius = {
        radius: np.zeros(checklist_count, dtype=bool) for radius in radii
    }

    with rasterio.open(logical_vrt) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Annual NLCD logical VRT has no CRS: {logical_vrt}")
        descriptions = list(dataset.descriptions)
        description_to_index = {
            description: index + 1
            for index, description in enumerate(descriptions)
            if description is not None
        }
        required_bands: dict[tuple[int, int], int] = {}
        for year in years:
            for radius in radii:
                identifier = (
                    f"availability__annual_nlcd__{class_name}_fraction__"
                    f"mean__r{radius}__y{year}"
                )
                if identifier not in description_to_index:
                    raise ValueError(
                        f"Annual NLCD logical VRT is missing support band {identifier}."
                    )
                required_bands[(year, radius)] = description_to_index[identifier]

        valid_positions = np.flatnonzero(coordinate_valid)
        transformer = Transformer.from_crs(
            "EPSG:4326", dataset.crs, always_xy=True
        )
        x, y = transformer.transform(
            longitude[valid_positions], latitude[valid_positions], errcheck=False
        )
        transformed_valid = np.isfinite(x) & np.isfinite(y)
        transformed_positions = valid_positions[transformed_valid]
        transformed_rows, transformed_columns = rasterio.transform.rowcol(
            dataset.transform,
            np.asarray(x)[transformed_valid],
            np.asarray(y)[transformed_valid],
        )
        rows[transformed_positions] = np.asarray(transformed_rows, dtype=np.int64)
        columns_index[transformed_positions] = np.asarray(
            transformed_columns, dtype=np.int64
        )
        in_raster_extent = (
            (rows >= 0)
            & (rows < dataset.height)
            & (columns_index >= 0)
            & (columns_index < dataset.width)
        )
        year_values = observation_year.to_numpy(dtype=np.float64, na_value=np.nan)
        for year in years:
            year_positions = np.flatnonzero(
                in_raster_extent & (year_values == float(year))
            )
            if year_positions.size == 0:
                continue
            for radius in radii:
                values = dataset.read(required_bands[(year, radius)], masked=True)
                value_mask = np.ma.getmaskarray(values)
                support_by_radius[radius][year_positions] = ~value_mask[
                    rows[year_positions], columns_index[year_positions]
                ]

    year_in_source = observation_year.isin(years).to_numpy(dtype=bool)
    eligible = coordinate_valid & in_raster_extent & year_in_source
    support_records: list[dict[str, Any]] = []
    for radius in radii:
        supported = support_by_radius[radius]
        supported_count = int(np.count_nonzero(eligible & supported))
        eligible_count = int(np.count_nonzero(eligible))
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

    year_radius_records: list[dict[str, Any]] = []
    year_values = observation_year.to_numpy(dtype=np.float64, na_value=np.nan)
    for year in years:
        year_eligible = eligible & (year_values == float(year))
        eligible_count = int(np.count_nonzero(year_eligible))
        for radius in radii:
            supported_count = int(
                np.count_nonzero(year_eligible & support_by_radius[radius])
            )
            year_radius_records.append(
                {
                    "year": year,
                    "radius_m": radius,
                    "eligible_checklists": eligible_count,
                    "supported_checklists": supported_count,
                    "unsupported_checklists": eligible_count - supported_count,
                    "supported_fraction": (
                        supported_count / eligible_count if eligible_count else None
                    ),
                }
            )

    all_radius_support = np.logical_and.reduce(
        [support_by_radius[radius] for radius in radii]
    )
    unsupported_any = ~eligible | ~all_radius_support
    diagnostics = checklists.copy()
    diagnostics["observation_year"] = observation_year
    diagnostics["coordinate_valid"] = coordinate_valid
    diagnostics["coordinate_in_raster_extent"] = in_raster_extent
    diagnostics["year_in_nlcd_source"] = year_in_source
    for radius in radii:
        diagnostics[f"nlcd_supported_r{radius}"] = support_by_radius[radius]
    unsupported = diagnostics.loc[unsupported_any].copy()

    validation = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": summary["build_id"],
        "source_id": "annual_nlcd",
        "release": summary["release"],
        "aoi_mask_rule": summary.get("aoi_mask_rule", "center"),
        "logical_vrt": str(logical_vrt),
        "checklist_path": str(checklist_path),
        "checklist_count": checklist_count,
        "coordinate_valid_count": int(np.count_nonzero(coordinate_valid)),
        "coordinate_in_raster_extent_count": int(
            np.count_nonzero(coordinate_valid & in_raster_extent)
        ),
        "year_in_source_count": int(np.count_nonzero(year_in_source)),
        "eligible_checklist_count": int(np.count_nonzero(eligible)),
        "unsupported_at_any_radius_count": int(np.count_nonzero(unsupported_any)),
        "support_by_radius": support_records,
        "support_by_year_radius": year_radius_records,
    }
    return validation, unsupported


def write_nlcd_checklist_support(
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
