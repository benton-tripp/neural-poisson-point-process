"""Derive model-scale Annual NLCD bands on the canonical tiled grid."""

from __future__ import annotations

import json
import math
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.signal import fftconvolve

from .nlcd import AWS_REGION, LAND_COVER_CLASSES
from .raster_engine import (
    aoi_mask_all_touched,
    load_plan_aoi,
    safe_band_slug,
    write_band_inventory,
    write_cog,
    write_logical_vrt,
)


NODATA = -9999.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_index(registration: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    records = registration.get("sources")
    if not isinstance(records, list) or not records:
        raise ValueError("Annual NLCD source registration has no sources.")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = (record.get("product_code"), record.get("year"))
        if not isinstance(key[0], str) or not isinstance(key[1], int):
            raise ValueError("Annual NLCD source records require product_code and year.")
        if key in indexed:
            raise ValueError(f"Duplicate Annual NLCD source registration: {key}")
        raster_uri = record.get("raster_uri")
        if not isinstance(raster_uri, str) or not raster_uri:
            raise ValueError(f"Annual NLCD source {key} has no raster_uri.")
        indexed[key] = record
    return indexed


@contextmanager
def raster_environment(registration: dict[str, Any]) -> Iterator[None]:
    if registration.get("backend") == "aws_requester_pays":
        with rasterio.Env(
            AWS_REQUEST_PAYER="requester",
            AWS_REGION=registration.get("aws_region", AWS_REGION),
        ):
            yield
    else:
        yield


def validate_source_grid_pair(current: Any, previous: Any, year: int) -> None:
    if (
        current.crs != previous.crs
        or current.width != previous.width
        or current.height != previous.height
        or not current.transform.almost_equals(previous.transform)
    ):
        raise ValueError(
            f"Annual NLCD land-cover grids for {year - 1} and {year} do not align."
        )


def expanded_tile_grid(
    tile: dict[str, Any],
    resolution: float,
    tile_cells: int,
    buffer_cells: int,
) -> tuple[list[float], Any, int]:
    min_x, min_y, max_x, max_y = (float(value) for value in tile["bounds_m"])
    buffer_m = buffer_cells * resolution
    bounds = [
        min_x - buffer_m,
        min_y - buffer_m,
        max_x + buffer_m,
        max_y + buffer_m,
    ]
    size = tile_cells + 2 * buffer_cells
    transform = from_origin(bounds[0], bounds[3], resolution, resolution)
    return bounds, transform, size


def source_window(dataset: Any, target_bounds: list[float], target_crs: str) -> Window:
    bounds = transform_bounds(
        target_crs,
        dataset.crs,
        *target_bounds,
        densify_pts=21,
    )
    floating = from_bounds(*bounds, transform=dataset.transform)
    col_start = math.floor(floating.col_off) - 2
    row_start = math.floor(floating.row_off) - 2
    col_stop = math.ceil(floating.col_off + floating.width) + 2
    row_stop = math.ceil(floating.row_off + floating.height) + 2
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def read_window(dataset: Any, window: Window) -> tuple[np.ndarray, np.ndarray, Any]:
    values = dataset.read(1, window=window, boundless=True, masked=True)
    mask = np.ma.getmaskarray(values)
    return np.asarray(values.data), ~mask, dataset.window_transform(window)


def reproject_average(
    source_values: np.ndarray,
    source_transform: Any,
    source_crs: Any,
    destination_transform: Any,
    destination_crs: str,
    destination_size: int,
) -> np.ndarray:
    destination = np.zeros((destination_size, destination_size), dtype=np.float32)
    reproject(
        source=source_values,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=None,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=0.0,
        resampling=Resampling.average,
        num_threads=2,
        init_dest_nodata=True,
    )
    return destination


def circular_kernel(
    radius_m: float,
    resolution_m: float,
    subpixel_samples: int = 32,
) -> np.ndarray:
    """Approximate circular cell-overlap weights by regular subpixel sampling."""
    if radius_m <= 0 or resolution_m <= 0:
        raise ValueError("Circular-kernel radius and resolution must be positive.")
    if subpixel_samples < 1:
        raise ValueError("subpixel_samples must be positive.")
    radius_cells = max(1, int(math.ceil(radius_m / resolution_m)))
    cell_offsets = np.arange(-radius_cells, radius_cells + 1)
    subpixel_offsets = (
        (np.arange(subpixel_samples, dtype=np.float64) + 0.5)
        / subpixel_samples
        - 0.5
    )
    kernel = np.zeros((len(cell_offsets), len(cell_offsets)), dtype=np.float32)
    for row, y_cell in enumerate(cell_offsets):
        y = (y_cell + subpixel_offsets) * resolution_m
        for column, x_cell in enumerate(cell_offsets):
            x = (x_cell + subpixel_offsets) * resolution_m
            yy, xx = np.meshgrid(y, x, indexing="ij")
            kernel[row, column] = np.mean(xx**2 + yy**2 <= radius_m**2)
    if kernel.sum() <= 0:
        raise ValueError("Circular-kernel approximation produced no supported area.")
    return kernel


def neighborhood_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    kernel: np.ndarray,
    minimum_coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    summed_numerator = fftconvolve(numerator, kernel, mode="same")
    summed_denominator = fftconvolve(denominator, kernel, mode="same")
    coverage = summed_denominator / float(kernel.sum())
    values = np.full(numerator.shape, NODATA, dtype=np.float32)
    supported = coverage >= minimum_coverage
    values[supported] = (
        summed_numerator[supported] / np.maximum(summed_denominator[supported], 1e-12)
    ).astype(np.float32)
    return values, coverage.astype(np.float32)


def crop_tile(values: np.ndarray, buffer_cells: int, tile_cells: int) -> np.ndarray:
    return values[
        buffer_cells : buffer_cells + tile_cells,
        buffer_cells : buffer_cells + tile_cells,
    ].copy()


def mask_tile_to_aoi(
    values: np.ndarray,
    tile: dict[str, Any],
    resolution: float,
    aoi_geometry: Any,
    all_touched: bool = False,
) -> np.ndarray:
    min_x, _, _, max_y = (float(value) for value in tile["bounds_m"])
    transform = from_origin(min_x, max_y, resolution, resolution)
    inside = geometry_mask(
        [aoi_geometry],
        out_shape=values.shape,
        transform=transform,
        invert=True,
        all_touched=all_touched,
    )
    values[~inside] = NODATA
    return values


def band_id(variable: str, radius_m: int, year: int, statistic: str = "mean") -> str:
    return (
        f"availability__annual_nlcd__{variable}__{statistic}__"
        f"r{radius_m}__y{year}"
    )


def output_band_tile(
    inventories: dict[str, dict[str, Any]],
    band_order: list[str],
    build_id: str,
    output_dir: Path,
    tile: dict[str, Any],
    values: np.ndarray,
    transform: Any,
    target_crs: str,
    identifier: str,
    tags: dict[str, str],
    overwrite: bool,
) -> None:
    slug = safe_band_slug(identifier)
    output_path = output_dir / "tiles" / tile["tile_id"] / f"{slug}.tif"
    if output_path.exists() and not overwrite:
        with rasterio.open(output_path) as existing:
            existing_mask_rule = existing.tags().get("aoi_mask_rule", "center")
            expected_mask_rule = tags.get("aoi_mask_rule", "center")
            if existing_mask_rule != expected_mask_rule:
                raise ValueError(
                    f"Existing tile {output_path} uses AOI mask rule "
                    f"{existing_mask_rule!r}, but the build plan requires "
                    f"{expected_mask_rule!r}. Rerun with --overwrite."
                )
            valid_cells = int((existing.read_masks(1) > 0).sum())
        reused = True
    else:
        valid_cells = int((np.isfinite(values) & (values != NODATA)).sum())
        if valid_cells == 0:
            return
        write_cog(
            output_path=output_path,
            values=values.astype(np.float32),
            transform=transform,
            crs=target_crs,
            nodata=NODATA,
            description=identifier,
            tags=tags,
            overview_resampling="average",
        )
        reused = False

    if identifier not in inventories:
        band_order.append(identifier)
        inventories[identifier] = {
            "schema_version": 1,
            "build_id": build_id,
            "band_id": identifier,
            "dtype": "float32",
            "nodata": NODATA,
            "resampling": "derived",
            "tiles": [],
        }
    inventories[identifier]["tiles"].append(
        {
            "tile_id": tile["tile_id"],
            "path": str(output_path.resolve()),
            "valid_cells": valid_cells,
            "coverage_fraction": valid_cells / values.size,
            "reused": reused,
        }
    )


def finalize_inventories(
    inventories: dict[str, dict[str, Any]],
    band_order: list[str],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    ordered: list[dict[str, Any]] = []
    paths: list[str] = []
    for identifier in band_order:
        inventory = inventories[identifier]
        inventory["tile_count"] = len(inventory["tiles"])
        inventory["valid_cells"] = sum(
            tile["valid_cells"] for tile in inventory["tiles"]
        )
        path = output_dir / "inventories" / f"{safe_band_slug(identifier)}.json"
        write_band_inventory(inventory, path)
        ordered.append(inventory)
        paths.append(str(path.resolve()))
    return ordered, paths


def write_derived_tile(
    values: np.ndarray,
    identifier: str,
    tags: dict[str, str],
    *,
    inventories: dict[str, dict[str, Any]],
    band_order: list[str],
    plan: dict[str, Any],
    output_dir: Path,
    tile: dict[str, Any],
    tile_transform: Any,
    target_crs: str,
    resolution: float,
    aoi_geometry: Any,
    aoi_all_touched: bool,
    buffer_cells: int,
    tile_cells: int,
    overwrite: bool,
) -> None:
    output_tags = {
        **tags,
        "aoi_mask_rule": "all_touched" if aoi_all_touched else "center",
    }
    cropped = crop_tile(values, buffer_cells, tile_cells)
    cropped = mask_tile_to_aoi(
        cropped,
        tile,
        resolution,
        aoi_geometry,
        all_touched=aoi_all_touched,
    )
    output_band_tile(
        inventories,
        band_order,
        plan["build_id"],
        output_dir,
        tile,
        cropped,
        tile_transform,
        target_crs,
        identifier,
        output_tags,
        overwrite,
    )


def derive_land_cover_tile_year(
    *,
    current: Any,
    previous: Any,
    year: int,
    class_map: dict[int, str],
    expanded_bounds: list[float],
    destination_transform: Any,
    destination_size: int,
    target_crs: str,
    radii: list[int],
    kernels: dict[int, np.ndarray],
    minimum_coverage: float,
    registration: dict[str, Any],
    write_context: dict[str, Any],
) -> None:
    window = source_window(current, expanded_bounds, target_crs)
    current_values, current_mask, source_transform = read_window(current, window)
    previous_values, previous_mask, _ = read_window(previous, window)
    class_values = np.array(list(class_map), dtype=current_values.dtype)
    current_valid = current_mask & np.isin(current_values, class_values)
    previous_valid = previous_mask & np.isin(previous_values, class_values)
    coverage = reproject_average(
        current_valid.astype(np.float32),
        source_transform,
        current.crs,
        destination_transform,
        target_crs,
        destination_size,
    )

    fractions_by_radius: dict[int, list[np.ndarray]] = {
        radius: [] for radius in radii
    }
    coverage_by_radius: dict[int, np.ndarray] = {}
    for class_value, class_name in class_map.items():
        class_area = reproject_average(
            (current_valid & (current_values == class_value)).astype(np.float32),
            source_transform,
            current.crs,
            destination_transform,
            target_crs,
            destination_size,
        )
        for radius in radii:
            fraction, neighborhood_coverage = neighborhood_ratio(
                class_area,
                coverage,
                kernels[radius],
                minimum_coverage,
            )
            fractions_by_radius[radius].append(fraction)
            coverage_by_radius[radius] = neighborhood_coverage
            write_derived_tile(
                fraction,
                band_id(f"{class_name}_fraction", radius, year),
                {
                    "source": "Annual NLCD",
                    "release": registration["release"],
                    "product": "LndCov",
                    "year": str(year),
                    "class_value": str(class_value),
                    "radius_m": str(radius),
                    "derivation": "area_fraction_then_circular_neighborhood",
                },
                **write_context,
            )

    for radius in radii:
        stacked = np.stack(fractions_by_radius[radius])
        supported = stacked[0] != NODATA
        probabilities = np.where(stacked == NODATA, 0.0, stacked)
        positive = probabilities > 0
        shannon_terms = np.zeros(probabilities.shape, dtype=np.float32)
        shannon_terms[positive] = (
            probabilities[positive] * np.log(probabilities[positive])
        )
        if len(class_map) > 1:
            diversity = -shannon_terms.sum(axis=0) / math.log(len(class_map))
        else:
            diversity = np.zeros(probabilities.shape[1:], dtype=np.float32)
        fragmentation = 1.0 - probabilities.max(axis=0)
        diversity[~supported] = NODATA
        fragmentation[~supported] = NODATA
        for variable, values, definition in (
            (
                "land_cover_shannon_diversity",
                diversity,
                "normalized_shannon_entropy",
            ),
            (
                "land_cover_fragmentation",
                fragmentation,
                "one_minus_dominant_class_fraction",
            ),
        ):
            write_derived_tile(
                values,
                band_id(variable, radius, year, "value"),
                {
                    "source": "Annual NLCD",
                    "release": registration["release"],
                    "product": "LndCov",
                    "year": str(year),
                    "radius_m": str(radius),
                    "derivation": definition,
                },
                **write_context,
            )

    local_radius = min(radii)
    local_coverage = np.clip(coverage_by_radius[local_radius], 0.0, 1.0)
    write_derived_tile(
        local_coverage,
        band_id("source_coverage_fraction", local_radius, year, "value"),
        {
            "source": "Annual NLCD",
            "release": registration["release"],
            "product": "LndCov",
            "year": str(year),
            "radius_m": str(local_radius),
            "derivation": "valid_source_area_fraction",
        },
        **write_context,
    )

    pair_valid = current_valid & previous_valid
    changed = pair_valid & (current_values != previous_values)
    pair_coverage = reproject_average(
        pair_valid.astype(np.float32),
        source_transform,
        current.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    changed_area = reproject_average(
        changed.astype(np.float32),
        source_transform,
        current.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    for radius in radii:
        change_fraction, _ = neighborhood_ratio(
            changed_area,
            pair_coverage,
            kernels[radius],
            minimum_coverage,
        )
        write_derived_tile(
            change_fraction,
            band_id("land_cover_change_fraction", radius, year),
            {
                "source": "Annual NLCD",
                "release": registration["release"],
                "product": "derived_from_LndCov",
                "year": str(year),
                "previous_year": str(year - 1),
                "radius_m": str(radius),
                "derivation": "adjacent_year_changed_area_fraction",
            },
            **write_context,
        )


def derive_impervious_tile_year(
    *,
    impervious: Any,
    year: int,
    expanded_bounds: list[float],
    destination_transform: Any,
    destination_size: int,
    target_crs: str,
    radii: list[int],
    kernels: dict[int, np.ndarray],
    minimum_coverage: float,
    registration: dict[str, Any],
    write_context: dict[str, Any],
) -> None:
    window = source_window(impervious, expanded_bounds, target_crs)
    values, source_mask, source_transform = read_window(impervious, window)
    valid = (
        source_mask
        & np.isfinite(values)
        & (values >= 0)
        & (values <= 100)
    )
    coverage = reproject_average(
        valid.astype(np.float32),
        source_transform,
        impervious.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    impervious_area = reproject_average(
        np.where(valid, values.astype(np.float32) / 100.0, 0.0),
        source_transform,
        impervious.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    for radius in radii:
        fraction, _ = neighborhood_ratio(
            impervious_area,
            coverage,
            kernels[radius],
            minimum_coverage,
        )
        write_derived_tile(
            fraction,
            band_id("impervious_fraction", radius, year),
            {
                "source": "Annual NLCD",
                "release": registration["release"],
                "product": "FctImp",
                "year": str(year),
                "radius_m": str(radius),
                "derivation": "percent_to_fraction_then_circular_neighborhood",
            },
            **write_context,
        )


def write_json_atomic(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive_nlcd(
    *,
    plan: dict[str, Any],
    registration: dict[str, Any],
    output_dir: Path,
    years: list[int] | None = None,
    neighborhoods_m: list[int] | None = None,
    tile_ids: list[str] | None = None,
    minimum_coverage: float = 0.8,
    classes: dict[int, str] | None = None,
    write_vrt: bool = True,
    overwrite: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    """Build Annual NLCD ecological bands for each active plan tile.

    Land-cover classes and imperviousness are aggregated as area fractions
    before circular-neighborhood summaries are calculated. Annual change is
    calculated from aligned adjacent-year land-cover rasters.
    """
    started_at = time.perf_counter()
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1].")
    if registration.get("source_id") != "annual_nlcd":
        raise ValueError("Source registration must describe annual_nlcd.")

    source_records = source_index(registration)
    temporal = plan["temporal"]
    target_years = sorted(
        set(
            years
            or range(
                int(temporal["start_year"]),
                int(temporal["end_year"]) + 1,
            )
        )
    )
    if not target_years:
        raise ValueError("At least one target year is required.")
    radii = sorted(
        set(
            int(round(value))
            for value in (neighborhoods_m or plan["neighborhoods_m"])
        )
    )
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("Annual NLCD neighborhood radii must be positive.")
    class_map = dict(classes or LAND_COVER_CLASSES)
    if not class_map:
        raise ValueError("At least one land-cover class is required.")

    required_keys = {
        *(("LndCov", year - 1) for year in target_years),
        *(("LndCov", year) for year in target_years),
        *(("FctImp", year) for year in target_years),
    }
    missing = sorted(required_keys - set(source_records))
    if missing:
        formatted = ", ".join(f"{product}:{year}" for product, year in missing)
        raise ValueError(f"Annual NLCD source registration is missing: {formatted}")

    grid = plan["grid"]
    plan_tiles = list(grid["tiles"])
    if tile_ids:
        requested_tile_ids = list(dict.fromkeys(tile_ids))
        known_tile_ids = {tile["tile_id"] for tile in plan_tiles}
        unknown_tile_ids = sorted(set(requested_tile_ids) - known_tile_ids)
        if unknown_tile_ids:
            raise ValueError(
                "Unknown Annual NLCD plan tile IDs: "
                + ", ".join(unknown_tile_ids)
            )
        requested = set(requested_tile_ids)
        selected_tiles = [
            tile for tile in plan_tiles if tile["tile_id"] in requested
        ]
    else:
        selected_tiles = plan_tiles
    if not selected_tiles:
        raise ValueError("Annual NLCD derivation requires at least one plan tile.")
    target_crs = str(grid["crs"])
    resolution = float(grid["resolution_m"])
    tile_cells = int(grid["tile_width_cells"])
    if tile_cells != int(round(float(grid["tile_size_m"]) / resolution)):
        raise ValueError("Build-plan tile dimensions do not match its resolution.")
    if any(
        int(tile.get("width", tile_cells)) != tile_cells
        or int(tile.get("height", tile_cells)) != tile_cells
        for tile in selected_tiles
    ):
        raise ValueError("Annual NLCD derivation currently requires square plan tiles.")

    buffer_cells = max(1, int(math.ceil(max(radii) / resolution)))
    kernels = {
        radius: circular_kernel(radius, resolution)
        for radius in radii
    }
    aoi_geometry = load_plan_aoi(plan)
    aoi_all_touched = aoi_mask_all_touched(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventories: dict[str, dict[str, Any]] = {}
    band_order: list[str] = []

    with raster_environment(registration), ExitStack() as stack:
        datasets = {
            key: stack.enter_context(rasterio.open(source_records[key]["raster_uri"]))
            for key in sorted(required_keys)
        }
        for key, dataset in datasets.items():
            if dataset.crs is None:
                raise ValueError(f"Annual NLCD source {key} has no CRS.")
            if dataset.count != 1:
                raise ValueError(f"Annual NLCD source {key} must have one band.")
        for year in target_years:
            validate_source_grid_pair(
                datasets[("LndCov", year)],
                datasets[("LndCov", year - 1)],
                year,
            )

        for tile_index, tile in enumerate(selected_tiles, start=1):
            if progress:
                print(
                    f"NLCD tile {tile_index}/{len(selected_tiles)}: "
                    f"{tile['tile_id']}"
                )
            expanded_bounds, destination_transform, destination_size = (
                expanded_tile_grid(
                    tile,
                    resolution,
                    tile_cells,
                    buffer_cells,
                )
            )
            min_x, _, _, max_y = (float(value) for value in tile["bounds_m"])
            tile_transform = from_origin(min_x, max_y, resolution, resolution)
            write_context = {
                "inventories": inventories,
                "band_order": band_order,
                "plan": plan,
                "output_dir": output_dir,
                "tile": tile,
                "tile_transform": tile_transform,
                "target_crs": target_crs,
                "resolution": resolution,
                "aoi_geometry": aoi_geometry,
                "aoi_all_touched": aoi_all_touched,
                "buffer_cells": buffer_cells,
                "tile_cells": tile_cells,
                "overwrite": overwrite,
            }
            for year in target_years:
                if progress:
                    print(f"  year {year}: land cover and imperviousness")
                derive_land_cover_tile_year(
                    current=datasets[("LndCov", year)],
                    previous=datasets[("LndCov", year - 1)],
                    year=year,
                    class_map=class_map,
                    expanded_bounds=expanded_bounds,
                    destination_transform=destination_transform,
                    destination_size=destination_size,
                    target_crs=target_crs,
                    radii=radii,
                    kernels=kernels,
                    minimum_coverage=minimum_coverage,
                    registration=registration,
                    write_context=write_context,
                )
                derive_impervious_tile_year(
                    impervious=datasets[("FctImp", year)],
                    year=year,
                    expanded_bounds=expanded_bounds,
                    destination_transform=destination_transform,
                    destination_size=destination_size,
                    target_crs=target_crs,
                    radii=radii,
                    kernels=kernels,
                    minimum_coverage=minimum_coverage,
                    registration=registration,
                    write_context=write_context,
                )

    ordered_inventories, inventory_paths = finalize_inventories(
        inventories,
        band_order,
        output_dir,
    )
    if not ordered_inventories:
        raise ValueError("Annual NLCD derivation produced no valid output bands.")
    vrt_path = output_dir / "annual_nlcd.vrt"
    if write_vrt:
        write_logical_vrt(plan, ordered_inventories, vrt_path)

    expected_band_count = len(target_years) * (
        len(class_map) * len(radii)
        + 2 * len(radii)
        + 1
        + len(radii)
        + len(radii)
    )
    cog_paths = [
        Path(tile["path"])
        for inventory in ordered_inventories
        for tile in inventory["tiles"]
    ]
    cog_bytes = sum(path.stat().st_size for path in cog_paths)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": plan["build_id"],
        "source_id": "annual_nlcd",
        "release": registration["release"],
        "source_backend": registration["backend"],
        "years": target_years,
        "neighborhoods_m": radii,
        "minimum_coverage": minimum_coverage,
        "aoi_mask_rule": grid.get("aoi_mask_rule", "center"),
        "land_cover_classes": [
            {"value": value, "name": name}
            for value, name in class_map.items()
        ],
        "tile_count": len(selected_tiles),
        "plan_tile_count": len(plan_tiles),
        "tile_ids": [tile["tile_id"] for tile in selected_tiles],
        "band_count": len(ordered_inventories),
        "expected_band_count": expected_band_count,
        "derived_cog_count": len(cog_paths),
        "derived_cog_bytes": cog_bytes,
        "derived_cog_mib": cog_bytes / (1024**2),
        "elapsed_seconds": time.perf_counter() - started_at,
        "inventory_paths": inventory_paths,
        "logical_vrt": str(vrt_path.resolve()) if write_vrt else None,
        "definitions": {
            "class_fraction": "source-pixel area fraction within a circular neighborhood",
            "land_cover_shannon_diversity": "normalized Shannon entropy across modeled classes",
            "land_cover_fragmentation": "one minus the dominant modeled-class fraction",
            "source_coverage_fraction": "valid modeled land-cover area at the smallest neighborhood",
            "land_cover_change_fraction": "valid area whose class changed from year - 1 to year",
            "impervious_fraction": "fractional impervious surface averaged within a circular neighborhood",
        },
    }
    if summary["band_count"] != expected_band_count:
        raise ValueError(
            "Annual NLCD output inventory is incomplete: "
            f"expected {expected_band_count}, found {summary['band_count']}."
        )
    write_json_atomic(summary, output_dir / "annual_nlcd_summary.json")
    return summary
