"""Derive model-scale LANDFIRE vegetation covariates from validated exports."""

from __future__ import annotations

import csv
import json
import math
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.signal import fftconvolve

from .landfire_crosswalk import MODEL_CLASSES
from .nlcd_derive import (
    NODATA,
    circular_kernel,
    expanded_tile_grid,
    finalize_inventories,
    neighborhood_ratio,
    reproject_average,
    write_derived_tile,
)
from .raster_engine import (
    aoi_mask_all_touched,
    load_plan_aoi,
    write_logical_vrt,
)


LIFE_FORMS = ("tree", "shrub", "herb")


def _artifact_paths(crosswalk_summary: dict[str, Any]) -> dict[str, Path]:
    artifacts = {
        artifact["id"]: Path(artifact["path"])
        for artifact in crosswalk_summary.get("artifacts", [])
    }
    required = {
        "evt_model_crosswalk",
        "evc_model_lookup",
        "evh_model_lookup",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(
            "LANDFIRE crosswalk summary lacks: " + ", ".join(missing)
        )
    return artifacts


def _read_rows(path: Path, release: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("release") == release
        ]
    if not rows:
        raise ValueError(f"LANDFIRE lookup {path} has no rows for {release}.")
    return rows


def _evt_lookup(
    crosswalk_summary: dict[str, Any], release: str
) -> dict[int, str | None]:
    rows = _read_rows(
        _artifact_paths(crosswalk_summary)["evt_model_crosswalk"],
        release,
    )
    return {
        int(row["Value"]): row["model_class"] or None
        for row in rows
    }


def _structure_lookup(
    crosswalk_summary: dict[str, Any], release: str, product: str
) -> dict[int, dict[str, Any]]:
    artifact_id = (
        "evc_model_lookup" if product == "EVC" else "evh_model_lookup"
    )
    rows = _read_rows(_artifact_paths(crosswalk_summary)[artifact_id], release)
    return {
        int(row["Value"]): {
            "lifeform": row["model_lifeform"] or None,
            "numeric_value": (
                float(row["numeric_value"])
                if row["numeric_value"] != ""
                else None
            ),
            "special_class": row["special_class"] or None,
        }
        for row in rows
    }


def _integer_lut(
    mapping: dict[int, Any],
    encode,
    *,
    default: Any,
    dtype: Any,
) -> tuple[np.ndarray, int]:
    minimum = min(mapping)
    maximum = max(mapping)
    lookup = np.full(maximum - minimum + 1, default, dtype=dtype)
    for value, record in mapping.items():
        lookup[value - minimum] = encode(record)
    return lookup, minimum


def _apply_lut(
    values: np.ndarray,
    lookup: np.ndarray,
    minimum: int,
    *,
    default: Any,
    dtype: Any,
) -> np.ndarray:
    output = np.full(values.shape, default, dtype=dtype)
    indices = values.astype(np.int64) - minimum
    supported = (indices >= 0) & (indices < len(lookup))
    output[supported] = lookup[indices[supported]]
    return output


def _validate_export_group(
    export_summary: dict[str, Any],
) -> tuple[str, str, dict[str, dict[str, Any]]]:
    exports = export_summary.get("exports")
    if not isinstance(exports, list) or not exports:
        raise ValueError("LANDFIRE export summary has no exports.")
    tile_ids = {record["tile_id"] for record in exports}
    releases = {record["version"] for record in exports}
    if len(tile_ids) != 1 or len(releases) != 1:
        raise ValueError(
            "LANDFIRE derivation requires one tile and one release per export summary."
        )
    by_product = {record["product"]: record for record in exports}
    missing = sorted({"EVT", "EVC", "EVH"} - set(by_product))
    if missing:
        raise ValueError(
            "LANDFIRE export summary lacks products: " + ", ".join(missing)
        )
    if len(by_product) != len(exports):
        raise ValueError("LANDFIRE export summary has duplicate product records.")
    return next(iter(tile_ids)), next(iter(releases)), by_product


def _read_source(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    values = dataset.read(1)
    mask = dataset.read_masks(1) > 0
    return values, mask


def _validate_aligned_sources(datasets: dict[str, Any]) -> None:
    reference = datasets["EVT"]
    for product, dataset in datasets.items():
        if (
            dataset.crs != reference.crs
            or dataset.shape != reference.shape
            or not dataset.transform.almost_equals(reference.transform)
        ):
            raise ValueError(
                f"LANDFIRE {product} export is not aligned with EVT."
            )


def conditional_neighborhood_mean(
    numerator: np.ndarray,
    lifeform_area: np.ndarray,
    total_coverage: np.ndarray,
    kernel: np.ndarray,
    *,
    minimum_coverage: float,
    minimum_lifeform_fraction: float,
) -> np.ndarray:
    summed_numerator = fftconvolve(numerator, kernel, mode="same")
    summed_lifeform = fftconvolve(lifeform_area, kernel, mode="same")
    summed_total = fftconvolve(total_coverage, kernel, mode="same")
    coverage = summed_total / float(kernel.sum())
    lifeform_fraction = summed_lifeform / np.maximum(summed_total, 1e-12)
    supported = (
        (coverage >= minimum_coverage)
        & (lifeform_fraction >= minimum_lifeform_fraction)
        & (summed_lifeform > 0)
    )
    values = np.full(numerator.shape, NODATA, dtype=np.float32)
    values[supported] = (
        summed_numerator[supported] / summed_lifeform[supported]
    ).astype(np.float32)
    return values


def band_id(
    variable: str,
    radius_m: int,
    release: str,
    statistic: str,
) -> str:
    return (
        f"availability__landfire__{variable}__{statistic}__"
        f"r{radius_m}__{release.lower()}"
    )


def expected_band_ids(release: str, radii: list[int]) -> list[str]:
    identifiers = [
        band_id(f"evt_{model_class}_fraction", radius, release, "mean")
        for model_class in MODEL_CLASSES
        for radius in radii
    ]
    identifiers.append(
        band_id(
            "source_coverage_fraction",
            min(radii),
            release,
            "value",
        )
    )
    for product in ("EVC", "EVH"):
        for lifeform in LIFE_FORMS:
            variable = (
                f"dominant_{lifeform}_cover_fraction_conditional"
                if product == "EVC"
                else f"dominant_{lifeform}_height_m_conditional"
            )
            identifiers.extend(
                band_id(variable, radius, release, "mean")
                for radius in radii
            )
    return identifiers


def _derive_evt(
    *,
    dataset: Any,
    release: str,
    lookup: dict[int, str | None],
    destination_transform: Any,
    destination_size: int,
    target_crs: str,
    radii: list[int],
    kernels: dict[int, np.ndarray],
    minimum_coverage: float,
    write_context: dict[str, Any],
) -> None:
    values, source_mask = _read_source(dataset)
    class_index = {name: index for index, name in enumerate(MODEL_CLASSES)}
    class_lut, minimum = _integer_lut(
        lookup,
        lambda model_class: (
            class_index[model_class] if model_class is not None else -1
        ),
        default=-1,
        dtype=np.int8,
    )
    mapped = _apply_lut(
        values,
        class_lut,
        minimum,
        default=-1,
        dtype=np.int8,
    )
    valid = source_mask & (mapped >= 0)
    coverage = reproject_average(
        valid.astype(np.float32),
        dataset.transform,
        dataset.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    coverage_by_radius: dict[int, np.ndarray] = {}
    for model_class, index in class_index.items():
        class_area = reproject_average(
            (valid & (mapped == index)).astype(np.float32),
            dataset.transform,
            dataset.crs,
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
            coverage_by_radius[radius] = neighborhood_coverage
            write_derived_tile(
                fraction,
                band_id(
                    f"evt_{model_class}_fraction",
                    radius,
                    release,
                    "mean",
                ),
                {
                    "source": "LANDFIRE",
                    "release": release,
                    "product": "EVT",
                    "model_class": model_class,
                    "radius_m": str(radius),
                    "derivation": "model_class_area_fraction",
                },
                **write_context,
            )
    local_radius = min(radii)
    write_derived_tile(
        np.clip(coverage_by_radius[local_radius], 0.0, 1.0),
        band_id(
            "source_coverage_fraction",
            local_radius,
            release,
            "value",
        ),
        {
            "source": "LANDFIRE",
            "release": release,
            "product": "EVT",
            "radius_m": str(local_radius),
            "derivation": "valid_modeled_source_area_fraction",
        },
        **write_context,
    )


def _derive_structure(
    *,
    dataset: Any,
    release: str,
    product: str,
    lookup: dict[int, dict[str, Any]],
    destination_transform: Any,
    destination_size: int,
    target_crs: str,
    radii: list[int],
    kernels: dict[int, np.ndarray],
    minimum_coverage: float,
    minimum_lifeform_fraction: float,
    write_context: dict[str, Any],
) -> None:
    values, source_mask = _read_source(dataset)
    known_lut, minimum = _integer_lut(
        lookup,
        lambda record: (
            record["numeric_value"] is not None
            or record["special_class"] is not None
        ),
        default=False,
        dtype=bool,
    )
    known = _apply_lut(
        values,
        known_lut,
        minimum,
        default=False,
        dtype=bool,
    )
    total_valid = source_mask & known
    total_coverage = reproject_average(
        total_valid.astype(np.float32),
        dataset.transform,
        dataset.crs,
        destination_transform,
        target_crs,
        destination_size,
    )
    for lifeform_index, lifeform in enumerate(LIFE_FORMS, start=1):
        lifeform_lut, lifeform_minimum = _integer_lut(
            lookup,
            lambda record: (
                lifeform_index
                if record["lifeform"] == lifeform
                and record["numeric_value"] is not None
                else 0
            ),
            default=0,
            dtype=np.int8,
        )
        numeric_lut, numeric_minimum = _integer_lut(
            lookup,
            lambda record: (
                record["numeric_value"]
                if record["lifeform"] == lifeform
                and record["numeric_value"] is not None
                else 0.0
            ),
            default=0.0,
            dtype=np.float32,
        )
        mapped_lifeform = _apply_lut(
            values,
            lifeform_lut,
            lifeform_minimum,
            default=0,
            dtype=np.int8,
        )
        mapped_numeric = _apply_lut(
            values,
            numeric_lut,
            numeric_minimum,
            default=0.0,
            dtype=np.float32,
        )
        lifeform_valid = total_valid & (mapped_lifeform == lifeform_index)
        lifeform_area = reproject_average(
            lifeform_valid.astype(np.float32),
            dataset.transform,
            dataset.crs,
            destination_transform,
            target_crs,
            destination_size,
        )
        scale = 0.01 if product == "EVC" else 1.0
        numerator = reproject_average(
            np.where(
                lifeform_valid,
                mapped_numeric * scale,
                0.0,
            ).astype(np.float32),
            dataset.transform,
            dataset.crs,
            destination_transform,
            target_crs,
            destination_size,
        )
        variable = (
            f"dominant_{lifeform}_cover_fraction_conditional"
            if product == "EVC"
            else f"dominant_{lifeform}_height_m_conditional"
        )
        for radius in radii:
            conditional = conditional_neighborhood_mean(
                numerator,
                lifeform_area,
                total_coverage,
                kernels[radius],
                minimum_coverage=minimum_coverage,
                minimum_lifeform_fraction=minimum_lifeform_fraction,
            )
            write_derived_tile(
                conditional,
                band_id(variable, radius, release, "mean"),
                {
                    "source": "LANDFIRE",
                    "release": release,
                    "product": product,
                    "lifeform": lifeform,
                    "radius_m": str(radius),
                    "minimum_lifeform_fraction": str(
                        minimum_lifeform_fraction
                    ),
                    "derivation": "conditional_dominant_lifeform_mean",
                },
                **write_context,
            )


def derive_landfire(
    *,
    plan: dict[str, Any],
    export_summary: dict[str, Any],
    crosswalk_summary: dict[str, Any],
    output_dir: Path,
    neighborhoods_m: list[int] | None = None,
    minimum_coverage: float = 0.8,
    minimum_lifeform_fraction: float = 0.01,
    overwrite: bool = False,
    write_vrt: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if not 0 < minimum_coverage <= 1:
        raise ValueError("LANDFIRE minimum coverage must be in (0, 1].")
    if not 0 < minimum_lifeform_fraction <= 1:
        raise ValueError(
            "LANDFIRE minimum lifeform fraction must be in (0, 1]."
        )
    tile_id, release, export_by_product = _validate_export_group(export_summary)
    grid = plan["grid"]
    plan_tile_by_id = {tile["tile_id"]: tile for tile in grid["tiles"]}
    if tile_id not in plan_tile_by_id:
        raise ValueError(f"LANDFIRE export tile is absent from plan: {tile_id}")
    tile = plan_tile_by_id[tile_id]
    radii = sorted(
        {
            int(round(value))
            for value in (neighborhoods_m or plan["neighborhoods_m"])
        }
    )
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("LANDFIRE neighborhood radii must be positive.")
    if float(export_summary.get("buffer_m", 0)) < max(radii):
        raise ValueError(
            "LANDFIRE raw export buffer is smaller than the largest neighborhood."
        )
    target_crs = str(grid["crs"])
    resolution = float(grid["resolution_m"])
    tile_cells = int(grid["tile_width_cells"])
    buffer_cells = max(1, int(math.ceil(max(radii) / resolution)))
    expanded_bounds, destination_transform, destination_size = expanded_tile_grid(
        tile,
        resolution,
        tile_cells,
        buffer_cells,
    )
    kernels = {
        radius: circular_kernel(radius, resolution)
        for radius in radii
    }
    aoi_geometry = load_plan_aoi(plan)
    aoi_all_touched = aoi_mask_all_touched(plan)
    min_x, _, _, max_y = (float(value) for value in tile["bounds_m"])
    tile_transform = from_origin(min_x, max_y, resolution, resolution)
    band_order = expected_band_ids(release, radii)
    inventories: dict[str, dict[str, Any]] = {
        identifier: {
            "schema_version": 1,
            "build_id": plan["build_id"],
            "band_id": identifier,
            "dtype": "float32",
            "nodata": NODATA,
            "resampling": "derived",
            "tiles": [],
        }
        for identifier in band_order
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
    with ExitStack() as stack:
        datasets = {
            product: stack.enter_context(
                rasterio.open(export_by_product[product]["output_path"])
            )
            for product in ("EVT", "EVC", "EVH")
        }
        _validate_aligned_sources(datasets)
        for product, dataset in datasets.items():
            if dataset.crs is None or dataset.count != 1:
                raise ValueError(
                    f"LANDFIRE {product} source must be one georeferenced band."
                )
        if progress:
            print(f"LANDFIRE derive {tile_id} {release}: EVT")
        _derive_evt(
            dataset=datasets["EVT"],
            release=release,
            lookup=_evt_lookup(crosswalk_summary, release),
            destination_transform=destination_transform,
            destination_size=destination_size,
            target_crs=target_crs,
            radii=radii,
            kernels=kernels,
            minimum_coverage=minimum_coverage,
            write_context=write_context,
        )
        for product in ("EVC", "EVH"):
            if progress:
                print(f"LANDFIRE derive {tile_id} {release}: {product}")
            _derive_structure(
                dataset=datasets[product],
                release=release,
                product=product,
                lookup=_structure_lookup(
                    crosswalk_summary,
                    release,
                    product,
                ),
                destination_transform=destination_transform,
                destination_size=destination_size,
                target_crs=target_crs,
                radii=radii,
                kernels=kernels,
                minimum_coverage=minimum_coverage,
                minimum_lifeform_fraction=minimum_lifeform_fraction,
                write_context=write_context,
            )
    ordered_inventories, inventory_paths = finalize_inventories(
        inventories,
        band_order,
        output_dir,
    )
    expected_band_count = len(band_order)
    if len(ordered_inventories) != expected_band_count:
        raise ValueError(
            "LANDFIRE output inventory is incomplete: "
            f"expected {expected_band_count}, found {len(ordered_inventories)}."
        )
    vrt_path = output_dir / f"landfire_{release.lower()}_{tile_id}.vrt"
    if write_vrt:
        write_logical_vrt(plan, ordered_inventories, vrt_path)
    cog_paths = [
        Path(tile_record["path"])
        for inventory in ordered_inventories
        for tile_record in inventory["tiles"]
    ]
    empty_band_ids = [
        inventory["band_id"]
        for inventory in ordered_inventories
        if not inventory["tiles"]
    ]
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": plan["build_id"],
        "source_id": "landfire",
        "release": release,
        "tile_id": tile_id,
        "neighborhoods_m": radii,
        "minimum_coverage": minimum_coverage,
        "minimum_lifeform_fraction": minimum_lifeform_fraction,
        "model_classes": list(MODEL_CLASSES),
        "lifeforms": list(LIFE_FORMS),
        "band_count": len(ordered_inventories),
        "expected_band_count": expected_band_count,
        "empty_band_count": len(empty_band_ids),
        "empty_band_ids": empty_band_ids,
        "derived_cog_count": len(cog_paths),
        "derived_cog_bytes": sum(path.stat().st_size for path in cog_paths),
        "elapsed_seconds": time.perf_counter() - started_at,
        "inventory_paths": inventory_paths,
        "logical_vrt": str(vrt_path.resolve()) if write_vrt else None,
        "definitions": {
            "evt_fraction": "portable EVT model-class area fraction",
            "evc_conditional": (
                "mean cover fraction among pixels mapped to the named "
                "dominant lifeform"
            ),
            "evh_conditional": (
                "mean height in meters among pixels mapped to the named "
                "dominant lifeform"
            ),
            "source_coverage": "valid modeled EVT area fraction",
        },
    }
    summary_path = output_dir / "landfire_derived_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
