"""Derive annual LANDFIRE disturbance fractions from validated exports."""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

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


def disturbance_band_id(year: int, radius_m: int) -> str:
    return (
        "availability__landfire__annual_disturbance_fraction__mean__"
        f"r{radius_m}__y{year}"
    )


def expected_disturbance_band_ids(
    years: list[int], radii: list[int]
) -> list[str]:
    return [
        disturbance_band_id(year, radius)
        for year in sorted(years)
        for radius in sorted(radii)
    ]


def _artifact_path(lookup_summary: dict[str, Any]) -> Path:
    matches = [
        Path(artifact["path"])
        for artifact in lookup_summary.get("artifacts", [])
        if artifact.get("id") == "disturbance_model_lookup"
    ]
    if len(matches) != 1:
        raise ValueError(
            "LANDFIRE disturbance lookup summary must contain one "
            "disturbance_model_lookup artifact."
        )
    return matches[0]


def _lookup_by_layer(
    lookup_summary: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    path = _artifact_path(lookup_summary)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["layer_name"], []).append(row)
    if not grouped:
        raise ValueError("LANDFIRE disturbance lookup is empty.")
    return grouped


def classify_source_values(
    values: np.ndarray,
    source_valid_mask: np.ndarray,
    lookup_rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    support_codes = [
        int(row["Value"])
        for row in lookup_rows
        if row["is_analysis_support"].casefold() == "true"
    ]
    event_codes = [
        int(row["Value"])
        for row in lookup_rows
        if row["is_disturbed"].casefold() == "true"
    ]
    if not support_codes or not event_codes:
        raise ValueError("Disturbance lookup lacks support or event codes.")
    support = source_valid_mask & np.isin(values, support_codes)
    disturbed = support & np.isin(values, event_codes)
    return support, disturbed


def _validate_export_group(
    export_summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    exports = export_summary.get("exports")
    if not isinstance(exports, list) or not exports:
        raise ValueError("LANDFIRE disturbance export summary has no exports.")
    tile_ids = {record.get("tile_id") for record in exports}
    if len(tile_ids) != 1:
        raise ValueError(
            "LANDFIRE disturbance derivation requires one tile per export summary."
        )
    if any(record.get("product") != "Dist" for record in exports):
        raise ValueError("LANDFIRE disturbance export contains non-Dist products.")
    years = [record.get("observation_year") for record in exports]
    if any(not isinstance(year, int) for year in years):
        raise ValueError("Every disturbance export requires observation_year.")
    if len(set(years)) != len(years):
        raise ValueError("LANDFIRE disturbance export has duplicate years.")
    return str(next(iter(tile_ids))), sorted(
        exports, key=lambda record: record["observation_year"]
    )


def derive_landfire_disturbance(
    *,
    plan: dict[str, Any],
    export_summary: dict[str, Any],
    lookup_summary: dict[str, Any],
    output_dir: Path,
    neighborhoods_m: list[int] | None = None,
    minimum_coverage: float = 0.8,
    overwrite: bool = False,
    write_vrt: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if not 0 < minimum_coverage <= 1:
        raise ValueError("LANDFIRE disturbance coverage must be in (0, 1].")
    tile_id, exports = _validate_export_group(export_summary)
    years = [int(record["observation_year"]) for record in exports]
    grid = plan["grid"]
    tile_by_id = {tile["tile_id"]: tile for tile in grid["tiles"]}
    if tile_id not in tile_by_id:
        raise ValueError(f"LANDFIRE disturbance tile is absent from plan: {tile_id}")
    tile = tile_by_id[tile_id]
    radii = sorted(
        {
            int(round(value))
            for value in (neighborhoods_m or plan["neighborhoods_m"])
        }
    )
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("LANDFIRE disturbance radii must be positive.")
    if float(export_summary.get("buffer_m", 0)) < max(radii):
        raise ValueError(
            "LANDFIRE disturbance export buffer is smaller than the largest radius."
        )

    target_crs = str(grid["crs"])
    resolution = float(grid["resolution_m"])
    tile_cells = int(grid["tile_width_cells"])
    buffer_cells = max(1, int(math.ceil(max(radii) / resolution)))
    _, destination_transform, destination_size = expanded_tile_grid(
        tile,
        resolution,
        tile_cells,
        buffer_cells,
    )
    kernels = {
        radius: circular_kernel(radius, resolution) for radius in radii
    }
    aoi_geometry = load_plan_aoi(plan)
    aoi_all_touched = aoi_mask_all_touched(plan)
    min_x, _, _, max_y = (float(value) for value in tile["bounds_m"])
    tile_transform = from_origin(min_x, max_y, resolution, resolution)
    band_order = expected_disturbance_band_ids(years, radii)
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
    lookup_by_layer = _lookup_by_layer(lookup_summary)
    source_statistics: list[dict[str, Any]] = []
    for index, record in enumerate(exports, start=1):
        year = int(record["observation_year"])
        layer_name = str(record["layer_name"])
        if layer_name not in lookup_by_layer:
            raise ValueError(
                f"Disturbance lookup has no rows for layer {layer_name}."
            )
        lookup_rows = lookup_by_layer[layer_name]
        lookup_years = {int(row["observation_year"]) for row in lookup_rows}
        if lookup_years != {year}:
            raise ValueError(
                f"Disturbance lookup year differs for layer {layer_name}."
            )
        if progress:
            print(
                f"LANDFIRE disturbance {index}/{len(exports)}: "
                f"{tile_id} year {year}"
            )
        with rasterio.open(record["output_path"]) as dataset:
            if dataset.count != 1 or dataset.crs is None:
                raise ValueError(
                    f"LANDFIRE disturbance source is not one georeferenced band: "
                    f"{record['output_path']}"
                )
            values = dataset.read(1, masked=True)
            source_valid = ~np.ma.getmaskarray(values)
            raw_values = np.asarray(values.data)
            support, disturbed = classify_source_values(
                raw_values,
                source_valid,
                lookup_rows,
            )
            support_area = reproject_average(
                support.astype(np.float32),
                dataset.transform,
                dataset.crs,
                destination_transform,
                target_crs,
                destination_size,
            )
            disturbed_area = reproject_average(
                disturbed.astype(np.float32),
                dataset.transform,
                dataset.crs,
                destination_transform,
                target_crs,
                destination_size,
            )
        source_statistics.append(
            {
                "year": year,
                "release": record["version"],
                "layer_name": layer_name,
                "source_valid_pixels": int(source_valid.sum()),
                "analysis_support_pixels": int(support.sum()),
                "disturbed_pixels": int(disturbed.sum()),
                "water_mask_pixels": int(
                    (
                        source_valid
                        & np.isin(
                            raw_values,
                            [
                                int(row["Value"])
                                for row in lookup_rows
                                if row["model_category"] == "water_mask"
                            ],
                        )
                    ).sum()
                ),
            }
        )
        for radius in radii:
            fraction, _ = neighborhood_ratio(
                disturbed_area,
                support_area,
                kernels[radius],
                minimum_coverage,
            )
            write_derived_tile(
                fraction,
                disturbance_band_id(year, radius),
                {
                    "source": "LANDFIRE",
                    "release": str(record["version"]),
                    "layer_name": layer_name,
                    "observation_year": str(year),
                    "radius_m": str(radius),
                    "minimum_coverage": str(minimum_coverage),
                    "derivation": (
                        "disturbed_area_over_mappable_terrestrial_support"
                    ),
                    "water_policy": "excluded_from_numerator_and_denominator",
                },
                **write_context,
            )

    ordered_inventories, inventory_paths = finalize_inventories(
        inventories,
        band_order,
        output_dir,
    )
    vrt_path = output_dir / f"landfire_disturbance_{tile_id}.vrt"
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
        "component": "annual_disturbance",
        "tile_id": tile_id,
        "disturbance_years": years,
        "neighborhoods_m": radii,
        "minimum_coverage": minimum_coverage,
        "band_count": len(ordered_inventories),
        "expected_band_count": len(band_order),
        "empty_band_count": len(empty_band_ids),
        "empty_band_ids": empty_band_ids,
        "derived_cog_count": len(cog_paths),
        "derived_cog_bytes": sum(path.stat().st_size for path in cog_paths),
        "elapsed_seconds": time.perf_counter() - started_at,
        "source_statistics": source_statistics,
        "inventory_paths": inventory_paths,
        "logical_vrt": str(vrt_path.resolve()) if write_vrt else None,
        "definitions": {
            "annual_disturbance_fraction": (
                "Official annual event-code area divided by mappable terrestrial "
                "support; background is zero and Water/fill are excluded."
            )
        },
    }
    summary_path = output_dir / "landfire_disturbance_derived_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
