"""Validate and plan a tiled, temporally versioned covariate build."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import box, mapping


SUPPORTED_CHANNELS = {"availability", "access", "evaluation"}
SUPPORTED_CADENCES = {
    "annual",
    "derived",
    "monthly",
    "periodic",
    "snapshot",
    "static",
}
SUPPORTED_TIME_AXES = {
    "release",
    "static",
    "year",
    "year_month",
    "year_season",
    "month_normal",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def require_positive_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive number.")
    return float(value)


def validate_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if registry.get("schema_version") != 1:
        raise ValueError("The source registry must use schema_version 1.")
    sources = registry.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("The source registry must contain a non-empty sources list.")

    indexed: dict[str, dict[str, Any]] = {}
    for index, source_value in enumerate(sources):
        source = require_mapping(source_value, f"sources[{index}]")
        source_id = require_nonempty_string(source.get("id"), f"sources[{index}].id")
        if source_id in indexed:
            raise ValueError(f"Duplicate source id in registry: {source_id}")
        channel = source.get("channel")
        if channel not in SUPPORTED_CHANNELS:
            raise ValueError(
                f"Source {source_id} has unsupported channel {channel!r}."
            )
        cadence = source.get("cadence")
        if cadence not in SUPPORTED_CADENCES:
            raise ValueError(
                f"Source {source_id} has unsupported cadence {cadence!r}."
            )
        require_nonempty_string(source.get("official_url"), f"{source_id}.official_url")
        products = source.get("planned_products", [])
        if not isinstance(products, list):
            raise ValueError(f"{source_id}.planned_products must be a list.")
        for product_index, product_value in enumerate(products):
            product = require_mapping(
                product_value,
                f"{source_id}.planned_products[{product_index}]",
            )
            require_nonempty_string(
                product.get("id"),
                f"{source_id}.planned_products[{product_index}].id",
            )
            bands = product.get("bands")
            if not isinstance(bands, int) or isinstance(bands, bool) or bands < 0:
                raise ValueError(
                    f"{source_id}.{product.get('id')}.bands must be a nonnegative integer."
                )
            if product.get("time_axis") not in SUPPORTED_TIME_AXES:
                raise ValueError(
                    f"{source_id}.{product.get('id')} has unsupported time_axis."
                )
            if product.get("spatial_scales") not in {"single", "neighborhoods"}:
                raise ValueError(
                    f"{source_id}.{product.get('id')} has unsupported spatial_scales."
                )
        indexed[source_id] = source
    return indexed


def normalize_source_requests(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError("config.sources must be a non-empty list.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if isinstance(value, str):
            request = {"id": value, "enabled": True}
        else:
            request = require_mapping(value, f"config.sources[{index}]").copy()
        source_id = require_nonempty_string(request.get("id"), f"config.sources[{index}].id")
        if source_id in seen:
            raise ValueError(f"Duplicate source id in config: {source_id}")
        if not isinstance(request.get("enabled", True), bool):
            raise ValueError(f"config.sources[{index}].enabled must be boolean.")
        request["id"] = source_id
        request.setdefault("enabled", True)
        normalized.append(request)
        seen.add(source_id)
    return normalized


def validate_config(
    config: dict[str, Any], registry_sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("The build config must use schema_version 1.")
    require_nonempty_string(config.get("build_id"), "config.build_id")

    aoi = require_mapping(config.get("aoi"), "config.aoi")
    require_nonempty_string(aoi.get("path"), "config.aoi.path")

    grid = require_mapping(config.get("grid"), "config.grid")
    crs_text = require_nonempty_string(grid.get("crs"), "config.grid.crs")
    crs = CRS.from_user_input(crs_text)
    if not crs.is_projected:
        raise ValueError("config.grid.crs must be projected.")
    axis_units = {axis.unit_name.lower() for axis in crs.axis_info if axis.unit_name}
    if axis_units and not all("metre" in unit or "meter" in unit for unit in axis_units):
        raise ValueError("config.grid.crs must use meter units.")
    resolution = require_positive_number(grid.get("resolution_m"), "config.grid.resolution_m")
    tile_size = require_positive_number(grid.get("tile_size_m"), "config.grid.tile_size_m")
    if not math.isclose(tile_size / resolution, round(tile_size / resolution)):
        raise ValueError("config.grid.tile_size_m must be divisible by resolution_m.")
    require_mapping(grid.get("origin_m"), "config.grid.origin_m")

    temporal = require_mapping(config.get("temporal"), "config.temporal")
    start_year = temporal.get("start_year")
    end_year = temporal.get("end_year")
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("temporal start_year and end_year must be integers.")
    if start_year > end_year:
        raise ValueError("temporal.start_year must not exceed end_year.")
    months = temporal.get("months")
    if (
        not isinstance(months, list)
        or not months
        or any(not isinstance(month, int) or month < 1 or month > 12 for month in months)
        or len(set(months)) != len(months)
    ):
        raise ValueError("temporal.months must contain unique integer months 1-12.")
    seasons = require_mapping(temporal.get("seasons"), "config.temporal.seasons")
    if not seasons:
        raise ValueError("config.temporal.seasons must not be empty.")
    for season_name, season_months in seasons.items():
        require_nonempty_string(season_name, "season name")
        if (
            not isinstance(season_months, list)
            or not season_months
            or any(month not in months for month in season_months)
        ):
            raise ValueError(
                f"Season {season_name} must contain months declared in temporal.months."
            )
    if temporal.get("selection_rule") != "latest_not_after":
        raise ValueError(
            "Phase 1 supports only temporal.selection_rule='latest_not_after'."
        )

    neighborhoods = config.get("neighborhoods_m")
    if (
        not isinstance(neighborhoods, list)
        or not neighborhoods
        or any(not isinstance(value, (int, float)) or value <= 0 for value in neighborhoods)
    ):
        raise ValueError("config.neighborhoods_m must be a non-empty positive list.")
    if len(set(neighborhoods)) != len(neighborhoods):
        raise ValueError("config.neighborhoods_m must not contain duplicates.")
    for value in neighborhoods:
        if value < resolution or not math.isclose(value / resolution, round(value / resolution)):
            raise ValueError(
                "Each neighborhood must be at least one cell and divisible by resolution_m."
            )

    source_requests = normalize_source_requests(config.get("sources"))
    missing = sorted(
        request["id"]
        for request in source_requests
        if request["id"] not in registry_sources
    )
    if missing:
        raise ValueError(f"Config references unknown source ids: {', '.join(missing)}")

    output = require_mapping(config.get("output"), "config.output")
    require_nonempty_string(output.get("root"), "config.output.root")
    require_nonempty_string(output.get("logical_raster"), "config.output.logical_raster")
    require_nonempty_string(output.get("manifest"), "config.output.manifest")

    validated = config.copy()
    validated["sources"] = source_requests
    return validated


def snap_bounds(
    bounds: tuple[float, float, float, float],
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bounds
    return (
        origin_x + math.floor((min_x - origin_x) / resolution) * resolution,
        origin_y + math.floor((min_y - origin_y) / resolution) * resolution,
        origin_x + math.ceil((max_x - origin_x) / resolution) * resolution,
        origin_y + math.ceil((max_y - origin_y) / resolution) * resolution,
    )


def encode_tile_index(prefix: str, value: int) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{prefix}{sign}{abs(value):04d}"


def plan_tiles(
    geometry: Any,
    resolution: float,
    tile_size: float,
    origin_x: float,
    origin_y: float,
) -> list[dict[str, Any]]:
    tile_cells = int(round(tile_size / resolution))
    min_x, min_y, max_x, max_y = geometry.bounds
    min_ix = math.floor((min_x - origin_x) / tile_size)
    max_ix = math.ceil((max_x - origin_x) / tile_size) - 1
    min_iy = math.floor((min_y - origin_y) / tile_size)
    max_iy = math.ceil((max_y - origin_y) / tile_size) - 1

    tiles: list[dict[str, Any]] = []
    for iy in range(min_iy, max_iy + 1):
        tile_min_y = origin_y + iy * tile_size
        tile_max_y = tile_min_y + tile_size
        for ix in range(min_ix, max_ix + 1):
            tile_min_x = origin_x + ix * tile_size
            tile_max_x = tile_min_x + tile_size
            tile_geometry = box(tile_min_x, tile_min_y, tile_max_x, tile_max_y)
            if not geometry.intersects(tile_geometry):
                continue
            clipped = geometry.intersection(tile_geometry)
            if clipped.is_empty:
                continue
            active = rasterize(
                [(mapping(clipped), 1)],
                out_shape=(tile_cells, tile_cells),
                transform=from_origin(
                    tile_min_x,
                    tile_max_y,
                    resolution,
                    resolution,
                ),
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            )
            active_cells = int(active.sum())
            if active_cells == 0:
                continue
            tiles.append(
                {
                    "tile_id": (
                        f"{encode_tile_index('x', ix)}_"
                        f"{encode_tile_index('y', iy)}"
                    ),
                    "x_index": ix,
                    "y_index": iy,
                    "bounds_m": [tile_min_x, tile_min_y, tile_max_x, tile_max_y],
                    "width": tile_cells,
                    "height": tile_cells,
                    "active_cells_center_rule": active_cells,
                }
            )
    return sorted(tiles, key=lambda value: (value["y_index"], value["x_index"]))


def time_period_count(time_axis: str, temporal: dict[str, Any]) -> int:
    years = temporal["end_year"] - temporal["start_year"] + 1
    if time_axis == "year":
        return years
    if time_axis == "year_month":
        return years * len(temporal["months"])
    if time_axis == "year_season":
        return years * len(temporal["seasons"])
    if time_axis == "month_normal":
        return len(temporal["months"])
    if time_axis in {"static", "release"}:
        return 1
    raise ValueError(f"Unsupported time axis: {time_axis}")


def expected_product_bands(
    product: dict[str, Any],
    temporal: dict[str, Any],
    neighborhoods: list[float],
) -> int:
    scale_count = len(neighborhoods) if product["spatial_scales"] == "neighborhoods" else 1
    return product["bands"] * scale_count * time_period_count(
        product["time_axis"], temporal
    )


def requested_periods(cadence: str, temporal: dict[str, Any]) -> dict[str, Any]:
    years = list(range(temporal["start_year"], temporal["end_year"] + 1))
    if cadence == "annual":
        return {"years": years}
    if cadence == "monthly":
        return {
            "years": years,
            "months": temporal["months"],
            "seasons": list(temporal["seasons"]),
        }
    if cadence == "static":
        return {"releases": ["static"]}
    if cadence == "derived":
        return {"releases": ["derived_from_dependency_releases"]}
    return {"releases": ["resolve_from_official_source_catalog"]}


def load_aoi(config: dict[str, Any], target_crs: str) -> tuple[Any, Path, str | None]:
    aoi_config = config["aoi"]
    path = Path(aoi_config["path"])
    if not path.exists():
        raise FileNotFoundError(f"AOI does not exist: {path}")
    layer = aoi_config.get("layer")
    frame = gpd.read_file(path, layer=layer)
    if frame.empty:
        raise ValueError(f"AOI has no features: {path}")
    if frame.crs is None:
        raise ValueError(f"AOI has no CRS: {path}")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.empty:
        raise ValueError(f"AOI has no valid geometries: {path}")
    frame = frame.to_crs(target_crs)
    geometry = frame.geometry.make_valid().union_all()
    if geometry.is_empty:
        raise ValueError(f"AOI union is empty: {path}")
    return geometry, path.resolve(), layer


def build_plan(
    config: dict[str, Any],
    registry: dict[str, Any],
    config_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    registry_sources = validate_registry(registry)
    config = validate_config(config, registry_sources)
    grid = config["grid"]
    temporal = config["temporal"]
    neighborhoods = [float(value) for value in config["neighborhoods_m"]]
    resolution = float(grid["resolution_m"])
    tile_size = float(grid["tile_size_m"])
    origin_x = float(grid["origin_m"].get("x", 0.0))
    origin_y = float(grid["origin_m"].get("y", 0.0))

    geometry, aoi_path, aoi_layer = load_aoi(config, grid["crs"])
    snapped = snap_bounds(
        tuple(float(value) for value in geometry.bounds),
        resolution,
        origin_x,
        origin_y,
    )
    width = int(round((snapped[2] - snapped[0]) / resolution))
    height = int(round((snapped[3] - snapped[1]) / resolution))
    tiles = plan_tiles(
        geometry,
        resolution,
        tile_size,
        origin_x,
        origin_y,
    )

    source_plans: list[dict[str, Any]] = []
    logical_band_count = 0
    enabled_requests = [request for request in config["sources"] if request["enabled"]]
    for request in enabled_requests:
        source = registry_sources[request["id"]]
        products = []
        source_band_count = 0
        for product in source.get("planned_products", []):
            count = expected_product_bands(product, temporal, neighborhoods)
            source_band_count += count
            products.append(
                {
                    "id": product["id"],
                    "time_axis": product["time_axis"],
                    "spatial_scales": product["spatial_scales"],
                    "base_bands": product["bands"],
                    "estimated_logical_bands": count,
                }
            )
        logical_band_count += source_band_count
        source_plans.append(
            {
                "id": source["id"],
                "name": source["name"],
                "channel": source["channel"],
                "scope": source["scope"],
                "cadence": source["cadence"],
                "adapter_status": source["adapter_status"],
                "official_url": source["official_url"],
                "requested_periods": requested_periods(source["cadence"], temporal),
                "estimated_logical_bands": source_band_count,
                "products": products,
                "config_overrides": {
                    key: value
                    for key, value in request.items()
                    if key not in {"id", "enabled"}
                },
            }
        )

    build_dir = Path(config["output"]["root"]) / config["build_id"]
    active_cells = sum(tile["active_cells_center_rule"] for tile in tiles)
    bytes_per_value = np.dtype(grid.get("dtype", "float32")).itemsize
    active_uncompressed_bytes = active_cells * logical_band_count * bytes_per_value
    bounding_uncompressed_bytes = width * height * logical_band_count * bytes_per_value
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": config["build_id"],
        "description": config.get("description"),
        "inputs": {
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "source_registry_path": str(registry_path.resolve()),
            "source_registry_sha256": sha256_file(registry_path),
            "source_registry_version": registry.get("registry_version"),
        },
        "aoi": {
            "name": config["aoi"].get("name"),
            "path": str(aoi_path),
            "layer": aoi_layer,
            "area_sq_km": geometry.area / 1_000_000.0,
            "bounds_m": [float(value) for value in geometry.bounds],
        },
        "grid": {
            "crs": CRS.from_user_input(grid["crs"]).to_string(),
            "resolution_m": resolution,
            "tile_size_m": tile_size,
            "tile_width_cells": int(round(tile_size / resolution)),
            "origin_m": {"x": origin_x, "y": origin_y},
            "snapped_bounds_m": list(snapped),
            "bounding_width_cells": width,
            "bounding_height_cells": height,
            "bounding_cells": width * height,
            "active_cells_center_rule": active_cells,
            "active_cell_area_sq_km": active_cells * resolution * resolution / 1_000_000.0,
            "tile_count": len(tiles),
            "tiles": tiles,
        },
        "temporal": temporal,
        "neighborhoods_m": neighborhoods,
        "sources": source_plans,
        "source_count": len(source_plans),
        "estimated_logical_band_count": logical_band_count,
        "band_count_note": (
            "Planning estimate from registry feature groups. Final count is fixed only "
            "after source releases and category crosswalks are resolved."
        ),
        "storage_estimate": {
            "dtype": grid.get("dtype", "float32"),
            "bytes_per_value": bytes_per_value,
            "active_cells_uncompressed_gib": active_uncompressed_bytes / (1024**3),
            "bounding_grid_uncompressed_gib": bounding_uncompressed_bytes / (1024**3),
            "note": (
                "Uncompressed value-array estimate only; physical source tiles, masks, "
                "overviews, indexes, compression, and temporary files are not included."
            ),
        },
        "outputs": {
            "build_dir": str(build_dir),
            "logical_raster": str(build_dir / config["output"]["logical_raster"]),
            "manifest": str(build_dir / config["output"]["manifest"]),
            "materialized_export": (
                str(build_dir / "exports" / config["output"]["materialized_export"])
                if config["output"].get("materialized_export")
                else None
            ),
        },
        "planner_warnings": [
            "All source adapters are currently planned; this command does not download data.",
            "Snapshot and periodic releases still require source-catalog resolution.",
            "The estimated band count includes both monthly and seasonal climate views.",
            "Use named export profiles; the all-band VRT is the complete logical interface.",
        ],
    }


def write_plan(plan: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def print_plan_summary(plan: dict[str, Any], output_path: Path | None) -> None:
    grid = plan["grid"]
    print(f"Covariate build plan: {plan['build_id']}")
    print(f"AOI: {plan['aoi']['name'] or plan['aoi']['path']}")
    print(
        "Grid: "
        f"{grid['crs']} at {grid['resolution_m']:g} m; "
        f"{grid['bounding_width_cells']:,} x {grid['bounding_height_cells']:,} bounding cells"
    )
    print(
        f"AOI cells: {grid['active_cells_center_rule']:,}; "
        f"tiles: {grid['tile_count']:,}"
    )
    print(
        f"Sources: {plan['source_count']}; "
        f"estimated logical bands: {plan['estimated_logical_band_count']:,}"
    )
    print(
        "Uncompressed float stack estimate: "
        f"{plan['storage_estimate']['active_cells_uncompressed_gib']:.1f} GiB over AOI cells; "
        f"{plan['storage_estimate']['bounding_grid_uncompressed_gib']:.1f} GiB over bounding grid"
    )
    print("Source plan:")
    for source in plan["sources"]:
        print(
            f"  {source['id']:<24} {source['channel']:<12} "
            f"{source['cadence']:<9} bands~{source['estimated_logical_bands']:,} "
            f"[{source['adapter_status']}]"
        )
    if output_path is None:
        print("Plan was not written (--no-write).")
    else:
        print(f"Wrote build plan to {output_path}")
