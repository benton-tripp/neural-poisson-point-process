"""Export and validate bounded raw LANDFIRE rasters from official ImageServers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests


EXPECTED_WKID = 5070


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Unsafe LANDFIRE identifier: {value!r}")
    return value


def select_layers(
    catalog: dict[str, Any], layer_names: list[str] | None
) -> list[dict[str, Any]]:
    layers = [
        layer
        for layer in catalog.get("layers", [])
        if layer.get("role") == "vegetation_release"
    ]
    requested = set(layer_names or [])
    if requested:
        known = {layer["layerName"] for layer in layers}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(
                "Requested LANDFIRE layers are absent from the catalog: "
                + ", ".join(unknown)
            )
        layers = [layer for layer in layers if layer["layerName"] in requested]
    if not layers:
        raise ValueError("No LANDFIRE vegetation layers were selected.")
    return layers


def select_tiles(
    plan: dict[str, Any], tile_ids: list[str] | None
) -> list[dict[str, Any]]:
    tiles = list(plan.get("grid", {}).get("tiles", []))
    requested = set(tile_ids or [])
    if requested:
        known = {tile["tile_id"] for tile in tiles}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(
                "Requested LANDFIRE tiles are absent from the plan: "
                + ", ".join(unknown)
            )
        tiles = [tile for tile in tiles if tile["tile_id"] in requested]
    if not tiles:
        raise ValueError("No LANDFIRE plan tiles were selected.")
    return tiles


def snap_bounds_to_source_grid(
    bounds: list[float],
    extent: dict[str, Any],
    pixel_size_m: float,
) -> tuple[list[float], int, int]:
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    origin_x = float(extent["xmin"])
    origin_y = float(extent["ymax"])
    col_start = math.floor((min_x - origin_x) / pixel_size_m)
    col_stop = math.ceil((max_x - origin_x) / pixel_size_m)
    row_start = math.floor((origin_y - max_y) / pixel_size_m)
    row_stop = math.ceil((origin_y - min_y) / pixel_size_m)
    width = col_stop - col_start
    height = row_stop - row_start
    if width <= 0 or height <= 0:
        raise ValueError("LANDFIRE export bounds produce an empty source window.")
    snapped = [
        origin_x + col_start * pixel_size_m,
        origin_y - row_stop * pixel_size_m,
        origin_x + col_stop * pixel_size_m,
        origin_y - row_start * pixel_size_m,
    ]
    return snapped, width, height


def _crosswalk_values(
    crosswalk_summary: dict[str, Any],
) -> dict[tuple[str, str], set[int]]:
    artifact_by_id = {
        artifact["id"]: artifact
        for artifact in crosswalk_summary.get("artifacts", [])
    }
    artifact_ids = {
        "EVT": "evt_model_crosswalk",
        "EVC": "evc_model_lookup",
        "EVH": "evh_model_lookup",
    }
    values: dict[tuple[str, str], set[int]] = {}
    for product, artifact_id in artifact_ids.items():
        artifact = artifact_by_id.get(artifact_id)
        if not artifact:
            raise ValueError(f"LANDFIRE crosswalk lacks {artifact_id}.")
        path = Path(artifact["path"])
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                values.setdefault((row["release"], product), set()).add(
                    int(row["Value"])
                )
    return values


def _export_request(
    layer: dict[str, Any],
    bounds: list[float],
    width: int,
    height: int,
) -> tuple[str, dict[str, str]]:
    image_server_url = layer.get("image_server_url")
    if not isinstance(image_server_url, str) or not image_server_url.startswith(
        "https://"
    ):
        raise ValueError(f"LANDFIRE layer {layer['layerName']} has no ImageServer.")
    endpoint = f"{image_server_url.rstrip('/')}/exportImage"
    params = {
        "bbox": ",".join(f"{value:.6f}" for value in bounds),
        "bboxSR": str(EXPECTED_WKID),
        "imageSR": str(EXPECTED_WKID),
        "size": f"{width},{height}",
        "format": "tiff",
        "pixelType": "UNKNOWN",
        "interpolation": "RSP_NearestNeighbor",
        "renderingRule": json.dumps(
            {"rasterFunction": f"{layer['layerName']}_CONUS"},
            separators=(",", ":"),
        ),
        "f": "image",
    }
    return endpoint, params


def _download(
    endpoint: str,
    params: dict[str, str],
    output_path: Path,
    *,
    timeout: float,
    session: requests.Session | None,
) -> None:
    requester = session or requests.Session()
    response = requester.get(
        endpoint,
        params=params,
        timeout=timeout,
        stream=True,
        headers={"User-Agent": "ebird-covariate-pipeline/1.0"},
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").lower()
    if "tiff" not in content_type and "octet-stream" not in content_type:
        preview = response.content[:500].decode("utf-8", errors="replace")
        raise ValueError(
            f"LANDFIRE export did not return TIFF content: {preview}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".download.tif")
    with temporary.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                stream.write(chunk)
    temporary.replace(output_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(
    path: Path,
    *,
    expected_bounds: list[float],
    expected_width: int,
    expected_height: int,
    expected_values: set[int],
) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        checks = {
            "one_band": dataset.count == 1,
            "epsg_5070": dataset.crs is not None
            and dataset.crs.to_epsg() == EXPECTED_WKID,
            "expected_dimensions": (
                dataset.width == expected_width
                and dataset.height == expected_height
            ),
            "nearest_source_grid_30m": (
                math.isclose(dataset.transform.a, 30.0, abs_tol=1e-6)
                and math.isclose(dataset.transform.e, -30.0, abs_tol=1e-6)
            ),
            "expected_bounds": all(
                math.isclose(actual, expected, abs_tol=1e-4)
                for actual, expected in zip(
                    dataset.bounds,
                    expected_bounds,
                    strict=True,
                )
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise ValueError(
                f"LANDFIRE export {path} failed raster checks: "
                + ", ".join(failed)
            )
        values = dataset.read(1, masked=True)
        valid = ~np.ma.getmaskarray(values)
        unique_values = {
            int(value)
            for value in np.unique(np.asarray(values.data)[valid])
        }
        unknown = sorted(unique_values - expected_values)
        checks["all_values_in_release_lookup"] = not unknown
        if unknown:
            raise ValueError(
                f"LANDFIRE export {path} has values absent from its lookup: "
                + ", ".join(str(value) for value in unknown[:20])
            )
        valid_cells = int(valid.sum())
        return {
            "width": dataset.width,
            "height": dataset.height,
            "dtype": dataset.dtypes[0],
            "crs": dataset.crs.to_string(),
            "transform": list(dataset.transform)[:6],
            "bounds": list(dataset.bounds),
            "nodata": dataset.nodata,
            "valid_cells": valid_cells,
            "coverage_fraction": valid_cells / valid.size,
            "unique_value_count": len(unique_values),
            "value_min": min(unique_values) if unique_values else None,
            "value_max": max(unique_values) if unique_values else None,
            "checks": checks,
        }


def export_landfire_tiles(
    *,
    plan: dict[str, Any],
    catalog: dict[str, Any],
    crosswalk_summary: dict[str, Any],
    output_dir: Path,
    tile_ids: list[str] | None,
    layer_names: list[str] | None,
    buffer_m: float = 5000.0,
    timeout: float = 300.0,
    overwrite: bool = False,
    session: requests.Session | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if buffer_m < 0:
        raise ValueError("LANDFIRE export buffer must be nonnegative.")
    if plan.get("grid", {}).get("crs") != f"EPSG:{EXPECTED_WKID}":
        raise ValueError("LANDFIRE export currently requires an EPSG:5070 plan.")
    layers = select_layers(catalog, layer_names)
    tiles = select_tiles(plan, tile_ids)
    lookup_values = _crosswalk_values(crosswalk_summary)
    records: list[dict[str, Any]] = []
    for tile_index, tile in enumerate(tiles, start=1):
        tile_id = _safe_id(tile["tile_id"])
        tile_bounds = [float(value) for value in tile["bounds_m"]]
        expanded_bounds = [
            tile_bounds[0] - buffer_m,
            tile_bounds[1] - buffer_m,
            tile_bounds[2] + buffer_m,
            tile_bounds[3] + buffer_m,
        ]
        if progress:
            print(f"LANDFIRE export tile {tile_index}/{len(tiles)}: {tile_id}")
        for layer in layers:
            layer_name = _safe_id(layer["layerName"])
            pixel_size = float(layer["pixel_size_m"][0])
            snapped_bounds, width, height = snap_bounds_to_source_grid(
                expanded_bounds,
                layer["extent"],
                pixel_size,
            )
            if width > int(layer["max_image_width"]) or height > int(
                layer["max_image_height"]
            ):
                raise ValueError(
                    f"LANDFIRE export {layer_name}:{tile_id} exceeds service limits."
                )
            key = (layer["version"], layer["acronym"])
            if key not in lookup_values:
                raise ValueError(
                    f"LANDFIRE crosswalk has no values for {key[0]}:{key[1]}."
                )
            output_path = output_dir / "tiles" / tile_id / f"{layer_name}.tif"
            endpoint, params = _export_request(
                layer,
                snapped_bounds,
                width,
                height,
            )
            reused = output_path.exists() and not overwrite
            if not reused:
                _download(
                    endpoint,
                    params,
                    output_path,
                    timeout=timeout,
                    session=session,
                )
            validation = validate_export(
                output_path,
                expected_bounds=snapped_bounds,
                expected_width=width,
                expected_height=height,
                expected_values=lookup_values[key],
            )
            records.append(
                {
                    "tile_id": tile_id,
                    "layer_name": layer_name,
                    "version": layer["version"],
                    "product": layer["acronym"],
                    "source_url": endpoint,
                    "request_parameters": params,
                    "tile_bounds_m": tile_bounds,
                    "buffer_m": buffer_m,
                    "export_bounds_m": snapped_bounds,
                    "output_path": str(output_path.resolve()),
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                    "reused": reused,
                    **validation,
                }
            )
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_id": plan["build_id"],
        "source_id": "landfire",
        "acquisition_method": "official_imageserver_export_image",
        "tile_count": len(tiles),
        "layer_count": len(layers),
        "export_count": len(records),
        "buffer_m": buffer_m,
        "exports": records,
        "total_bytes": sum(record["bytes"] for record in records),
        "all_checks_passed": all(
            all(record["checks"].values()) for record in records
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "landfire_export_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
