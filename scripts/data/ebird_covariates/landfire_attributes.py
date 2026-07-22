"""Catalog release-specific LANDFIRE raster attribute tables."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


REQUIRED_FIELDS = {
    "EVT": {"Value", "EVT_NAME", "EVT_LF", "EVT_PHYS"},
    "EVC": {"Value", "CLASSNAMES"},
    "EVH": {"Value", "CLASSNAMES"},
}


def safe_layer_name(value: str) -> str:
    if not re.fullmatch(r"LF\d{4}_[A-Za-z0-9]+", value):
        raise ValueError(f"Unsafe LANDFIRE layer name: {value!r}")
    return value


def select_vegetation_layers(
    catalog: dict[str, Any], layer_names: list[str] | None = None
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


def _attribute_table_request(
    layer: dict[str, Any],
    *,
    timeout: float,
    session: requests.Session | None,
) -> tuple[bytes, dict[str, Any], str, str]:
    layer_name = safe_layer_name(layer["layerName"])
    image_server_url = layer.get("image_server_url")
    if not isinstance(image_server_url, str) or not image_server_url.startswith(
        "https://"
    ):
        raise ValueError(f"LANDFIRE layer {layer_name} has no HTTPS ImageServer URL.")
    endpoint = f"{image_server_url.rstrip('/')}/rasterAttributeTable"
    raster_function = f"{layer_name}_CONUS"
    requester = session or requests.Session()
    response = requester.get(
        endpoint,
        params={
            "f": "json",
            "renderingRule": json.dumps(
                {"rasterFunction": raster_function},
                separators=(",", ":"),
            ),
        },
        timeout=timeout,
        headers={"User-Agent": "ebird-covariate-pipeline/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("error"):
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise ValueError(
            f"LANDFIRE raster attribute table failed for {layer_name}: {detail}"
        )
    return response.content, payload, endpoint, raster_function


def _validate_attribute_table(
    layer: dict[str, Any], payload: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    layer_name = safe_layer_name(layer["layerName"])
    product = layer["acronym"]
    field_records = payload.get("fields")
    feature_records = payload.get("features")
    if not isinstance(field_records, list) or not isinstance(feature_records, list):
        raise ValueError(
            f"LANDFIRE attribute table {layer_name} lacks fields or features."
        )
    field_names = [
        field["name"]
        for field in field_records
        if isinstance(field, dict) and isinstance(field.get("name"), str)
    ]
    missing = sorted(REQUIRED_FIELDS.get(product, {"Value"}) - set(field_names))
    if missing:
        raise ValueError(
            f"LANDFIRE attribute table {layer_name} lacks required fields: "
            + ", ".join(missing)
        )
    rows = [
        feature["attributes"]
        for feature in feature_records
        if isinstance(feature, dict) and isinstance(feature.get("attributes"), dict)
    ]
    if len(rows) != len(feature_records) or not rows:
        raise ValueError(
            f"LANDFIRE attribute table {layer_name} has invalid or empty features."
        )
    values = [row.get("Value") for row in rows]
    if any(value is None for value in values) or len(set(values)) != len(values):
        raise ValueError(
            f"LANDFIRE attribute table {layer_name} has null or duplicate Value keys."
        )
    return field_names, rows


def _write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise FileExistsError(f"Existing output differs from source: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _csv_bytes(field_names: list[str], rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def extract_attribute_tables(
    catalog: dict[str, Any],
    output_dir: Path,
    layer_names: list[str] | None = None,
    timeout: float = 60.0,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    layers = select_vegetation_layers(catalog, layer_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for layer in layers:
        layer_name = safe_layer_name(layer["layerName"])
        raw_json, payload, endpoint, raster_function = _attribute_table_request(
            layer,
            timeout=timeout,
            session=session,
        )
        field_names, rows = _validate_attribute_table(layer, payload)
        json_path = output_dir / f"{layer_name}.attribute_table.json"
        csv_path = output_dir / f"{layer_name}.attribute_table.csv"
        csv_payload = _csv_bytes(field_names, rows)
        _write_bytes(json_path, raw_json, overwrite)
        _write_bytes(csv_path, csv_payload, overwrite)
        records.append(
            {
                "layer_name": layer_name,
                "version": layer["version"],
                "product": layer["acronym"],
                "attribute_table_url": endpoint,
                "raster_function": raster_function,
                "row_count": len(rows),
                "field_count": len(field_names),
                "field_names": field_names,
                "value_min": min(row["Value"] for row in rows),
                "value_max": max(row["Value"] for row in rows),
                "raw_json_bytes": len(raw_json),
                "raw_json_sha256": hashlib.sha256(raw_json).hexdigest(),
                "csv_bytes": len(csv_payload),
                "csv_sha256": hashlib.sha256(csv_payload).hexdigest(),
                "json_output_path": str(json_path.resolve()),
                "csv_output_path": str(csv_path.resolve()),
                "checks": {
                    "nonempty": True,
                    "required_fields_present": True,
                    "value_keys_nonnull_unique": True,
                },
            }
        )
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "landfire",
        "acquisition_method": "official_imageserver_raster_attribute_table",
        "catalog_snapshot_sha256": catalog.get("products_api_snapshot_sha256"),
        "table_count": len(records),
        "tables": records,
        "total_rows": sum(record["row_count"] for record in records),
        "total_response_bytes": sum(record["raw_json_bytes"] for record in records),
        "notes": [
            "Tables use each service's named raster function.",
            "No full CONUS raster archive was downloaded.",
        ],
    }


def write_attribute_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
