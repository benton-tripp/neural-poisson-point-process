"""Build portable model lookups from release-specific LANDFIRE class tables."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_CLASSES = (
    "forest_tree",
    "shrub",
    "herbaceous",
    "riparian",
    "agriculture",
    "developed",
    "sparse_barren",
    "open_water",
    "snow_ice",
)

EVT_PHYS_CLASS = {
    "Fill-NoData": None,
    "Open Water": "open_water",
    "Snow-Ice": "snow_ice",
    "Quarries-Strip Mines-Gravel Pits-Well and Wind Pads": "sparse_barren",
    "Developed-Low Intensity": "developed",
    "Developed-Medium Intensity": "developed",
    "Developed-High Intensity": "developed",
    "Developed-Roads": "developed",
    "Riparian": "riparian",
    "Conifer": "forest_tree",
    "Hardwood": "forest_tree",
    "Shrubland": "shrub",
    "Grassland": "herbaceous",
    "Sparsely Vegetated": "sparse_barren",
    "Developed": "developed",
    "Conifer-Hardwood": "forest_tree",
    "Agricultural": "agriculture",
    "Exotic Herbaceous": "herbaceous",
}

SPECIAL_STRUCTURE_CLASS = {
    "Fill-NoData": None,
    "Open Water": "open_water",
    "Snow/Ice": "snow_ice",
    "Developed-Upland Deciduous Forest": "developed_tree",
    "Developed-Upland Evergreen Forest": "developed_tree",
    "Developed-Upland Mixed Forest": "developed_tree",
    "Developed-Upland Herbaceous": "developed_herbaceous",
    "Developed-Upland Shrubland": "developed_shrub",
    "Developed-Low Intensity": "developed",
    "Developed-Medium Intensity": "developed",
    "Developed-High Intensity": "developed",
    "Developed - Low Intensity": "developed",
    "Developed - Medium Intensity": "developed",
    "Developed - High Intensity": "developed",
    "Developed-Roads": "developed",
    "Barren": "sparse_barren",
    "Quarries-Strip Mines-Gravel Pits-Well and Wind Pads": "sparse_barren",
    "NASS-Vineyard": "agriculture",
    "NASS-Row Crop-Close Grown Crop": "agriculture",
    "NASS-Row Crop": "agriculture",
    "NASS-Close Grown Crop": "agriculture",
    "NASS-Wheat": "agriculture",
    "NASS-Aquaculture": "agriculture",
    "Cultivated Crops": "agriculture",
    "Sparse Vegetation Canopy": "sparse_barren",
}

STRUCTURE_PATTERN = re.compile(
    r"^(Tree|Shrub|Herb) (Cover|Height) (=|>=) ([0-9.]+)(%| meters?| meter)$"
)


def evt_model_class(row: dict[str, str]) -> str | None:
    physiognomy = row.get("EVT_PHYS", "")
    if physiognomy == "Exotic Tree-Shrub":
        lifeform = row.get("EVT_LF", "")
        if lifeform == "Tree":
            return "forest_tree"
        if lifeform == "Shrub":
            return "shrub"
        raise ValueError(
            "Exotic Tree-Shrub requires EVT_LF equal to Tree or Shrub; "
            f"received {lifeform!r} for Value {row.get('Value')}."
        )
    if physiognomy not in EVT_PHYS_CLASS:
        raise ValueError(
            f"Unmapped EVT_PHYS {physiognomy!r} for Value {row.get('Value')}."
        )
    return EVT_PHYS_CLASS[physiognomy]


def structure_mapping(product: str, row: dict[str, str]) -> dict[str, Any]:
    class_name = row.get("CLASSNAMES", "")
    match = STRUCTURE_PATTERN.fullmatch(class_name)
    if match:
        lifeform, measure, operator, value, unit = match.groups()
        expected_measure = "Cover" if product == "EVC" else "Height"
        if measure != expected_measure:
            raise ValueError(
                f"{product} class has unexpected measure {measure}: {class_name}"
            )
        return {
            "model_lifeform": lifeform.lower(),
            "numeric_value": float(value),
            "unit": "percent" if unit == "%" else "m",
            "censored_lower_bound": operator == ">=",
            "special_class": None,
        }
    if class_name not in SPECIAL_STRUCTURE_CLASS:
        raise ValueError(
            f"Unmapped {product} CLASSNAMES {class_name!r} "
            f"for Value {row.get('Value')}."
        )
    return {
        "model_lifeform": None,
        "numeric_value": None,
        "unit": None,
        "censored_lower_bound": False,
        "special_class": SPECIAL_STRUCTURE_CLASS[class_name],
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"LANDFIRE attribute CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        field_names = list(reader.fieldnames or [])
    if not rows or "Value" not in field_names:
        raise ValueError(f"Invalid LANDFIRE attribute CSV: {path}")
    return field_names, rows


def _csv_bytes(field_names: list[str], rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise FileExistsError(f"Existing crosswalk differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build_crosswalks(
    attribute_summary: dict[str, Any],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    tables = attribute_summary.get("tables")
    if not isinstance(tables, list) or not tables:
        raise ValueError("LANDFIRE attribute summary has no tables.")
    evt_rows: list[dict[str, Any]] = []
    structure_rows: dict[str, list[dict[str, Any]]] = {"EVC": [], "EVH": []}
    evt_source_fields: list[str] | None = None
    for table in tables:
        product = table.get("product")
        csv_path = Path(table.get("csv_output_path", ""))
        source_fields, rows = _read_csv(csv_path)
        release = table["version"]
        layer_name = table["layer_name"]
        if product == "EVT":
            evt_source_fields = evt_source_fields or source_fields
            if source_fields != evt_source_fields:
                raise ValueError("EVT attribute fields differ across releases.")
            for row in rows:
                model_class = evt_model_class(row)
                evt_rows.append(
                    {
                        "release": release,
                        "layer_name": layer_name,
                        "model_class": model_class or "",
                        "model_valid": model_class is not None,
                        **row,
                    }
                )
        elif product in structure_rows:
            for row in rows:
                structure_rows[product].append(
                    {
                        "release": release,
                        "layer_name": layer_name,
                        "product": product,
                        "Value": row["Value"],
                        "CLASSNAMES": row["CLASSNAMES"],
                        **structure_mapping(product, row),
                    }
                )

    expected_products = {"EVT", "EVC", "EVH"}
    found_products = {table.get("product") for table in tables}
    if expected_products - found_products:
        raise ValueError(
            "Crosswalk requires EVT, EVC, and EVH tables; missing "
            + ", ".join(sorted(expected_products - found_products))
        )
    present_classes = {row["model_class"] for row in evt_rows if row["model_class"]}
    if present_classes != set(MODEL_CLASSES):
        raise ValueError(
            "Portable EVT hierarchy is incomplete; found "
            + ", ".join(sorted(present_classes))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    evt_fields = [
        "release",
        "layer_name",
        "model_class",
        "model_valid",
        *(evt_source_fields or []),
    ]
    outputs = [
        (
            "evt_model_crosswalk",
            output_dir / "evt_model_crosswalk.csv",
            evt_fields,
            evt_rows,
        ),
        (
            "evc_model_lookup",
            output_dir / "evc_model_lookup.csv",
            [
                "release",
                "layer_name",
                "product",
                "Value",
                "CLASSNAMES",
                "model_lifeform",
                "numeric_value",
                "unit",
                "censored_lower_bound",
                "special_class",
            ],
            structure_rows["EVC"],
        ),
        (
            "evh_model_lookup",
            output_dir / "evh_model_lookup.csv",
            [
                "release",
                "layer_name",
                "product",
                "Value",
                "CLASSNAMES",
                "model_lifeform",
                "numeric_value",
                "unit",
                "censored_lower_bound",
                "special_class",
            ],
            structure_rows["EVH"],
        ),
    ]
    for artifact_id, path, fields, rows in outputs:
        payload = _csv_bytes(fields, rows)
        _write_bytes(path, payload, overwrite)
        artifacts.append(
            {
                "id": artifact_id,
                "path": str(path.resolve()),
                "rows": len(rows),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "landfire",
        "model_classes": list(MODEL_CLASSES),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "evt_release_rows": {
            release: sum(row["release"] == release for row in evt_rows)
            for release in sorted({row["release"] for row in evt_rows})
        },
        "evc_numeric_rows": sum(
            row["numeric_value"] is not None for row in structure_rows["EVC"]
        ),
        "evh_numeric_rows": sum(
            row["numeric_value"] is not None for row in structure_rows["EVH"]
        ),
        "definitions": {
            "evt_model_class": (
                "Portable nine-class landscape hierarchy derived from official "
                "EVT_PHYS and, only for Exotic Tree-Shrub, EVT_LF."
            ),
            "structure_numeric_value": (
                "Cover or height of the dominant mapped lifeform; values are not "
                "simultaneous multi-stratum measurements."
            ),
        },
    }
    summary_path = output_dir / "landfire_crosswalk_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
