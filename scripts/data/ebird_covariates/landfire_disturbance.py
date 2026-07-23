"""Build auditable annual LANDFIRE disturbance-code lookups."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "Value",
    "DIST_YEAR",
    "DIST_TYPE",
    "SEVERITY",
    "DESCRIPTIO",
}
FILL_TYPES = {"fill-nodata", "fill-not mapped"}
NON_EVENT_TYPES = {"na", "water"}


def classify_disturbance_row(row: dict[str, str]) -> dict[str, Any]:
    value = int(row["Value"])
    disturbance_type = row["DIST_TYPE"].strip()
    type_key = disturbance_type.casefold()
    description_key = row["DESCRIPTIO"].strip().casefold()
    is_fill = value < 0 or type_key in FILL_TYPES
    is_background = (
        value == 0
        and type_key == "na"
        and description_key == "background"
    )
    is_water = type_key == "water"
    is_disturbed = value > 0 and not is_water and type_key not in NON_EVENT_TYPES
    is_source_valid = not is_fill
    is_analysis_support = is_background or is_disturbed
    if is_fill:
        category = "fill"
    elif is_background:
        category = "background"
    elif is_water:
        category = "water_mask"
    elif is_disturbed:
        category = "disturbance_event"
    else:
        category = "unsupported_non_event"
    return {
        "is_source_valid": is_source_valid,
        "is_analysis_support": is_analysis_support,
        "is_disturbed": is_disturbed,
        "model_category": category,
    }


def _write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise FileExistsError(f"Existing output differs from source: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _csv_bytes(field_names: list[str], rows: list[dict[str, Any]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_disturbance_lookup(
    attribute_summary: dict[str, Any],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    tables = [
        table
        for table in attribute_summary.get("tables", [])
        if table.get("product") == "Dist"
    ]
    if not tables:
        raise ValueError("No LANDFIRE disturbance attribute tables were supplied.")

    output_rows: list[dict[str, Any]] = []
    year_records: list[dict[str, Any]] = []
    seen_years: set[int] = set()
    for table in tables:
        path = Path(table["csv_output_path"])
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            field_names = set(reader.fieldnames or [])
            missing = sorted(REQUIRED_FIELDS - field_names)
            if missing:
                raise ValueError(
                    f"LANDFIRE disturbance table {path} lacks: "
                    + ", ".join(missing)
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"LANDFIRE disturbance table is empty: {path}")
        years = {int(row["DIST_YEAR"]) for row in rows}
        if len(years) != 1:
            raise ValueError(
                f"LANDFIRE disturbance table {path} has multiple DIST_YEAR values."
            )
        year = years.pop()
        if year in seen_years:
            raise ValueError(f"Duplicate LANDFIRE disturbance year: {year}")
        seen_years.add(year)

        classified = []
        for row in rows:
            flags = classify_disturbance_row(row)
            classified.append(
                {
                    "release": table["version"],
                    "layer_name": table["layer_name"],
                    "product": "Dist",
                    "observation_year": year,
                    **row,
                    **{
                        key: str(value).lower() if isinstance(value, bool) else value
                        for key, value in flags.items()
                    },
                }
            )
        backgrounds = [
            row for row in classified if row["model_category"] == "background"
        ]
        events = [
            row
            for row in classified
            if row["model_category"] == "disturbance_event"
        ]
        unsupported = [
            row
            for row in classified
            if row["model_category"] == "unsupported_non_event"
        ]
        if len(backgrounds) != 1 or int(backgrounds[0]["Value"]) != 0:
            raise ValueError(
                f"LANDFIRE disturbance year {year} lacks one canonical background row."
            )
        if not events:
            raise ValueError(f"LANDFIRE disturbance year {year} has no event codes.")
        if unsupported:
            values = ", ".join(str(row["Value"]) for row in unsupported[:10])
            raise ValueError(
                f"LANDFIRE disturbance year {year} has unclassified non-events: {values}"
            )
        output_rows.extend(classified)
        year_records.append(
            {
                "year": year,
                "release": table["version"],
                "layer_name": table["layer_name"],
                "row_count": len(classified),
                "event_code_count": len(events),
                "water_mask_code_count": sum(
                    row["model_category"] == "water_mask" for row in classified
                ),
                "fill_code_count": sum(
                    row["model_category"] == "fill" for row in classified
                ),
                "event_types": sorted({row["DIST_TYPE"] for row in events}),
            }
        )

    output_rows.sort(
        key=lambda row: (int(row["observation_year"]), int(row["Value"]))
    )
    preferred = [
        "release",
        "layer_name",
        "product",
        "observation_year",
        "Value",
        "DIST_YEAR",
        "DIST_TYPE",
        "TYPE_CONFI",
        "SEVERITY",
        "SEV_SOURCE",
        "SEV_CONFID",
        "SOURCE1",
        "SOURCE2",
        "SOURCE3",
        "SOURCE4",
        "DESCRIPTIO",
        "is_source_valid",
        "is_analysis_support",
        "is_disturbed",
        "model_category",
    ]
    field_names = [
        name for name in preferred if any(name in row for row in output_rows)
    ]
    payload = _csv_bytes(field_names, output_rows)
    output_path = output_dir / "disturbance_model_lookup.csv"
    _write_bytes(output_path, payload, overwrite)
    artifact = {
        "id": "disturbance_model_lookup",
        "path": str(output_path.resolve()),
        "rows": len(output_rows),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "landfire",
        "artifact_count": 1,
        "artifacts": [artifact],
        "disturbance_years": sorted(seen_years),
        "year_records": sorted(year_records, key=lambda record: record["year"]),
        "definitions": {
            "disturbance_event": (
                "Positive official code with an event DIST_TYPE; Water is excluded."
            ),
            "analysis_support": (
                "Background land plus disturbance-event pixels; fill and Water masks "
                "are excluded from the denominator."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "landfire_disturbance_lookup_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary
