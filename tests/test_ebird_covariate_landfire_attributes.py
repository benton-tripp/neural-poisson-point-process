from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_attributes import (  # noqa: E402
    extract_attribute_tables,
)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.requests.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return FakeResponse(self.payload)


def evt_payload():
    fields = [
        {"name": "OID"},
        {"name": "Value"},
        {"name": "Count"},
        {"name": "EVT_NAME"},
        {"name": "EVT_LF"},
        {"name": "EVT_PHYS"},
    ]
    return {
        "objectIdFieldName": "OBJECTID",
        "fields": fields,
        "features": [
            {
                "attributes": {
                    "OID": 1,
                    "Value": 7008,
                    "Count": 12,
                    "EVT_NAME": "Open Water",
                    "EVT_LF": "Open Water",
                    "EVT_PHYS": "Water",
                }
            },
            {
                "attributes": {
                    "OID": 2,
                    "Value": 7292,
                    "Count": 34,
                    "EVT_NAME": "Forest",
                    "EVT_LF": "Forest",
                    "EVT_PHYS": "Tree",
                }
            },
        ],
    }


def catalog_fixture():
    return {
        "products_api_snapshot_sha256": "fixture",
        "layers": [
            {
                "role": "vegetation_release",
                "layerName": "LF2023_EVT",
                "version": "LF2023",
                "acronym": "EVT",
                "image_server_url": (
                    "https://example.test/Landfire_LF2023/"
                    "LF2023_EVT_CONUS/ImageServer"
                ),
            }
        ],
    }


class LandfireAttributeTests(unittest.TestCase):
    def test_catalogs_raw_json_and_normalized_csv(self) -> None:
        session = FakeSession(evt_payload())
        with tempfile.TemporaryDirectory() as temporary:
            summary = extract_attribute_tables(
                catalog_fixture(),
                Path(temporary),
                session=session,
            )
            record = summary["tables"][0]
            csv_text = Path(record["csv_output_path"]).read_text(encoding="utf-8")
            raw_payload = json.loads(
                Path(record["json_output_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(summary["table_count"], 1)
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(record["value_min"], 7008)
        self.assertEqual(record["value_max"], 7292)
        self.assertIn("EVT_PHYS", csv_text)
        self.assertEqual(len(raw_payload["features"]), 2)
        self.assertTrue(
            session.requests[0]["url"].endswith(
                "/ImageServer/rasterAttributeTable"
            )
        )
        rendering_rule = json.loads(
            session.requests[0]["params"]["renderingRule"]
        )
        self.assertEqual(
            rendering_rule["rasterFunction"],
            "LF2023_EVT_CONUS",
        )

    def test_rejects_missing_required_semantic_field(self) -> None:
        payload = evt_payload()
        payload["fields"] = [
            field for field in payload["fields"] if field["name"] != "EVT_PHYS"
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "EVT_PHYS"):
                extract_attribute_tables(
                    catalog_fixture(),
                    Path(temporary),
                    session=FakeSession(payload),
                )

    def test_rejects_duplicate_value_keys(self) -> None:
        payload = evt_payload()
        payload["features"][1]["attributes"]["Value"] = 7008
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "duplicate Value"):
                extract_attribute_tables(
                    catalog_fixture(),
                    Path(temporary),
                    session=FakeSession(payload),
                )


if __name__ == "__main__":
    unittest.main()
