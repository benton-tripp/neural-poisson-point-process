from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_disturbance import (  # noqa: E402
    build_disturbance_lookup,
    classify_disturbance_row,
)


def row(value: int, disturbance_type: str, description: str) -> dict[str, str]:
    return {
        "Value": str(value),
        "DIST_YEAR": "2023",
        "DIST_TYPE": disturbance_type,
        "TYPE_CONFI": "NA",
        "SEVERITY": "NA",
        "SEV_SOURCE": "NA",
        "SEV_CONFID": "NA",
        "SOURCE1": "fixture",
        "SOURCE2": "NA",
        "SOURCE3": "NA",
        "SOURCE4": "NA",
        "DESCRIPTIO": description,
    }


class LandfireDisturbanceTests(unittest.TestCase):
    def test_classifies_fill_background_water_and_event(self) -> None:
        self.assertEqual(
            classify_disturbance_row(row(-9999, "Fill-NoData", "Fill-NoData"))[
                "model_category"
            ],
            "fill",
        )
        self.assertTrue(
            classify_disturbance_row(row(0, "NA", "Background"))[
                "is_analysis_support"
            ]
        )
        water = classify_disturbance_row(
            row(16, "Water", "Water body or other non-mappable land cover type.")
        )
        self.assertFalse(water["is_analysis_support"])
        self.assertFalse(water["is_disturbed"])
        event = classify_disturbance_row(row(11, "Fire", "Mapped fire"))
        self.assertTrue(event["is_analysis_support"])
        self.assertTrue(event["is_disturbed"])

    def test_builds_release_aware_lookup(self) -> None:
        rows = [
            row(-9999, "Fill-NoData", "Fill-NoData"),
            row(0, "NA", "Background"),
            row(16, "Water", "Water body"),
            row(11, "Fire", "Mapped fire"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dist.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary = build_disturbance_lookup(
                {
                    "tables": [
                        {
                            "product": "Dist",
                            "version": "LF2023",
                            "layer_name": "LF2023_Dist23",
                            "csv_output_path": str(source),
                        }
                    ]
                },
                root / "output",
            )
            lookup_path = Path(summary["artifacts"][0]["path"])
            with lookup_path.open(encoding="utf-8") as stream:
                lookup_rows = list(csv.DictReader(stream))
            persisted = json.loads(
                Path(summary["summary_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(summary["disturbance_years"], [2023])
        self.assertEqual(persisted["year_records"][0]["event_code_count"], 1)
        event = next(record for record in lookup_rows if record["Value"] == "11")
        self.assertEqual(event["is_disturbed"], "true")
        water = next(record for record in lookup_rows if record["Value"] == "16")
        self.assertEqual(water["is_analysis_support"], "false")


if __name__ == "__main__":
    unittest.main()
