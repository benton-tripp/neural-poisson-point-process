from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_crosswalk import (  # noqa: E402
    MODEL_CLASSES,
    build_crosswalks,
    evt_model_class,
    structure_mapping,
)


class LandfireCrosswalkTests(unittest.TestCase):
    def test_evt_hierarchy_uses_lifeform_for_exotic_woody_rows(self) -> None:
        self.assertEqual(
            evt_model_class(
                {
                    "Value": "9302",
                    "EVT_PHYS": "Exotic Tree-Shrub",
                    "EVT_LF": "Tree",
                }
            ),
            "forest_tree",
        )
        self.assertEqual(
            evt_model_class(
                {
                    "Value": "9310",
                    "EVT_PHYS": "Exotic Tree-Shrub",
                    "EVT_LF": "Shrub",
                }
            ),
            "shrub",
        )

    def test_structure_parser_preserves_censored_lower_bound(self) -> None:
        mapping = structure_mapping(
            "EVC",
            {"Value": "399", "CLASSNAMES": "Herb Cover >= 99%"},
        )
        self.assertEqual(mapping["model_lifeform"], "herb")
        self.assertEqual(mapping["numeric_value"], 99.0)
        self.assertEqual(mapping["unit"], "percent")
        self.assertTrue(mapping["censored_lower_bound"])

    def test_rejects_unknown_official_category(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unmapped EVT_PHYS"):
            evt_model_class(
                {
                    "Value": "999",
                    "EVT_PHYS": "Unexpected category",
                    "EVT_LF": "Tree",
                }
            )

    def test_builds_three_release_aware_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tables = []
            evt_categories = [
                ("Fill-NoData", "Tree"),
                ("Conifer", "Tree"),
                ("Shrubland", "Shrub"),
                ("Grassland", "Herb"),
                ("Riparian", "Tree"),
                ("Agricultural", "Herb"),
                ("Developed", "Tree"),
                ("Sparsely Vegetated", "Sparse"),
                ("Open Water", "Water"),
                ("Snow-Ice", "Snow"),
            ]
            for release in ("LF2016", "LF2022", "LF2023"):
                for product in ("EVT", "EVC", "EVH"):
                    path = root / f"{release}_{product}.csv"
                    if product == "EVT":
                        fields = ["Value", "EVT_NAME", "EVT_LF", "EVT_PHYS"]
                        rows = [
                            {
                                "Value": str(index),
                                "EVT_NAME": physiognomy,
                                "EVT_LF": lifeform,
                                "EVT_PHYS": physiognomy,
                            }
                            for index, (physiognomy, lifeform) in enumerate(
                                evt_categories
                            )
                        ]
                    else:
                        fields = ["Value", "CLASSNAMES"]
                        noun = "Cover" if product == "EVC" else "Height"
                        suffix = "10%" if product == "EVC" else "1 meter"
                        rows = [
                            {
                                "Value": "101",
                                "CLASSNAMES": f"Tree {noun} = {suffix}",
                            },
                            {"Value": "11", "CLASSNAMES": "Open Water"},
                        ]
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                    tables.append(
                        {
                            "version": release,
                            "layer_name": f"{release}_{product}",
                            "product": product,
                            "csv_output_path": str(path),
                        }
                    )
            output_dir = root / "crosswalks"
            summary = build_crosswalks(
                {"tables": tables},
                output_dir,
            )
            persisted = json.loads(
                (output_dir / "landfire_crosswalk_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["artifact_count"], 3)
        self.assertEqual(set(summary["model_classes"]), set(MODEL_CLASSES))
        self.assertEqual(persisted["evt_release_rows"]["LF2023"], 10)


if __name__ == "__main__":
    unittest.main()
