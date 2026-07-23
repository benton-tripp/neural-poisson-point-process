from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_qa import (  # noqa: E402
    compare_landfire_releases,
    validate_landfire_checklist_support,
)


class LandfireChecklistSupportTests(unittest.TestCase):
    def test_support_distinguishes_masks_and_release_extent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_path = root / "landfire_lf2016_tile_a.tif"
            nodata = -9999.0
            values = np.ones((3, 2, 2), dtype=np.float32)
            values[1, 0, 1] = nodata
            descriptions = [
                "availability__landfire__evt_forest_tree_fraction__mean__r250__lf2016",
                "availability__landfire__evt_forest_tree_fraction__mean__r1000__lf2016",
                "availability__landfire__evt_forest_tree_fraction__mean__r5000__lf2016",
            ]
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=3,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(-80.0, 36.0, 1.0, 1.0),
                nodata=nodata,
            ) as dataset:
                dataset.write(values)
                for index, description in enumerate(descriptions, start=1):
                    dataset.set_band_description(index, description)

            summary_path = root / "landfire_derived_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "release": "LF2016",
                        "tile_id": "tile_a",
                        "neighborhoods_m": [250, 1000, 5000],
                        "logical_vrt": str(raster_path),
                    }
                ),
                encoding="utf-8",
            )
            validation_path = root / "landfire_validation.json"
            validation_path.write_text(
                json.dumps({"all_checks_passed": True}),
                encoding="utf-8",
            )
            checklist_path = root / "checklists.parquet"
            pd.DataFrame(
                {
                    "sampling_event_identifier": ["inside", "masked", "outside"],
                    "latitude": [35.5, 35.5, 20.0],
                    "longitude": [-79.5, -78.5, -90.0],
                    "observation_date": [
                        "2020-05-01",
                        "2020-05-01",
                        "2020-05-01",
                    ],
                }
            ).to_parquet(checklist_path, index=False)
            plan = {
                "build_id": "support_test",
                "grid": {
                    "crs": "EPSG:4326",
                    "tiles": [
                        {
                            "tile_id": "tile_a",
                            "bounds_m": [-80.0, 34.0, -78.0, 36.0],
                        }
                    ],
                },
            }
            manifest = {
                "build_id": "support_test",
                "units": [
                    {
                        "component": "vegetation",
                        "release": "LF2016",
                        "tile_id": "tile_a",
                        "status": "completed",
                        "summary_path": str(summary_path),
                        "validation_path": str(validation_path),
                    }
                ],
            }

            validation, unsupported = validate_landfire_checklist_support(
                plan,
                manifest,
                checklist_path,
                "LF2016",
            )

            self.assertEqual(validation["checklist_count"], 3)
            self.assertEqual(validation["eligible_checklist_count"], 2)
            support = {
                record["radius_m"]: record
                for record in validation["support_by_radius"]
            }
            self.assertEqual(support[250]["supported_checklists"], 2)
            self.assertEqual(support[1000]["supported_checklists"], 1)
            self.assertEqual(support[5000]["supported_checklists"], 2)
            self.assertEqual(
                set(unsupported["sampling_event_identifier"]),
                {"masked", "outside"},
            )

    def test_support_rejects_incomplete_release(self) -> None:
        plan = {
            "grid": {
                "crs": "EPSG:5070",
                "tiles": [{"tile_id": "tile_a", "bounds_m": [0, 0, 1, 1]}],
            }
        }
        manifest = {
            "units": [
                {
                    "component": "vegetation",
                    "release": "LF2016",
                    "tile_id": "tile_a",
                    "status": "pending",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "incomplete tiles"):
            validate_landfire_checklist_support(
                plan,
                manifest,
                Path("unused.parquet"),
                "LF2016",
            )

    def test_release_comparison_matches_schema_and_change_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summaries = {}
            validations = {}
            units = []
            for release, offset in (("LF2016", 0.0), ("LF2022", 0.1)):
                release_slug = release.lower()
                raster_path = root / f"landfire_{release_slug}_tile_a.tif"
                values = np.stack(
                    [
                        np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
                        + offset,
                        np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
                        + 10.0 * offset,
                        np.ones((2, 2), dtype=np.float32),
                    ]
                )
                descriptions = [
                    "availability__landfire__evt_forest_tree_fraction__mean__"
                    f"r250__{release_slug}",
                    "availability__landfire__dominant_tree_height_m_conditional__"
                    f"mean__r1000__{release_slug}",
                    "availability__landfire__source_coverage_fraction__value__"
                    f"r250__{release_slug}",
                ]
                with rasterio.open(
                    raster_path,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=3,
                    dtype="float32",
                    crs="EPSG:5070",
                    transform=from_origin(0.0, 2.0, 1.0, 1.0),
                    nodata=-9999.0,
                ) as dataset:
                    dataset.write(values)
                    for index, description in enumerate(descriptions, start=1):
                        dataset.set_band_description(index, description)

                summary_path = root / f"{release_slug}_summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "release": release,
                            "tile_id": "tile_a",
                            "logical_vrt": str(raster_path),
                        }
                    ),
                    encoding="utf-8",
                )
                validation_path = root / f"{release_slug}_validation.json"
                validation_path.write_text(
                    json.dumps(
                        {
                            "all_checks_passed": True,
                            "evt_fraction_sum_checks": [
                                {
                                    "radius_m": 250,
                                    "supported_aoi_fraction": 1.0,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                summaries[release] = summary_path
                validations[release] = validation_path
                units.append(
                    {
                        "component": "vegetation",
                        "release": release,
                        "tile_id": "tile_a",
                        "status": "completed",
                        "summary_path": str(summary_path),
                        "validation_path": str(validation_path),
                    }
                )

            plan = {
                "build_id": "comparison_test",
                "grid": {
                    "crs": "EPSG:5070",
                    "tiles": [
                        {"tile_id": "tile_a", "bounds_m": [0.0, 0.0, 2.0, 2.0]}
                    ],
                },
            }
            summary, metrics = compare_landfire_releases(
                plan,
                {"build_id": "comparison_test", "units": units},
                "LF2016",
                "LF2022",
                tile_ids=["tile_a"],
            )

            self.assertTrue(summary["structural_checks_passed"])
            self.assertEqual(summary["matched_band_count"], 3)
            self.assertEqual(summary["maximum_absolute_support_difference"], 0.0)
            by_variable = {metric["variable"]: metric for metric in metrics}
            forest = by_variable["evt_forest_tree_fraction"]
            self.assertAlmostEqual(forest["mean_absolute_delta"], 0.1, places=6)
            self.assertAlmostEqual(forest["mean_delta"], 0.1, places=6)
            self.assertAlmostEqual(forest["pearson"], 1.0, places=6)
            height = by_variable["dominant_tree_height_m_conditional"]
            self.assertAlmostEqual(height["mean_absolute_delta"], 1.0, places=6)
            coverage = by_variable["source_coverage_fraction"]
            self.assertEqual(coverage["mean_absolute_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
