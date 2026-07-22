from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.nlcd_derive import (  # noqa: E402
    circular_kernel,
    derive_nlcd,
    mask_tile_to_aoi,
)
from ebird_covariates.nlcd_qa import (  # noqa: E402
    plot_nlcd_tile_preview,
    validate_nlcd_checklist_support,
    validate_nlcd_derivation,
)


class AnnualNlcdDerivationTests(unittest.TestCase):
    def test_circular_kernel_uses_fractional_cell_overlap(self) -> None:
        kernel = circular_kernel(250.0, 250.0)
        self.assertAlmostEqual(float(kernel.sum()), float(np.pi), delta=0.03)
        self.assertTrue(np.any((kernel > 0.0) & (kernel < 1.0)))

    def test_all_touched_aoi_mask_retains_boundary_cell(self) -> None:
        tile = {"bounds_m": [0.0, 0.0, 500.0, 500.0]}
        values = np.ones((2, 2), dtype=np.float32)
        geometry = box(1.0, 1.0, 10.0, 10.0)
        center = mask_tile_to_aoi(
            values.copy(),
            tile,
            250.0,
            geometry,
            all_touched=False,
        )
        touched = mask_tile_to_aoi(
            values.copy(),
            tile,
            250.0,
            geometry,
            all_touched=True,
        )
        self.assertTrue(np.all(center == -9999.0))
        self.assertEqual(int(np.count_nonzero(touched != -9999.0)), 1)

    def test_checklist_support_distinguishes_masks_extent_and_years(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vrt_path = root / "annual_nlcd.tif"
            nodata = -9999.0
            values = np.ones((3, 2, 2), dtype=np.float32)
            values[1, 0, 1] = nodata
            descriptions = [
                "availability__annual_nlcd__open_water_fraction__mean__r250__y2020",
                "availability__annual_nlcd__open_water_fraction__mean__r1000__y2020",
                "availability__annual_nlcd__open_water_fraction__mean__r5000__y2020",
            ]
            with rasterio.open(
                vrt_path,
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

            checklist_path = root / "checklists.parquet"
            pd.DataFrame(
                {
                    "sampling_event_identifier": ["inside", "masked", "outside", "old"],
                    "latitude": [35.5, 35.5, 20.0, 34.5],
                    "longitude": [-79.5, -78.5, -90.0, -79.5],
                    "observation_date": [
                        "2020-05-01",
                        "2020-05-01",
                        "2020-05-01",
                        "2019-05-01",
                    ],
                }
            ).to_parquet(checklist_path, index=False)
            summary = {
                "build_id": "support_test",
                "release": "C1V2",
                "logical_vrt": str(vrt_path),
                "years": [2020],
                "neighborhoods_m": [250, 1000, 5000],
                "land_cover_classes": [{"value": 11, "name": "open_water"}],
            }

            validation, unsupported = validate_nlcd_checklist_support(
                summary,
                checklist_path,
            )
            self.assertEqual(validation["checklist_count"], 4)
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
                {"masked", "outside", "old"},
            )

    def write_source(
        self,
        path: Path,
        values: np.ndarray,
        nodata: int = 255,
    ) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype="uint8",
            crs="EPSG:5070",
            transform=from_origin(-500.0, 1500.0, 50.0, 50.0),
            nodata=nodata,
        ) as dataset:
            dataset.write(values, 1)

    def test_derives_fractions_change_and_vrt_on_plan_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aoi_path = root / "aoi.geojson"
            output_dir = root / "derived"
            gpd.GeoDataFrame(
                {"name": ["test"]},
                geometry=[box(0.0, 0.0, 1000.0, 1000.0)],
                crs="EPSG:5070",
            ).to_file(aoi_path, driver="GeoJSON")

            previous = np.full((40, 40), 41, dtype=np.uint8)
            current = previous.copy()
            current[:, 20:] = 42
            impervious = np.zeros((40, 40), dtype=np.uint8)
            impervious[:, 20:] = 100
            paths = {
                ("LndCov", 2019): root / "landcover_2019.tif",
                ("LndCov", 2020): root / "landcover_2020.tif",
                ("FctImp", 2020): root / "impervious_2020.tif",
            }
            self.write_source(paths[("LndCov", 2019)], previous)
            self.write_source(paths[("LndCov", 2020)], current)
            self.write_source(paths[("FctImp", 2020)], impervious)

            tiles = []
            for y_index in range(2):
                for x_index in range(2):
                    tiles.append(
                        {
                            "tile_id": f"xp{x_index:04d}_yp{y_index:04d}",
                            "x_index": x_index,
                            "y_index": y_index,
                            "bounds_m": [
                                x_index * 500.0,
                                y_index * 500.0,
                                (x_index + 1) * 500.0,
                                (y_index + 1) * 500.0,
                            ],
                            "width": 2,
                            "height": 2,
                            "active_cells_center_rule": 4,
                        }
                    )
            plan = {
                "schema_version": 1,
                "build_id": "synthetic_nlcd",
                "aoi": {"path": str(aoi_path), "layer": None},
                "grid": {
                    "crs": "EPSG:5070",
                    "resolution_m": 250.0,
                    "tile_size_m": 500.0,
                    "tile_width_cells": 2,
                    "snapped_bounds_m": [0.0, 0.0, 1000.0, 1000.0],
                    "bounding_width_cells": 4,
                    "bounding_height_cells": 4,
                    "tiles": tiles,
                },
                "temporal": {"start_year": 2020, "end_year": 2020},
                "neighborhoods_m": [250.0],
                "outputs": {"build_dir": str(root / "build")},
            }
            registration = {
                "schema_version": 1,
                "source_id": "annual_nlcd",
                "release": "C1V2",
                "backend": "local",
                "sources": [
                    {
                        "product_code": product,
                        "year": year,
                        "raster_uri": str(path),
                    }
                    for (product, year), path in paths.items()
                ],
            }

            summary = derive_nlcd(
                plan=plan,
                registration=registration,
                output_dir=output_dir,
                classes={41: "deciduous_forest", 42: "evergreen_forest"},
            )

            self.assertEqual(summary["band_count"], 7)
            self.assertEqual(summary["expected_band_count"], 7)
            self.assertEqual(summary["tile_count"], 4)
            self.assertEqual(summary["plan_tile_count"], 4)
            self.assertEqual(summary["aoi_mask_rule"], "center")
            self.assertEqual(summary["derived_cog_count"], 28)
            self.assertGreater(summary["derived_cog_bytes"], 0)
            self.assertGreater(summary["derived_cog_mib"], 0.0)
            self.assertGreater(summary["elapsed_seconds"], 0.0)
            vrt_path = Path(summary["logical_vrt"])
            with rasterio.open(vrt_path) as dataset:
                self.assertEqual(dataset.count, 7)
                self.assertEqual((dataset.width, dataset.height), (4, 4))
                descriptions = list(dataset.descriptions)
                deciduous = dataset.read(
                    descriptions.index(
                        "availability__annual_nlcd__deciduous_forest_fraction__mean__r250__y2020"
                    )
                    + 1
                )
                evergreen = dataset.read(
                    descriptions.index(
                        "availability__annual_nlcd__evergreen_forest_fraction__mean__r250__y2020"
                    )
                    + 1
                )
                change = dataset.read(
                    descriptions.index(
                        "availability__annual_nlcd__land_cover_change_fraction__mean__r250__y2020"
                    )
                    + 1
                )
                impervious_result = dataset.read(
                    descriptions.index(
                        "availability__annual_nlcd__impervious_fraction__mean__r250__y2020"
                    )
                    + 1
                )

            np.testing.assert_allclose(deciduous + evergreen, 1.0, atol=1e-5)
            np.testing.assert_allclose(
                deciduous,
                np.repeat(deciduous[:1, :], deciduous.shape[0], axis=0),
                atol=1e-5,
            )
            self.assertGreater(float(deciduous[:, 0].mean()), 0.9)
            self.assertGreater(float(evergreen[:, -1].mean()), 0.9)
            self.assertLess(float(change[:, 0].mean()), float(change[:, -1].mean()))
            self.assertLess(
                float(impervious_result[:, 0].mean()),
                float(impervious_result[:, -1].mean()),
            )

            pilot_output_dir = root / "derived_pilot"
            pilot = derive_nlcd(
                plan=plan,
                registration=registration,
                output_dir=pilot_output_dir,
                classes={41: "deciduous_forest", 42: "evergreen_forest"},
                tile_ids=["xp0000_yp0000"],
            )
            self.assertEqual(pilot["tile_count"], 1)
            self.assertEqual(pilot["plan_tile_count"], 4)
            self.assertEqual(pilot["tile_ids"], ["xp0000_yp0000"])
            with rasterio.open(pilot["logical_vrt"]) as dataset:
                self.assertEqual((dataset.width, dataset.height), (4, 4))

            validation = validate_nlcd_derivation(plan, pilot)
            self.assertTrue(validation["all_checks_passed"], validation["issues"])
            self.assertEqual(validation["band_count"], 7)
            self.assertEqual(validation["derived_cog_count"], 7)
            self.assertLessEqual(
                validation["maximum_class_fraction_sum_error"],
                1e-5,
            )
            self.assertEqual(validation["minimum_supported_aoi_fraction"], 1.0)
            self.assertTrue(
                all(
                    check["supported_aoi_fraction"] == 1.0
                    for check in validation["class_fraction_sum_checks"]
                )
            )
            preview_path = plot_nlcd_tile_preview(
                plan,
                pilot,
                "xp0000_yp0000",
                root / "pilot_preview.png",
            )
            self.assertTrue(preview_path.exists())
            self.assertGreater(preview_path.stat().st_size, 0)

            plan["grid"]["aoi_mask_rule"] = "all_touched"
            mismatched_validation = validate_nlcd_derivation(plan, pilot)
            self.assertFalse(mismatched_validation["all_checks_passed"])
            self.assertTrue(
                any(
                    "AOI mask rule" in issue
                    for issue in mismatched_validation["issues"]
                )
            )
            with self.assertRaisesRegex(ValueError, "Rerun with --overwrite"):
                derive_nlcd(
                    plan=plan,
                    registration=registration,
                    output_dir=pilot_output_dir,
                    classes={41: "deciduous_forest", 42: "evergreen_forest"},
                    tile_ids=["xp0000_yp0000"],
                )
            plan["grid"]["aoi_mask_rule"] = "center"

            with self.assertRaisesRegex(ValueError, "unknown_tile"):
                derive_nlcd(
                    plan=plan,
                    registration=registration,
                    output_dir=root / "derived_unknown",
                    classes={41: "deciduous_forest", 42: "evergreen_forest"},
                    tile_ids=["unknown_tile"],
                )


if __name__ == "__main__":
    unittest.main()
