from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.nlcd_derive import circular_kernel, derive_nlcd  # noqa: E402


class AnnualNlcdDerivationTests(unittest.TestCase):
    def test_circular_kernel_uses_fractional_cell_overlap(self) -> None:
        kernel = circular_kernel(250.0, 250.0)
        self.assertAlmostEqual(float(kernel.sum()), float(np.pi), delta=0.03)
        self.assertTrue(np.any((kernel > 0.0) & (kernel < 1.0)))

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
