from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.planner import plan_tiles  # noqa: E402
from ebird_covariates.raster_engine import (  # noqa: E402
    normalize_raster_to_tiles,
    write_logical_vrt,
)


class RasterEngineTests(unittest.TestCase):
    def test_cog_tiles_assemble_without_seam_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            aoi_path = root / "aoi.geojson"
            source_path = root / "source.tif"
            tile_dir = root / "tiles"
            vrt_path = root / "covariates.vrt"

            geometry = box(0.0, 0.0, 1000.0, 1000.0)
            gpd.GeoDataFrame(
                {"name": ["test"]},
                geometry=[geometry],
                crs="EPSG:5070",
            ).to_file(aoi_path, driver="GeoJSON")

            source_values = np.arange(100, dtype=np.float32).reshape(10, 10)
            source_transform = from_origin(0.0, 1000.0, 100.0, 100.0)
            with rasterio.open(
                source_path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="float32",
                crs="EPSG:5070",
                transform=source_transform,
                nodata=-9999.0,
            ) as source:
                source.write(source_values, 1)

            tiles = plan_tiles(
                geometry=geometry,
                resolution=250.0,
                tile_size=500.0,
                origin_x=0.0,
                origin_y=0.0,
            )
            plan = {
                "schema_version": 1,
                "build_id": "synthetic",
                "aoi": {"path": str(aoi_path), "layer": None},
                "grid": {
                    "crs": "EPSG:5070",
                    "resolution_m": 250.0,
                    "tile_size_m": 500.0,
                    "snapped_bounds_m": [0.0, 0.0, 1000.0, 1000.0],
                    "bounding_width_cells": 4,
                    "bounding_height_cells": 4,
                    "tiles": tiles,
                },
                "outputs": {"logical_raster": str(vrt_path)},
            }

            inventory = normalize_raster_to_tiles(
                plan=plan,
                source_path=source_path,
                output_dir=tile_dir,
                band_id="availability__synthetic__gradient__mean__r250__static",
                resampling="average",
            )
            self.assertEqual(inventory["tile_count"], 4)
            self.assertEqual(inventory["valid_cells"], 16)
            with rasterio.open(inventory["tiles"][0]["path"]) as tile_source:
                self.assertEqual(
                    tile_source.tags(ns="IMAGE_STRUCTURE").get("LAYOUT"),
                    "COG",
                )
                self.assertEqual(tile_source.tags().get("aoi_mask_rule"), "center")

            plan["grid"]["aoi_mask_rule"] = "all_touched"
            with self.assertRaisesRegex(ValueError, "Rerun with --overwrite"):
                normalize_raster_to_tiles(
                    plan=plan,
                    source_path=source_path,
                    output_dir=tile_dir,
                    band_id="availability__synthetic__gradient__mean__r250__static",
                    resampling="average",
                )
            plan["grid"]["aoi_mask_rule"] = "center"

            write_logical_vrt(plan, [inventory], vrt_path)
            expected = np.full((4, 4), -9999.0, dtype=np.float32)
            reproject(
                source=source_values,
                destination=expected,
                src_transform=source_transform,
                src_crs="EPSG:5070",
                src_nodata=-9999.0,
                dst_transform=from_origin(0.0, 1000.0, 250.0, 250.0),
                dst_crs="EPSG:5070",
                dst_nodata=-9999.0,
                resampling=Resampling.average,
            )
            with rasterio.open(vrt_path) as assembled:
                self.assertEqual(assembled.count, 1)
                self.assertEqual((assembled.width, assembled.height), (4, 4))
                self.assertEqual(assembled.crs.to_epsg(), 5070)
                self.assertEqual(
                    assembled.descriptions[0],
                    "availability__synthetic__gradient__mean__r250__static",
                )
                np.testing.assert_allclose(assembled.read(1), expected)


if __name__ == "__main__":
    unittest.main()
