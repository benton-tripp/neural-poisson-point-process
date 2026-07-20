from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.nlcd import (  # noqa: E402
    expected_filename,
    expected_raster_filename,
    gdal_vsis3_uri,
    parse_year_expression,
    register_aws_sources,
    register_local_sources,
    resolve_catalog,
    validate_registered_sources,
)


class AnnualNlcdCatalogTests(unittest.TestCase):
    def test_year_expression_supports_ranges_and_values(self) -> None:
        self.assertEqual(parse_year_expression("2020-2022,2025"), [2020, 2021, 2022, 2025])

    def test_expected_filename_is_release_pinned(self) -> None:
        self.assertEqual(
            expected_filename("LndCov", 2020),
            "Annual_NLCD_LndCov_2020_CU_C1V2.zip",
        )
        self.assertEqual(
            expected_raster_filename("LndCov", 2020),
            "Annual_NLCD_LndCov_2020_CU_C1V2.tif",
        )
        self.assertEqual(
            gdal_vsis3_uri("LndCov", 2020),
            "/vsis3/usgs-landcover/annual-nlcd/c1/v2/cu/mosaic/"
            "Annual_NLCD_LndCov_2020_CU_C1V2.tif",
        )

    @patch("ebird_covariates.nlcd.fetch_item_metadata")
    def test_catalog_selects_exact_product_year_files(self, fetch_metadata) -> None:
        def fake_item(product_code: str, timeout: float):
            del timeout
            filenames = [expected_filename(product_code, year) for year in (2019, 2020)]
            return {
                "id": f"item-{product_code}",
                "title": f"Annual NLCD Collection 1.2 test {product_code}",
                "files": [
                    {
                        "name": filename,
                        "size": 1234,
                        "url": f"https://example.test/{filename}",
                        "downloadUri": f"https://example.test/request/{filename}",
                        "s3DownloadRequestPageUri": (
                            f"https://example.test/request-page/{filename}"
                        ),
                    }
                    for filename in filenames
                ] + [
                    {
                        "name": "unrelated.xml",
                        "size": 10,
                        "url": "https://example.test/unrelated.xml",
                    },
                ],
            }

        fetch_metadata.side_effect = fake_item
        catalog = resolve_catalog(
            years=[2020],
            products=["LndCov", "FctImp"],
            extra_years_by_product={"LndCov": [2019]},
            max_workers=2,
        )
        self.assertEqual(catalog["release"], "C1V2")
        self.assertEqual(len(catalog["files"]), 3)
        self.assertEqual(catalog["schema_version"], 2)
        self.assertEqual(catalog["total_source_bytes"], 3702)
        self.assertEqual(
            [file["product_code"] for file in catalog["files"]],
            ["LndCov", "LndCov", "FctImp"],
        )
        self.assertEqual(catalog["product_years"]["LndCov"], [2019, 2020])
        self.assertEqual(catalog["product_years"]["FctImp"], [2020])
        self.assertEqual(len(catalog["land_cover_classes"]), 16)
        self.assertEqual(
            catalog["acquisition_status"],
            "metadata_resolved_manual_or_aws_credentials_required",
        )
        self.assertFalse(catalog["files"][0]["direct_download_available"])
        self.assertNotIn("download_url", catalog["files"][0])

    def test_local_zip_and_aws_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raster_name = expected_raster_filename("LndCov", 2020)
            raster_path = root / raster_name
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=4,
                height=3,
                count=1,
                dtype="uint8",
                crs="EPSG:5070",
                transform=from_origin(0, 750, 250, 250),
                nodata=255,
            ) as dataset:
                dataset.write(np.full((3, 4), 41, dtype=np.uint8), 1)

            archive_name = expected_filename("LndCov", 2020)
            archive_path = root / archive_name
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.write(raster_path, arcname=f"nested/{raster_name}")
            raster_path.unlink()

            catalog = {
                "release": "C1V2",
                "files": [
                    {
                        "product_code": "LndCov",
                        "product_name": "Land Cover",
                        "year": 2020,
                        "filename": archive_name,
                        "raster_filename": raster_name,
                        "size_bytes": archive_path.stat().st_size,
                        "aws_s3_uri": "s3://example/source.tif",
                        "aws_gdal_vsi_uri": "/vsis3/example/source.tif",
                        "aws_region": "us-west-2",
                    }
                ],
            }
            local = register_local_sources(catalog, root, calculate_sha256=True)
            self.assertEqual(local["sources"][0]["source_type"], "local_zip")
            self.assertEqual(local["sources"][0]["raster"]["crs"], "EPSG:5070")
            self.assertTrue(local["sources"][0]["catalog_archive_size_matches"])
            self.assertEqual(len(local["sources"][0]["source_sha256"]), 64)
            validation = validate_registered_sources(local)
            self.assertTrue(validation["all_sources_opened"])
            self.assertTrue(validation["land_cover_grids_aligned"])
            self.assertEqual(validation["source_count"], 1)

            aws = register_aws_sources(catalog)
            self.assertEqual(aws["backend"], "aws_requester_pays")
            self.assertEqual(
                aws["sources"][0]["raster_uri"], "/vsis3/example/source.tif"
            )


if __name__ == "__main__":
    unittest.main()
