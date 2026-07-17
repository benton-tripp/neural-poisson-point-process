from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.nlcd import (  # noqa: E402
    expected_filename,
    parse_year_expression,
    resolve_catalog,
)


class AnnualNlcdCatalogTests(unittest.TestCase):
    def test_year_expression_supports_ranges_and_values(self) -> None:
        self.assertEqual(parse_year_expression("2020-2022,2025"), [2020, 2021, 2022, 2025])

    def test_expected_filename_is_release_pinned(self) -> None:
        self.assertEqual(
            expected_filename("LndCov", 2020),
            "Annual_NLCD_LndCov_2020_CU_C1V2.zip",
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


if __name__ == "__main__":
    unittest.main()
