from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire import (  # noqa: E402
    PRODUCTS_API_URL,
    resolve_catalog,
    validate_release_by_year,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, products):
        self.products = products
        self.requested_urls: list[str] = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.requested_urls.append(url)
        if url == PRODUCTS_API_URL:
            return FakeResponse({"products": self.products})
        layer_name = url.rsplit("/", 2)[-2].removesuffix("_CONUS")
        version = layer_name.split("_", 1)[0]
        return FakeResponse(
            {
                "name": f"Landfire_{version}/{layer_name}_CONUS",
                "bandCount": 1,
                "pixelType": "S16",
                "pixelSizeX": 30,
                "pixelSizeY": 30,
                "noDataValue": -9999,
                "spatialReference": {"wkid": 5070, "latestWkid": 5070},
                "extent": {
                    "xmin": -2362395,
                    "ymin": 221265,
                    "xmax": 2327655,
                    "ymax": 3267405,
                    "spatialReference": {"wkid": 5070},
                },
                "serviceDataType": "esriImageServiceDataTypeThematic",
                "defaultResamplingMethod": "Nearest",
                "maxImageWidth": 100000,
                "maxImageHeight": 100000,
            }
        )


def product_record(version: str, acronym: str, product_name: str, layer: str):
    return {
        "productName": product_name,
        "theme": "Disturbance" if acronym == "Dist" else "Vegetation",
        "layerName": layer,
        "acronym": acronym,
        "version": version,
        "conus": True,
        "ak": True,
        "hi": True,
        "prvi": True,
        "geoAreas": "All",
    }


def product_fixture():
    products = []
    names = {
        "EVT": "Existing Vegetation Type",
        "EVC": "Existing Vegetation Cover",
        "EVH": "Existing Vegetation Height",
    }
    for version in ("LF2016", "LF2022", "LF2023"):
        for acronym, name in names.items():
            products.append(
                product_record(version, acronym, name, f"{version}_{acronym}")
            )
    disturbance_versions = {
        2020: "LF2020",
        2021: "LF2022",
        2022: "LF2022",
        2023: "LF2023",
    }
    for year, version in disturbance_versions.items():
        products.append(
            product_record(
                version,
                "Dist",
                "Final Annual Disturbance",
                f"{version}_Dist{year % 100:02d}",
            )
        )
    return products


class LandfireCatalogTests(unittest.TestCase):
    def test_resolves_release_policy_and_exact_disturbance_years(self) -> None:
        session = FakeSession(product_fixture())
        catalog = resolve_catalog(
            observation_years=[2020, 2021, 2022, 2023],
            vegetation_releases=["LF2016", "LF2022", "LF2023"],
            release_by_year={
                "2020": "LF2016",
                "2021": "LF2016",
                "2022": "LF2022",
                "2023": "LF2023",
            },
            disturbance_years=[2020, 2021, 2022, 2023],
            max_workers=1,
            session=session,
        )

        self.assertEqual(catalog["layer_count"], 13)
        self.assertTrue(catalog["all_services_validated"])
        self.assertEqual(
            catalog["vegetation_source_age_by_year"],
            {"2020": 4, "2021": 5, "2022": 0, "2023": 0},
        )
        disturbance = [
            layer for layer in catalog["layers"] if layer["role"] == "annual_disturbance"
        ]
        self.assertEqual(
            [layer["layerName"] for layer in disturbance],
            [
                "LF2020_Dist20",
                "LF2022_Dist21",
                "LF2022_Dist22",
                "LF2023_Dist23",
            ],
        )
        self.assertTrue(
            all(layer["wkid"] == 5070 for layer in catalog["layers"])
        )

    def test_release_mapping_rejects_future_vegetation(self) -> None:
        with self.assertRaisesRegex(ValueError, "later than observation year"):
            validate_release_by_year(
                [2020],
                ["LF2016", "LF2022"],
                {2020: "LF2022"},
            )


if __name__ == "__main__":
    unittest.main()
