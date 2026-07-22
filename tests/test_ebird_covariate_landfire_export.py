from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_export import (  # noqa: E402
    export_landfire_tiles,
    snap_bounds_to_source_grid,
)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.headers = {"Content-Type": "image/tiff"}

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size=None):
        yield self.content


class FakeSession:
    def __init__(self, content: bytes):
        self.content = content
        self.requests = []

    def get(self, url, params=None, timeout=None, stream=None, headers=None):
        self.requests.append({"url": url, "params": params})
        return FakeResponse(self.content)


def tiff_bytes(values: np.ndarray) -> bytes:
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=values.shape[1],
            height=values.shape[0],
            count=1,
            dtype=values.dtype,
            crs="EPSG:5070",
            transform=from_origin(0, 60, 30, 30),
            nodata=-9999,
        ) as dataset:
            dataset.write(values, 1)
        return memory.read()


def write_lookup(path: Path, release: str, values: list[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["release", "Value"])
        writer.writeheader()
        writer.writerows(
            {"release": release, "Value": value}
            for value in values
        )


def fixtures(root: Path):
    artifacts = []
    for product, artifact_id in (
        ("EVT", "evt_model_crosswalk"),
        ("EVC", "evc_model_lookup"),
        ("EVH", "evh_model_lookup"),
    ):
        path = root / f"{product}.csv"
        write_lookup(path, "LF2023", [1, 2])
        artifacts.append({"id": artifact_id, "path": str(path)})
    crosswalk = {"artifacts": artifacts}
    plan = {
        "build_id": "fixture",
        "grid": {
            "crs": "EPSG:5070",
            "tiles": [
                {
                    "tile_id": "xp0000_yp0000",
                    "bounds_m": [0, 0, 60, 60],
                }
            ],
        },
    }
    catalog = {
        "layers": [
            {
                "role": "vegetation_release",
                "layerName": "LF2023_EVT",
                "version": "LF2023",
                "acronym": "EVT",
                "image_server_url": (
                    "https://example.test/LF2023_EVT_CONUS/ImageServer"
                ),
                "pixel_size_m": [30, 30],
                "extent": {"xmin": 0, "ymin": 0, "xmax": 60, "ymax": 60},
                "max_image_width": 100,
                "max_image_height": 100,
            }
        ]
    }
    return plan, catalog, crosswalk


class LandfireExportTests(unittest.TestCase):
    def test_source_grid_snapping_expands_partial_pixels(self) -> None:
        bounds, width, height = snap_bounds_to_source_grid(
            [1, 1, 59, 59],
            {"xmin": 0, "ymax": 60},
            30,
        )
        self.assertEqual(bounds, [0, 0, 60, 60])
        self.assertEqual((width, height), (2, 2))

    def test_exports_and_validates_release_codes(self) -> None:
        values = np.array([[1, 2], [2, 1]], dtype=np.int16)
        session = FakeSession(tiff_bytes(values))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, catalog, crosswalk = fixtures(root)
            summary = export_landfire_tiles(
                plan=plan,
                catalog=catalog,
                crosswalk_summary=crosswalk,
                output_dir=root / "exports",
                tile_ids=["xp0000_yp0000"],
                layer_names=["LF2023_EVT"],
                buffer_m=0,
                session=session,
            )

        self.assertEqual(summary["export_count"], 1)
        self.assertTrue(summary["all_checks_passed"])
        self.assertEqual(summary["exports"][0]["unique_value_count"], 2)
        self.assertTrue(
            session.requests[0]["url"].endswith("/ImageServer/exportImage")
        )

    def test_rejects_code_absent_from_release_lookup(self) -> None:
        values = np.array([[1, 3], [2, 1]], dtype=np.int16)
        session = FakeSession(tiff_bytes(values))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, catalog, crosswalk = fixtures(root)
            with self.assertRaisesRegex(ValueError, "absent from its lookup"):
                export_landfire_tiles(
                    plan=plan,
                    catalog=catalog,
                    crosswalk_summary=crosswalk,
                    output_dir=root / "exports",
                    tile_ids=["xp0000_yp0000"],
                    layer_names=["LF2023_EVT"],
                    buffer_m=0,
                    session=session,
                )


if __name__ == "__main__":
    unittest.main()
