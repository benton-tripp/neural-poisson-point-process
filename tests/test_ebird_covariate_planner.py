from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.planner import (  # noqa: E402
    expected_product_bands,
    plan_tiles,
    snap_bounds,
    time_period_count,
    validate_config,
    validate_registry,
)


class PlannerUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporal = {
            "start_year": 2020,
            "end_year": 2023,
            "months": list(range(1, 13)),
            "seasons": {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            },
            "selection_rule": "latest_not_after",
        }

    def test_snap_bounds_uses_fixed_origin(self) -> None:
        self.assertEqual(
            snap_bounds((-125.0, 249.0, 751.0, 1001.0), 250.0, 0.0, 0.0),
            (-250.0, 0.0, 1000.0, 1250.0),
        )

    def test_temporal_period_counts(self) -> None:
        self.assertEqual(time_period_count("year", self.temporal), 4)
        self.assertEqual(time_period_count("year_month", self.temporal), 48)
        self.assertEqual(time_period_count("year_season", self.temporal), 16)
        self.assertEqual(time_period_count("month_normal", self.temporal), 12)
        self.assertEqual(time_period_count("release", self.temporal), 1)
        self.assertEqual(
            time_period_count(
                "release", self.temporal, ["LF2016", "LF2022", "LF2023"]
            ),
            3,
        )

    def test_all_touched_mask_retains_boundary_intersection_cells(self) -> None:
        geometry = box(1.0, 1.0, 10.0, 10.0)
        center_tiles = plan_tiles(
            geometry,
            resolution=250.0,
            tile_size=500.0,
            origin_x=0.0,
            origin_y=0.0,
            aoi_mask_rule="center",
        )
        touched_tiles = plan_tiles(
            geometry,
            resolution=250.0,
            tile_size=500.0,
            origin_x=0.0,
            origin_y=0.0,
            aoi_mask_rule="all_touched",
        )
        self.assertEqual(center_tiles, [])
        self.assertEqual(len(touched_tiles), 1)
        self.assertEqual(touched_tiles[0]["active_cells_center_rule"], 0)
        self.assertEqual(touched_tiles[0]["active_cells_all_touched_rule"], 1)
        self.assertEqual(touched_tiles[0]["active_cells_selected_mask_rule"], 1)

    def test_expected_bands_expands_time_and_neighborhoods(self) -> None:
        product = {
            "bands": 3,
            "time_axis": "year",
            "spatial_scales": "neighborhoods",
        }
        self.assertEqual(
            expected_product_bands(product, self.temporal, [250.0, 1000.0, 5000.0]),
            36,
        )

    def test_checked_in_registry_and_config_validate(self) -> None:
        import json

        registry_path = (
            PROJECT_ROOT / "scripts" / "data" / "ebird_covariates" / "source_registry.json"
        )
        config_path = PROJECT_ROOT / "config" / "ebird_covariates" / "nc_2020_2023_v1.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        sources = validate_registry(registry)
        validated = validate_config(config, sources)
        self.assertEqual(len(sources), 17)
        self.assertEqual(len(validated["sources"]), 17)
        self.assertEqual(validated["grid"]["aoi_mask_rule"], "all_touched")


if __name__ == "__main__":
    unittest.main()
