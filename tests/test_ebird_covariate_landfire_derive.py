from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_derive import (  # noqa: E402
    NODATA,
    _apply_lut,
    _integer_lut,
    band_id,
    conditional_neighborhood_mean,
    expected_band_ids,
)


class LandfireDeriveTests(unittest.TestCase):
    def test_integer_lut_handles_negative_nodata_code(self) -> None:
        lookup, minimum = _integer_lut(
            {-9999: None, 11: "water", 7292: "forest"},
            lambda value: -1 if value is None else len(value),
            default=-1,
            dtype=np.int16,
        )
        values = np.array([[-9999, 11, 7292, 99]], dtype=np.int16)
        mapped = _apply_lut(
            values,
            lookup,
            minimum,
            default=-1,
            dtype=np.int16,
        )
        np.testing.assert_array_equal(mapped, [[-1, 5, 6, -1]])

    def test_conditional_mean_does_not_zero_other_lifeforms(self) -> None:
        numerator = np.array(
            [[0.8, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        lifeform = np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        total = np.ones((3, 3), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        values = conditional_neighborhood_mean(
            numerator,
            lifeform,
            total,
            kernel,
            minimum_coverage=0.5,
            minimum_lifeform_fraction=0.2,
        )
        self.assertAlmostEqual(float(values[1, 1]), 0.6, places=6)

    def test_conditional_mean_masks_insufficient_lifeform_support(self) -> None:
        numerator = np.zeros((3, 3), dtype=np.float32)
        numerator[0, 0] = 0.8
        lifeform = np.zeros((3, 3), dtype=np.float32)
        lifeform[0, 0] = 1.0
        total = np.ones((3, 3), dtype=np.float32)
        values = conditional_neighborhood_mean(
            numerator,
            lifeform,
            total,
            np.ones((3, 3), dtype=np.float32),
            minimum_coverage=0.5,
            minimum_lifeform_fraction=0.2,
        )
        self.assertEqual(float(values[1, 1]), NODATA)

    def test_band_id_preserves_release_and_radius(self) -> None:
        self.assertEqual(
            band_id("evt_forest_tree_fraction", 1000, "LF2023", "mean"),
            (
                "availability__landfire__evt_forest_tree_fraction__mean__"
                "r1000__lf2023"
            ),
        )

    def test_expected_schema_retains_logical_all_nodata_bands(self) -> None:
        identifiers = expected_band_ids("LF2023", [250, 1000, 5000])
        self.assertEqual(len(identifiers), 46)
        self.assertEqual(len(set(identifiers)), 46)
        self.assertIn(
            (
                "availability__landfire__source_coverage_fraction__value__"
                "r250__lf2023"
            ),
            identifiers,
        )
        self.assertIn(
            (
                "availability__landfire__dominant_shrub_height_m_conditional__"
                "mean__r5000__lf2023"
            ),
            identifiers,
        )


if __name__ == "__main__":
    unittest.main()
