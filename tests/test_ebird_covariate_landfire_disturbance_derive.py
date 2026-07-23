from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_disturbance_derive import (  # noqa: E402
    classify_source_values,
    disturbance_band_id,
    expected_disturbance_band_ids,
)
from ebird_covariates.landfire_disturbance_qa import (  # noqa: E402
    parse_disturbance_band_id,
)


class LandfireDisturbanceDeriveTests(unittest.TestCase):
    def test_schema_is_year_and_radius_aware(self) -> None:
        identifiers = expected_disturbance_band_ids(
            [2020, 2021, 2022, 2023],
            [250, 1000, 5000],
        )
        self.assertEqual(len(identifiers), 12)
        self.assertEqual(
            identifiers[0],
            disturbance_band_id(2020, 250),
        )
        self.assertEqual(
            parse_disturbance_band_id(identifiers[-1]),
            {"radius_m": 5000, "year": 2023},
        )

    def test_source_classification_excludes_fill_and_water(self) -> None:
        values = np.array([[-1111, 0, 16, 11]], dtype=np.int16)
        source_valid = np.ones_like(values, dtype=bool)
        lookup = [
            {
                "Value": "-1111",
                "is_analysis_support": "false",
                "is_disturbed": "false",
            },
            {
                "Value": "0",
                "is_analysis_support": "true",
                "is_disturbed": "false",
            },
            {
                "Value": "16",
                "is_analysis_support": "false",
                "is_disturbed": "false",
            },
            {
                "Value": "11",
                "is_analysis_support": "true",
                "is_disturbed": "true",
            },
        ]
        support, disturbed = classify_source_values(
            values,
            source_valid,
            lookup,
        )
        np.testing.assert_array_equal(support, [[False, True, False, True]])
        np.testing.assert_array_equal(disturbed, [[False, False, False, True]])


if __name__ == "__main__":
    unittest.main()
