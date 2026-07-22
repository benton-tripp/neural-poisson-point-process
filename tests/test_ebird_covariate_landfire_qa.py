from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_qa import (  # noqa: E402
    parse_band_id,
    variable_range,
)


class LandfireQaTests(unittest.TestCase):
    def test_parses_release_aware_band_id(self) -> None:
        parsed = parse_band_id(
            "availability__landfire__evt_riparian_fraction__mean__"
            "r1000__lf2023"
        )
        self.assertEqual(
            parsed,
            {
                "variable": "evt_riparian_fraction",
                "statistic": "mean",
                "radius_m": 1000,
                "release": "LF2023",
            },
        )

    def test_assigns_fraction_and_height_range_contracts(self) -> None:
        self.assertEqual(
            variable_range("dominant_tree_cover_fraction_conditional", 100.0),
            (0.0, 1.0),
        )
        self.assertEqual(
            variable_range("dominant_tree_height_m_conditional", 100.0),
            (0.0, 100.0),
        )

    def test_rejects_unknown_variable_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "range contract"):
            variable_range("unknown", 100.0)


if __name__ == "__main__":
    unittest.main()
