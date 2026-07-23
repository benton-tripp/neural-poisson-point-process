from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))

from ebird_covariates.landfire_build import (  # noqa: E402
    assemble_landfire_profiles,
    build_landfire_state,
    build_landfire_work_units,
    completed_unit_artifacts,
    expected_state_band_order,
    profile_band_order,
)
from ebird_covariates.raster_engine import write_band_inventory  # noqa: E402


def fixtures() -> tuple[dict, dict, dict, dict]:
    tile_ids = ["xp0000_yp0000", "xp0001_yp0000"]
    plan = {
        "build_id": "fixture",
        "neighborhoods_m": [250, 1000, 5000],
        "grid": {
            "crs": "EPSG:5070",
            "resolution_m": 250,
            "tile_size_m": 100000,
            "snapped_bounds_m": [0, 0, 200000, 100000],
            "bounding_width_cells": 800,
            "bounding_height_cells": 400,
            "aoi_mask_rule": "all_touched",
            "tiles": [
                {
                    "tile_id": tile_id,
                    "bounds_m": [
                        index * 100000,
                        0,
                        (index + 1) * 100000,
                        100000,
                    ],
                }
                for index, tile_id in enumerate(tile_ids)
            ],
        },
        "sources": [
            {
                "id": "landfire",
                "config_overrides": {
                    "release_by_year": {
                        "2020": "LF2016",
                        "2021": "LF2016",
                        "2022": "LF2022",
                    }
                },
            }
        ],
    }
    layers = []
    for release, content_year in (("LF2016", 2016), ("LF2022", 2022)):
        for product in ("EVT", "EVC", "EVH"):
            layers.append(
                {
                    "role": "vegetation_release",
                    "version": release,
                    "acronym": product,
                    "layerName": f"{release}_{product}",
                    "content_year": content_year,
                }
            )
    for year in (2020, 2021):
        layers.append(
            {
                "role": "annual_disturbance",
                "version": "LF2023",
                "acronym": "Dist",
                "layerName": f"LF2023_Dist{str(year)[-2:]}",
                "observation_year": year,
            }
        )
    catalog = {"schema_version": 1, "layers": layers}
    crosswalk = {
        "schema_version": 1,
        "artifacts": [{"id": "evt_model_crosswalk", "path": "fixture.csv"}],
    }
    disturbance = {
        "schema_version": 1,
        "artifacts": [
            {"id": "disturbance_model_lookup", "path": "fixture.csv"}
        ],
    }
    return plan, catalog, crosswalk, disturbance


class LandfireStateBuildTests(unittest.TestCase):
    def test_work_units_are_tile_release_atomic(self) -> None:
        plan, catalog, _, _ = fixtures()
        units = build_landfire_work_units(
            plan,
            catalog,
            Path("state"),
        )
        self.assertEqual(len(units), 6)
        self.assertEqual(
            [unit["unit_id"] for unit in units],
            [
                "vegetation__lf2016__xp0000_yp0000",
                "vegetation__lf2016__xp0001_yp0000",
                "vegetation__lf2022__xp0000_yp0000",
                "vegetation__lf2022__xp0001_yp0000",
                "disturbance__xp0000_yp0000",
                "disturbance__xp0001_yp0000",
            ],
        )

    def test_named_profiles_have_expected_full_schema_counts(self) -> None:
        identifiers = expected_state_band_order(
            ["LF2016", "LF2022", "LF2023"],
            [2020, 2021, 2022, 2023],
            [250, 1000, 5000],
        )
        self.assertEqual(len(identifiers), 150)
        self.assertEqual(len(profile_band_order(identifiers, "core")), 150)
        self.assertEqual(
            len(profile_band_order(identifiers, "no-structure")),
            96,
        )
        self.assertEqual(
            len(profile_band_order(identifiers, "no-disturbance")),
            138,
        )

    def test_completed_unit_requires_every_referenced_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            derived = root / "derived"
            inventory_path = derived / "inventories" / "band.json"
            cog_path = derived / "tiles" / "tile.tif"
            cog_path.parent.mkdir(parents=True)
            cog_path.touch()
            vrt_path = derived / "unit.vrt"
            vrt_path.parent.mkdir(parents=True, exist_ok=True)
            vrt_path.touch()
            write_band_inventory(
                {
                    "schema_version": 1,
                    "build_id": "fixture",
                    "band_id": "fixture_band",
                    "dtype": "float32",
                    "nodata": -9999.0,
                    "resampling": "derived",
                    "tiles": [
                        {
                            "tile_id": "xp0000_yp0000",
                            "path": str(cog_path),
                        }
                    ],
                },
                inventory_path,
            )
            summary_path = derived / "landfire_derived_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "tile_id": "xp0000_yp0000",
                        "release": "LF2023",
                        "band_count": 1,
                        "inventory_paths": [str(inventory_path)],
                        "logical_vrt": str(vrt_path),
                    }
                ),
                encoding="utf-8",
            )
            validation_path = derived / "diagnostics" / "landfire_validation.json"
            validation_path.parent.mkdir(parents=True)
            validation_path.write_text(
                json.dumps(
                    {
                        "tile_id": "xp0000_yp0000",
                        "release": "LF2023",
                        "all_checks_passed": True,
                    }
                ),
                encoding="utf-8",
            )
            unit = {
                "component": "vegetation",
                "release": "LF2023",
                "tile_id": "xp0000_yp0000",
                "derived_dir": str(derived),
            }
            self.assertIsNotNone(completed_unit_artifacts(unit))
            cog_path.unlink()
            self.assertIsNone(completed_unit_artifacts(unit))

    def test_dry_run_is_resumable_and_contract_locked(self) -> None:
        plan, catalog, crosswalk, disturbance = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "state"
            first = build_landfire_state(
                plan=plan,
                catalog=catalog,
                vegetation_crosswalk=crosswalk,
                disturbance_lookup=disturbance,
                output_dir=output_dir,
                dry_run=True,
            )
            second = build_landfire_state(
                plan=plan,
                catalog=catalog,
                vegetation_crosswalk=crosswalk,
                disturbance_lookup=disturbance,
                output_dir=output_dir,
                dry_run=True,
            )
            self.assertFalse(first["resumed_manifest"])
            self.assertTrue(second["resumed_manifest"])
            expected = second["manifest"]["expected_state"]
            self.assertEqual(second["manifest"]["unit_count"], 6)
            self.assertEqual(expected["shared_raster_band_count"], 98)
            self.assertEqual(
                expected["profile_raster_band_counts"],
                {"core": 98, "no-structure": 62, "no-disturbance": 92},
            )
            with self.assertRaisesRegex(ValueError, "different build contract"):
                build_landfire_state(
                    plan=plan,
                    catalog=catalog,
                    vegetation_crosswalk=crosswalk,
                    disturbance_lookup=disturbance,
                    output_dir=output_dir,
                    profiles=["core"],
                    dry_run=True,
                )

    def test_profile_assembly_uses_shared_inventories(self) -> None:
        plan, catalog, _, _ = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units = build_landfire_work_units(
                plan,
                catalog,
                root / "state_build",
                tile_ids=["xp0000_yp0000"],
                releases=["LF2016"],
            )
            summaries = {}
            for unit in units:
                if unit["component"] == "vegetation":
                    identifiers = expected_state_band_order(
                        ["LF2016"], [], plan["neighborhoods_m"]
                    )
                    summary = {
                        "neighborhoods_m": plan["neighborhoods_m"],
                        "inventory_paths": [],
                    }
                else:
                    identifiers = expected_state_band_order(
                        [], [2020, 2021], plan["neighborhoods_m"]
                    )
                    summary = {
                        "neighborhoods_m": plan["neighborhoods_m"],
                        "disturbance_years": [2020, 2021],
                        "inventory_paths": [],
                    }
                inventory_dir = root / "fixtures" / unit["unit_id"]
                for index, identifier in enumerate(identifiers):
                    path = inventory_dir / f"{index:03d}.json"
                    write_band_inventory(
                        {
                            "schema_version": 1,
                            "build_id": "fixture",
                            "band_id": identifier,
                            "dtype": "float32",
                            "nodata": -9999.0,
                            "resampling": "derived",
                            "tiles": [],
                        },
                        path,
                    )
                    summary["inventory_paths"].append(str(path))
                summaries[unit["unit_id"]] = summary
            assembled = assemble_landfire_profiles(
                plan=plan,
                catalog=catalog,
                output_dir=root / "state_build",
                units=units,
                summaries=summaries,
                profiles=["core", "no-structure", "no-disturbance"],
            )
            self.assertEqual(assembled["shared_raster_band_count"], 52)
            self.assertEqual(
                {
                    profile: output["raster_band_count"]
                    for profile, output in assembled["profiles"].items()
                },
                {"core": 52, "no-structure": 34, "no-disturbance": 46},
            )
            self.assertEqual(assembled["shared_inventory_count"], 52)
            for output in assembled["profiles"].values():
                self.assertTrue(Path(output["logical_vrt"]).exists())
                self.assertTrue(Path(output["manifest"]).exists())

    def test_rejects_unknown_profile(self) -> None:
        plan, catalog, crosswalk, disturbance = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                build_landfire_state(
                    plan=plan,
                    catalog=catalog,
                    vegetation_crosswalk=crosswalk,
                    disturbance_lookup=disturbance,
                    output_dir=Path(temporary),
                    profiles=["not-a-profile"],
                    dry_run=True,
                )

    def test_profile_name_cannot_hide_missing_component(self) -> None:
        plan, catalog, crosswalk, disturbance = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ValueError, "core requires components: disturbance"
            ):
                build_landfire_state(
                    plan=plan,
                    catalog=catalog,
                    vegetation_crosswalk=crosswalk,
                    disturbance_lookup=disturbance,
                    output_dir=Path(temporary),
                    profiles=["core"],
                    components=["vegetation"],
                    dry_run=True,
                )

    def test_max_units_caps_failed_attempts(self) -> None:
        plan, catalog, crosswalk, disturbance = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "ebird_covariates.landfire_build._process_unit",
                side_effect=RuntimeError("fixture failure"),
            ) as process:
                result = build_landfire_state(
                    plan=plan,
                    catalog=catalog,
                    vegetation_crosswalk=crosswalk,
                    disturbance_lookup=disturbance,
                    output_dir=Path(temporary),
                    max_units=1,
                    continue_on_error=True,
                )
            self.assertEqual(process.call_count, 1)
            self.assertEqual(result["attempted_unit_count"], 1)
            self.assertEqual(result["manifest"]["failed_unit_count"], 1)


if __name__ == "__main__":
    unittest.main()
