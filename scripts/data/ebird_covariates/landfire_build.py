"""Resumable orchestration and profile assembly for statewide LANDFIRE builds."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .landfire_derive import derive_landfire, expected_band_ids
from .landfire_disturbance_derive import (
    derive_landfire_disturbance,
    expected_disturbance_band_ids,
)
from .landfire_disturbance_qa import (
    validate_landfire_disturbance_derivation,
    write_landfire_disturbance_validation,
)
from .landfire_export import export_landfire_tiles
from .landfire_qa import (
    validate_landfire_derivation,
    write_landfire_derivation_validation,
)
from .raster_engine import (
    load_band_inventory,
    safe_artifact_path,
    write_band_inventory,
    write_logical_vrt,
)


PROFILE_DEFINITIONS = {
    "core": {
        "description": (
            "All validated LANDFIRE vegetation structure, EVT fractions, "
            "source coverage, and annual disturbance."
        ),
        "include_structure": True,
        "include_disturbance": True,
        "required_components": {"vegetation", "disturbance"},
    },
    "no-structure": {
        "description": (
            "Sensitivity profile retaining EVT fractions, source coverage, "
            "and annual disturbance while excluding conditional EVC/EVH."
        ),
        "include_structure": False,
        "include_disturbance": True,
        "required_components": {"vegetation", "disturbance"},
    },
    "no-disturbance": {
        "description": (
            "Sensitivity profile retaining all vegetation bands while "
            "excluding annual disturbance."
        ),
        "include_structure": True,
        "include_disturbance": False,
        "required_components": {"vegetation"},
    },
}

COMPONENTS = {"vegetation", "disturbance"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _release_sort_key(release: str) -> tuple[int, str]:
    digits = "".join(character for character in release if character.isdigit())
    return (int(digits) if digits else 0, release)


def _select_tiles(
    plan: dict[str, Any], tile_ids: list[str] | None
) -> list[str]:
    available = [
        str(tile["tile_id"]) for tile in plan.get("grid", {}).get("tiles", [])
    ]
    requested = set(tile_ids or available)
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError(
            "Requested LANDFIRE state-build tiles are absent from the plan: "
            + ", ".join(unknown)
        )
    selected = [tile_id for tile_id in available if tile_id in requested]
    if not selected:
        raise ValueError("At least one LANDFIRE state-build tile is required.")
    return selected


def _selected_catalog_layers(
    catalog: dict[str, Any],
    releases: list[str] | None,
    components: list[str] | None,
) -> tuple[dict[str, list[str]], list[str]]:
    selected_components = set(components or sorted(COMPONENTS))
    unknown_components = sorted(selected_components - COMPONENTS)
    if unknown_components:
        raise ValueError(
            "Unsupported LANDFIRE state-build components: "
            + ", ".join(unknown_components)
        )
    vegetation_layers = [
        layer
        for layer in catalog.get("layers", [])
        if layer.get("role") == "vegetation_release"
    ]
    available_releases = sorted(
        {str(layer["version"]) for layer in vegetation_layers},
        key=_release_sort_key,
    )
    requested_releases = set(releases or available_releases)
    unknown_releases = sorted(requested_releases - set(available_releases))
    if unknown_releases:
        raise ValueError(
            "Requested LANDFIRE releases are absent from the catalog: "
            + ", ".join(unknown_releases)
        )
    layers_by_release: dict[str, list[str]] = {}
    if "vegetation" in selected_components:
        for release in available_releases:
            if release not in requested_releases:
                continue
            release_layers = [
                layer
                for layer in vegetation_layers
                if str(layer["version"]) == release
            ]
            product_to_layer = {
                str(layer["acronym"]): str(layer["layerName"])
                for layer in release_layers
            }
            missing = sorted({"EVT", "EVC", "EVH"} - set(product_to_layer))
            if missing:
                raise ValueError(
                    f"LANDFIRE catalog release {release} lacks products: "
                    + ", ".join(missing)
                )
            layers_by_release[release] = [
                product_to_layer[product] for product in ("EVT", "EVC", "EVH")
            ]
    disturbance_layers: list[str] = []
    if "disturbance" in selected_components:
        disturbance_records = [
            layer
            for layer in catalog.get("layers", [])
            if layer.get("role") == "annual_disturbance"
        ]
        if not disturbance_records:
            raise ValueError("LANDFIRE catalog has no annual disturbance layers.")
        years = [record.get("observation_year") for record in disturbance_records]
        if any(not isinstance(year, int) for year in years):
            raise ValueError(
                "Every LANDFIRE disturbance layer requires observation_year."
            )
        if len(set(years)) != len(years):
            raise ValueError("LANDFIRE catalog has duplicate disturbance years.")
        disturbance_layers = [
            str(record["layerName"])
            for record in sorted(
                disturbance_records,
                key=lambda record: int(record["observation_year"]),
            )
        ]
    return layers_by_release, disturbance_layers


def build_landfire_work_units(
    plan: dict[str, Any],
    catalog: dict[str, Any],
    output_dir: Path,
    *,
    tile_ids: list[str] | None = None,
    releases: list[str] | None = None,
    components: list[str] | None = None,
) -> list[dict[str, Any]]:
    selected_tiles = _select_tiles(plan, tile_ids)
    layers_by_release, disturbance_layers = _selected_catalog_layers(
        catalog,
        releases,
        components,
    )
    units: list[dict[str, Any]] = []
    for release in sorted(layers_by_release, key=_release_sort_key):
        for tile_id in selected_tiles:
            base = (
                output_dir
                / "work"
                / "vegetation"
                / release.lower()
                / tile_id
            )
            units.append(
                {
                    "unit_id": f"vegetation__{release.lower()}__{tile_id}",
                    "component": "vegetation",
                    "release": release,
                    "tile_id": tile_id,
                    "layers": layers_by_release[release],
                    "raw_dir": str((base / "raw").resolve()),
                    "derived_dir": str((base / "derived").resolve()),
                }
            )
    if disturbance_layers:
        for tile_id in selected_tiles:
            base = output_dir / "work" / "disturbance" / tile_id
            units.append(
                {
                    "unit_id": f"disturbance__{tile_id}",
                    "component": "disturbance",
                    "release": None,
                    "tile_id": tile_id,
                    "layers": disturbance_layers,
                    "raw_dir": str((base / "raw").resolve()),
                    "derived_dir": str((base / "derived").resolve()),
                }
            )
    if not units:
        raise ValueError("LANDFIRE state build has no work units.")
    return units


def profile_band_order(
    all_band_ids: list[str],
    profile: str,
) -> list[str]:
    try:
        definition = PROFILE_DEFINITIONS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported LANDFIRE profile: {profile}") from exc
    selected: list[str] = []
    for identifier in all_band_ids:
        is_disturbance = "__annual_disturbance_fraction__" in identifier
        is_structure = "__dominant_" in identifier
        if is_disturbance and not definition["include_disturbance"]:
            continue
        if is_structure and not definition["include_structure"]:
            continue
        selected.append(identifier)
    return selected


def expected_state_band_order(
    releases: list[str],
    disturbance_years: list[int],
    radii: list[int],
) -> list[str]:
    identifiers: list[str] = []
    for release in sorted(releases, key=_release_sort_key):
        identifiers.extend(expected_band_ids(release, radii))
    if disturbance_years:
        identifiers.extend(
            expected_disturbance_band_ids(disturbance_years, radii)
        )
    return identifiers


def _manifest_signature(
    plan: dict[str, Any],
    units: list[dict[str, Any]],
    profiles: list[str],
    source_contract_sha256: dict[str, str],
    *,
    buffer_m: float,
    minimum_coverage: float,
    minimum_lifeform_fraction: float,
) -> str:
    contract = {
        "build_id": plan["build_id"],
        "grid": {
            key: plan["grid"][key]
            for key in (
                "crs",
                "resolution_m",
                "tile_size_m",
                "snapped_bounds_m",
                "aoi_mask_rule",
            )
            if key in plan["grid"]
        },
        "neighborhoods_m": plan["neighborhoods_m"],
        "units": [
            {
                key: unit[key]
                for key in (
                    "unit_id",
                    "component",
                    "release",
                    "tile_id",
                    "layers",
                )
            }
            for unit in units
        ],
        "profiles": profiles,
        "source_contract_sha256": source_contract_sha256,
        "buffer_m": buffer_m,
        "minimum_coverage": minimum_coverage,
        "minimum_lifeform_fraction": minimum_lifeform_fraction,
    }
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _new_manifest(
    plan: dict[str, Any],
    units: list[dict[str, Any]],
    profiles: list[str],
    signature: str,
) -> dict[str, Any]:
    created_at = _now()
    return {
        "schema_version": 1,
        "build_id": plan["build_id"],
        "source_id": "landfire",
        "contract_sha256": signature,
        "created_at_utc": created_at,
        "updated_at_utc": created_at,
        "status": "planned",
        "profiles": profiles,
        "unit_count": len(units),
        "completed_unit_count": 0,
        "failed_unit_count": 0,
        "pending_unit_count": len(units),
        "units": [
            {
                **copy.deepcopy(unit),
                "status": "pending",
                "attempt_count": 0,
                "last_error": None,
            }
            for unit in units
        ],
        "profile_outputs": {},
    }


def _load_or_create_manifest(
    manifest_path: Path,
    plan: dict[str, Any],
    units: list[dict[str, Any]],
    profiles: list[str],
    signature: str,
) -> tuple[dict[str, Any], bool]:
    if not manifest_path.exists():
        manifest = _new_manifest(plan, units, profiles, signature)
        _write_json(manifest, manifest_path)
        return manifest, False
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("LANDFIRE state manifest must use schema_version 1.")
    if manifest.get("contract_sha256") != signature:
        raise ValueError(
            "The existing LANDFIRE state manifest has a different build "
            "contract. Use a different output directory or intentionally "
            "remove the incompatible state build."
        )
    expected_ids = [unit["unit_id"] for unit in units]
    actual_ids = [unit.get("unit_id") for unit in manifest.get("units", [])]
    if actual_ids != expected_ids:
        raise ValueError("LANDFIRE state manifest work-unit ordering changed.")
    return manifest, True


def _unit_artifact_paths(unit: dict[str, Any]) -> tuple[Path, Path]:
    derived_dir = Path(unit["derived_dir"])
    if unit["component"] == "vegetation":
        return (
            derived_dir / "landfire_derived_summary.json",
            derived_dir / "diagnostics" / "landfire_validation.json",
        )
    return (
        derived_dir / "landfire_disturbance_derived_summary.json",
        derived_dir
        / "diagnostics"
        / "landfire_disturbance_validation.json",
    )


def _summary_artifacts_exist(summary: dict[str, Any]) -> bool:
    inventory_paths = [Path(path) for path in summary.get("inventory_paths", [])]
    if len(inventory_paths) != int(summary.get("band_count", -1)):
        return False
    logical_vrt = summary.get("logical_vrt")
    if not logical_vrt or not Path(logical_vrt).exists():
        return False
    try:
        for inventory_path in inventory_paths:
            inventory = load_band_inventory(inventory_path)
            for tile in inventory["tiles"]:
                if not Path(tile["path"]).exists():
                    return False
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def completed_unit_artifacts(
    unit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summary_path, validation_path = _unit_artifact_paths(unit)
    if not summary_path.exists() or not validation_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
        validation = _load_json(validation_path)
    except (OSError, json.JSONDecodeError):
        return None
    if not validation.get("all_checks_passed"):
        return None
    if str(summary.get("tile_id")) != unit["tile_id"]:
        return None
    if str(validation.get("tile_id")) != unit["tile_id"]:
        return None
    if unit["component"] == "vegetation":
        if str(summary.get("release")) != unit["release"]:
            return None
        if str(validation.get("release")) != unit["release"]:
            return None
    elif validation.get("component") != "annual_disturbance":
        return None
    if not _summary_artifacts_exist(summary):
        return None
    return summary, validation


def _validate_existing_summary(
    unit: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summary_path, validation_path = _unit_artifact_paths(unit)
    if not summary_path.exists():
        return None
    try:
        summary = _load_json(summary_path)
        if not _summary_artifacts_exist(summary):
            return None
        if unit["component"] == "vegetation":
            validation = validate_landfire_derivation(plan, summary)
            write_landfire_derivation_validation(validation, validation_path)
        else:
            validation = validate_landfire_disturbance_derivation(plan, summary)
            write_landfire_disturbance_validation(validation, validation_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not validation.get("all_checks_passed"):
        return None
    return summary, validation


def _process_unit(
    unit: dict[str, Any],
    *,
    plan: dict[str, Any],
    catalog: dict[str, Any],
    vegetation_crosswalk: dict[str, Any],
    disturbance_lookup: dict[str, Any],
    buffer_m: float,
    minimum_coverage: float,
    minimum_lifeform_fraction: float,
    timeout: float,
    overwrite: bool,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    if not overwrite:
        existing = _validate_existing_summary(unit, plan)
        if existing is not None:
            return existing[0], existing[1], True
    lookup = (
        vegetation_crosswalk
        if unit["component"] == "vegetation"
        else disturbance_lookup
    )
    export_summary = export_landfire_tiles(
        plan=plan,
        catalog=catalog,
        crosswalk_summary=lookup,
        output_dir=Path(unit["raw_dir"]),
        tile_ids=[unit["tile_id"]],
        layer_names=unit["layers"],
        buffer_m=buffer_m,
        timeout=timeout,
        overwrite=overwrite,
        progress=progress,
    )
    summary_path, validation_path = _unit_artifact_paths(unit)
    if unit["component"] == "vegetation":
        summary = derive_landfire(
            plan=plan,
            export_summary=export_summary,
            crosswalk_summary=vegetation_crosswalk,
            output_dir=summary_path.parent,
            neighborhoods_m=None,
            minimum_coverage=minimum_coverage,
            minimum_lifeform_fraction=minimum_lifeform_fraction,
            overwrite=overwrite,
            write_vrt=True,
            progress=progress,
        )
        validation = validate_landfire_derivation(plan, summary)
        write_landfire_derivation_validation(validation, validation_path)
    else:
        summary = derive_landfire_disturbance(
            plan=plan,
            export_summary=export_summary,
            lookup_summary=disturbance_lookup,
            output_dir=summary_path.parent,
            neighborhoods_m=None,
            minimum_coverage=minimum_coverage,
            overwrite=overwrite,
            write_vrt=True,
            progress=progress,
        )
        validation = validate_landfire_disturbance_derivation(plan, summary)
        write_landfire_disturbance_validation(validation, validation_path)
    if not validation["all_checks_passed"]:
        issues = "; ".join(validation.get("issues", [])[:5])
        raise RuntimeError(
            f"LANDFIRE unit {unit['unit_id']} failed validation: {issues}"
        )
    return summary, validation, False


def _unit_band_order(
    unit: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    radii = [int(value) for value in summary["neighborhoods_m"]]
    if unit["component"] == "vegetation":
        return expected_band_ids(str(unit["release"]), radii)
    return expected_disturbance_band_ids(
        [int(value) for value in summary["disturbance_years"]],
        radii,
    )


def merge_state_inventories(
    units: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    expected_contributors: dict[str, set[str]] = {}
    actual_contributors: dict[str, set[str]] = {}
    order: list[str] = []
    for unit in units:
        summary = summaries[unit["unit_id"]]
        expected = _unit_band_order(unit, summary)
        inventories = [
            load_band_inventory(Path(path))
            for path in summary["inventory_paths"]
        ]
        actual = [str(inventory["band_id"]) for inventory in inventories]
        if actual != expected:
            raise ValueError(
                f"LANDFIRE unit {unit['unit_id']} inventory order changed."
            )
        for inventory in inventories:
            identifier = str(inventory["band_id"])
            expected_contributors.setdefault(identifier, set()).add(
                unit["tile_id"]
            )
            actual_contributors.setdefault(identifier, set()).add(
                unit["tile_id"]
            )
            if identifier not in merged:
                merged[identifier] = {
                    key: copy.deepcopy(inventory[key])
                    for key in (
                        "schema_version",
                        "build_id",
                        "band_id",
                        "dtype",
                        "nodata",
                        "resampling",
                    )
                }
                merged[identifier]["tiles"] = []
                order.append(identifier)
            target = merged[identifier]
            for key in ("build_id", "dtype", "nodata", "resampling"):
                if inventory[key] != target[key]:
                    raise ValueError(
                        f"LANDFIRE band {identifier} has inconsistent {key}."
                    )
            records = copy.deepcopy(inventory["tiles"])
            if len(records) > 1:
                raise ValueError(
                    f"LANDFIRE unit {unit['unit_id']} has multiple tile records "
                    f"for {identifier}."
                )
            if records and records[0]["tile_id"] != unit["tile_id"]:
                raise ValueError(
                    f"LANDFIRE unit {unit['unit_id']} inventory tile mismatch."
                )
            existing_ids = {
                str(record["tile_id"]) for record in target["tiles"]
            }
            if any(str(record["tile_id"]) in existing_ids for record in records):
                raise ValueError(
                    f"Duplicate LANDFIRE tile record for band {identifier}."
                )
            target["tiles"].extend(records)
    for identifier in order:
        if actual_contributors[identifier] != expected_contributors[identifier]:
            raise ValueError(
                f"LANDFIRE band {identifier} is missing a work-unit contribution."
            )
        inventory = merged[identifier]
        inventory["tiles"].sort(key=lambda record: str(record["tile_id"]))
        inventory["tile_count"] = len(inventory["tiles"])
        inventory["valid_cells"] = sum(
            int(record.get("valid_cells", 0)) for record in inventory["tiles"]
        )
    return order, merged


def _source_time_mapping(
    plan: dict[str, Any],
    catalog: dict[str, Any],
    releases: list[str],
) -> list[dict[str, Any]]:
    landfire_sources = [
        source for source in plan["sources"] if source["id"] == "landfire"
    ]
    if len(landfire_sources) != 1:
        raise ValueError("Build plan must contain exactly one LANDFIRE source.")
    release_by_year = landfire_sources[0].get("config_overrides", {}).get(
        "release_by_year", {}
    )
    content_year_by_release: dict[str, int] = {}
    for layer in catalog.get("layers", []):
        release = str(layer.get("version", ""))
        content_year = layer.get("content_year")
        if (
            layer.get("role") == "vegetation_release"
            and release in releases
            and isinstance(content_year, int)
        ):
            previous = content_year_by_release.setdefault(release, content_year)
            if previous != content_year:
                raise ValueError(
                    f"LANDFIRE release {release} has inconsistent content years."
                )
    records: list[dict[str, Any]] = []
    for year_text, release in sorted(
        release_by_year.items(), key=lambda item: int(item[0])
    ):
        if release not in releases:
            continue
        year = int(year_text)
        content_year = content_year_by_release.get(release)
        if content_year is None:
            raise ValueError(
                f"LANDFIRE release {release} lacks a catalog content year."
            )
        records.append(
            {
                "observation_year": year,
                "vegetation_release": release,
                "vegetation_content_year": content_year,
                "vegetation_source_age_years": year - content_year,
            }
        )
    return records


def assemble_landfire_profiles(
    *,
    plan: dict[str, Any],
    catalog: dict[str, Any],
    output_dir: Path,
    units: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    profiles: list[str],
) -> dict[str, Any]:
    state_dir = output_dir / "state"
    inventory_dir = state_dir / "inventories"
    profile_dir = state_dir / "profiles"
    all_band_ids, merged = merge_state_inventories(units, summaries)
    inventory_paths: dict[str, str] = {}
    for identifier in all_band_ids:
        path = safe_artifact_path(inventory_dir, identifier, ".json")
        write_band_inventory(merged[identifier], path)
        inventory_paths[identifier] = str(path.resolve())
    tile_ids = list(
        dict.fromkeys(unit["tile_id"] for unit in units)
    )
    releases = sorted(
        {
            str(unit["release"])
            for unit in units
            if unit["component"] == "vegetation"
        },
        key=_release_sort_key,
    )
    source_time_mapping = _source_time_mapping(plan, catalog, releases)
    profile_outputs: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        definition = PROFILE_DEFINITIONS[profile]
        band_ids = profile_band_order(all_band_ids, profile)
        if not band_ids:
            raise ValueError(f"LANDFIRE profile {profile} has no raster bands.")
        target_dir = profile_dir / profile
        vrt_path = target_dir / f"landfire_{profile.replace('-', '_')}.vrt"
        selected_inventories = [merged[identifier] for identifier in band_ids]
        write_logical_vrt(plan, selected_inventories, vrt_path)
        manifest_path = target_dir / "profile_manifest.json"
        profile_manifest = {
            "schema_version": 1,
            "generated_at_utc": _now(),
            "build_id": plan["build_id"],
            "source_id": "landfire",
            "profile": profile,
            "description": definition["description"],
            "scope": {
                "tile_ids": tile_ids,
                "releases": releases,
                "components": sorted(
                    {str(unit["component"]) for unit in units}
                ),
            },
            "raster_band_count": len(band_ids),
            "band_ids": band_ids,
            "inventory_paths": [
                inventory_paths[identifier] for identifier in band_ids
            ],
            "logical_vrt": str(vrt_path.resolve()),
            "row_scalar_fields": [
                "landfire_vegetation_release",
                "landfire_vegetation_source_age_years",
            ],
            "vegetation_release_by_observation_year": source_time_mapping,
            "storage_policy": (
                "Profiles share the same derived COGs and merged inventories; "
                "only VRT and manifest metadata are profile-specific."
            ),
        }
        _write_json(profile_manifest, manifest_path)
        profile_outputs[profile] = {
            "raster_band_count": len(band_ids),
            "logical_vrt": str(vrt_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        }
    summary = {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "build_id": plan["build_id"],
        "source_id": "landfire",
        "tile_count": len(tile_ids),
        "tile_ids": tile_ids,
        "release_count": len(releases),
        "releases": releases,
        "component_count": len({unit["component"] for unit in units}),
        "work_unit_count": len(units),
        "shared_raster_band_count": len(all_band_ids),
        "shared_inventory_count": len(inventory_paths),
        "shared_inventory_paths": [
            inventory_paths[identifier] for identifier in all_band_ids
        ],
        "profiles": profile_outputs,
        "vegetation_release_by_observation_year": source_time_mapping,
    }
    summary_path = state_dir / "landfire_state_summary.json"
    _write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def _refresh_manifest_counts(manifest: dict[str, Any]) -> None:
    completed = sum(
        unit["status"] == "completed" for unit in manifest["units"]
    )
    failed = sum(unit["status"] == "failed" for unit in manifest["units"])
    manifest["completed_unit_count"] = completed
    manifest["failed_unit_count"] = failed
    manifest["pending_unit_count"] = len(manifest["units"]) - completed
    if completed == len(manifest["units"]):
        manifest["status"] = (
            "complete"
            if len(manifest.get("profile_outputs", {}))
            == len(manifest.get("profiles", []))
            else "units_complete"
        )
    elif failed:
        manifest["status"] = "partial_with_failures"
    elif completed:
        manifest["status"] = "partial"
    else:
        manifest["status"] = "planned"
    manifest["updated_at_utc"] = _now()


def _json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_landfire_state(
    *,
    plan: dict[str, Any],
    catalog: dict[str, Any],
    vegetation_crosswalk: dict[str, Any],
    disturbance_lookup: dict[str, Any],
    output_dir: Path,
    profiles: list[str] | None = None,
    tile_ids: list[str] | None = None,
    releases: list[str] | None = None,
    components: list[str] | None = None,
    buffer_m: float = 5000.0,
    minimum_coverage: float = 0.8,
    minimum_lifeform_fraction: float = 0.01,
    timeout: float = 300.0,
    max_units: int | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
    overwrite: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    selected_profiles = list(profiles or PROFILE_DEFINITIONS)
    if len(selected_profiles) != len(set(selected_profiles)):
        raise ValueError("LANDFIRE state-build profile names must be unique.")
    unknown_profiles = sorted(
        set(selected_profiles) - set(PROFILE_DEFINITIONS)
    )
    if unknown_profiles:
        raise ValueError(
            "Unsupported LANDFIRE profiles: " + ", ".join(unknown_profiles)
        )
    if not selected_profiles:
        raise ValueError("At least one LANDFIRE profile is required.")
    selected_components = set(components or COMPONENTS)
    for profile in selected_profiles:
        missing_components = (
            PROFILE_DEFINITIONS[profile]["required_components"]
            - selected_components
        )
        if missing_components:
            raise ValueError(
                f"LANDFIRE profile {profile} requires components: "
                + ", ".join(sorted(missing_components))
            )
    if max_units is not None and max_units <= 0:
        raise ValueError("LANDFIRE max_units must be positive.")
    if buffer_m < max(int(value) for value in plan["neighborhoods_m"]):
        raise ValueError(
            "LANDFIRE state-build buffer must cover the largest neighborhood."
        )
    units = build_landfire_work_units(
        plan,
        catalog,
        output_dir,
        tile_ids=tile_ids,
        releases=releases,
        components=components,
    )
    source_contract_sha256 = {
        "catalog": _json_sha256(catalog),
        "vegetation_crosswalk": _json_sha256(vegetation_crosswalk),
        "disturbance_lookup": _json_sha256(disturbance_lookup),
    }
    signature = _manifest_signature(
        plan,
        units,
        selected_profiles,
        source_contract_sha256,
        buffer_m=buffer_m,
        minimum_coverage=minimum_coverage,
        minimum_lifeform_fraction=minimum_lifeform_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "landfire_state_build_manifest.json"
    manifest, resumed_manifest = _load_or_create_manifest(
        manifest_path,
        plan,
        units,
        selected_profiles,
        signature,
    )
    planned_releases = sorted(
        {
            str(unit["release"])
            for unit in units
            if unit["component"] == "vegetation"
        },
        key=_release_sort_key,
    )
    disturbance_layer_names = {
        layer_name
        for unit in units
        if unit["component"] == "disturbance"
        for layer_name in unit["layers"]
    }
    planned_disturbance_years = sorted(
        int(layer["observation_year"])
        for layer in catalog.get("layers", [])
        if layer.get("layerName") in disturbance_layer_names
    )
    planned_band_ids = expected_state_band_order(
        planned_releases,
        planned_disturbance_years,
        [int(value) for value in plan["neighborhoods_m"]],
    )
    manifest["expected_state"] = {
        "shared_raster_band_count": len(planned_band_ids),
        "profile_raster_band_counts": {
            profile: len(profile_band_order(planned_band_ids, profile))
            for profile in selected_profiles
        },
        "releases": planned_releases,
        "disturbance_years": planned_disturbance_years,
    }
    manifest["source_contract_sha256"] = source_contract_sha256
    manifest["settings"] = {
        "buffer_m": buffer_m,
        "minimum_coverage": minimum_coverage,
        "minimum_lifeform_fraction": minimum_lifeform_fraction,
        "timeout": timeout,
    }
    if dry_run:
        _refresh_manifest_counts(manifest)
        _write_json(manifest, manifest_path)
        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path.resolve()),
            "resumed_manifest": resumed_manifest,
            "dry_run": True,
            "processed_unit_count": 0,
            "reused_unit_count": 0,
            "elapsed_seconds": time.perf_counter() - started_at,
        }

    states = {state["unit_id"]: state for state in manifest["units"]}
    summaries: dict[str, dict[str, Any]] = {}
    processed = 0
    attempted = 0
    reused = 0
    for unit_index, unit in enumerate(units, start=1):
        state = states[unit["unit_id"]]
        existing = None if overwrite else completed_unit_artifacts(unit)
        if existing is not None:
            summaries[unit["unit_id"]] = existing[0]
            state.update(
                {
                    "status": "completed",
                    "summary_path": str(_unit_artifact_paths(unit)[0].resolve()),
                    "validation_path": str(
                        _unit_artifact_paths(unit)[1].resolve()
                    ),
                    "last_error": None,
                    "reused": True,
                }
            )
            reused += 1
            _refresh_manifest_counts(manifest)
            _write_json(manifest, manifest_path)
            if progress:
                print(
                    f"LANDFIRE unit {unit_index}/{len(units)} reused: "
                    f"{unit['unit_id']}"
                )
            continue
        if max_units is not None and attempted >= max_units:
            continue
        attempted += 1
        state["status"] = "running"
        state["attempt_count"] = int(state.get("attempt_count", 0)) + 1
        state["started_at_utc"] = _now()
        state["last_error"] = None
        manifest["status"] = "running"
        manifest["updated_at_utc"] = _now()
        _write_json(manifest, manifest_path)
        if progress:
            print(
                f"LANDFIRE unit {unit_index}/{len(units)}: "
                f"{unit['unit_id']}"
            )
        try:
            summary, validation, recovered = _process_unit(
                unit,
                plan=plan,
                catalog=catalog,
                vegetation_crosswalk=vegetation_crosswalk,
                disturbance_lookup=disturbance_lookup,
                buffer_m=buffer_m,
                minimum_coverage=minimum_coverage,
                minimum_lifeform_fraction=minimum_lifeform_fraction,
                timeout=timeout,
                overwrite=overwrite,
                progress=progress,
            )
            summaries[unit["unit_id"]] = summary
            summary_path, validation_path = _unit_artifact_paths(unit)
            state.update(
                {
                    "status": "completed",
                    "completed_at_utc": _now(),
                    "summary_path": str(summary_path.resolve()),
                    "validation_path": str(validation_path.resolve()),
                    "last_error": None,
                    "reused": recovered,
                }
            )
            if recovered:
                reused += 1
            processed += 1
        except Exception as exc:
            state.update(
                {
                    "status": "failed",
                    "failed_at_utc": _now(),
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            )
            _refresh_manifest_counts(manifest)
            _write_json(manifest, manifest_path)
            if not continue_on_error:
                raise
        _refresh_manifest_counts(manifest)
        _write_json(manifest, manifest_path)

    for unit in units:
        if unit["unit_id"] in summaries:
            continue
        existing = completed_unit_artifacts(unit)
        if existing is not None:
            summaries[unit["unit_id"]] = existing[0]
    if len(summaries) == len(units):
        state_summary = assemble_landfire_profiles(
            plan=plan,
            catalog=catalog,
            output_dir=output_dir,
            units=units,
            summaries=summaries,
            profiles=selected_profiles,
        )
        manifest["profile_outputs"] = state_summary["profiles"]
        manifest["state_summary_path"] = state_summary["summary_path"]
        for state in manifest["units"]:
            state["status"] = "completed"
        _refresh_manifest_counts(manifest)
    else:
        _refresh_manifest_counts(manifest)
    _write_json(manifest, manifest_path)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path.resolve()),
        "resumed_manifest": resumed_manifest,
        "dry_run": False,
        "attempted_unit_count": attempted,
        "processed_unit_count": processed,
        "reused_unit_count": reused,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
