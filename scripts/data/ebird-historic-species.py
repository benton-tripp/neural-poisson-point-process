"""
Retrieve eBird historic observations for one species over a date range.

The eBird historic endpoint returns observations for all taxa on a given
region/date, so this script fetches each date and filters the response
locally by species code, common name, or scientific name.

Requires an eBird API token in .env:

    EBIRD_API_KEY=your_token_here

Examples:

    python scripts/data/ebird-historic-species.py --region US-NC-183 --start 2025-05-01 --end 2025-05-31 --species-code norcar
    python scripts/data/ebird-historic-species.py --region US-NC --start 2025-01-01 --end 2025-01-07 --species "Northern Cardinal"
    python scripts/data/ebird-historic-species.py --region US-NC --start 2020-01-01 --end 2020-12-31 --species-code woothr --output data/wood_thrush_nc_2020.csv
    python scripts/data/ebird-historic-species.py --region US-NC --start 2021-01-01 --end 2021-12-31 --species-code woothr --output data/wood_thrush_nc_2021.csv
    python scripts/data/ebird-historic-species.py --region US-NC --start 2022-01-01 --end 2022-12-31 --species-code woothr --output data/wood_thrush_nc_2022.csv
    python scripts/data/ebird-historic-species.py --region US-NC --start 2023-01-01 --end 2023-12-31 --species-code woothr --output data/wood_thrush_nc_2023.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


EBIRD_HISTORIC_URL = "https://api.ebird.org/v2/data/obs/{region}/historic/{year}/{month}/{day}"
DEFAULT_OUTPUT = "data/ebird_historic_species_observations.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch eBird historic observations for a species in a region/date range."
    )
    parser.add_argument(
        "--region",
        required=True,
        help="eBird region, subregion, or location code, e.g. US, US-NC, US-NC-183, L123456.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_iso_date,
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_iso_date,
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--species-code",
        help="Exact eBird species code to match, e.g. norcar for Northern Cardinal.",
    )
    parser.add_argument(
        "--species",
        help="Common or scientific name to match case-insensitively if --species-code is not used.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json"),
        default="csv",
        help="Output format. Defaults to csv.",
    )
    parser.add_argument(
        "--detail",
        choices=("simple", "full"),
        default="simple",
        help="eBird detail level. Defaults to simple.",
    )
    parser.add_argument(
        "--rank",
        choices=("mrec", "create"),
        default="mrec",
        help="Use latest observation of the day (mrec) or first added (create).",
    )
    parser.add_argument(
        "--cat",
        action="append",
        help="Taxonomic category filter. Can be repeated, e.g. --cat species --cat issf.",
    )
    parser.add_argument(
        "--hotspot",
        action="store_true",
        help="Only fetch observations from hotspots.",
    )
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Include observations that have not yet been reviewed.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        help="Maximum results per date returned by eBird, from 1 to 10000.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        help="Optional eBird location codes. eBird supports up to 50 values.",
    )
    parser.add_argument(
        "--locale",
        default="en",
        help="Species common-name locale. Defaults to en.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Seconds to pause between daily requests. Defaults to 0.25.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Request timeout in seconds. Defaults to 120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries for timeouts, connection errors, HTTP 429, and HTTP 5xx. Defaults to 5.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing output/progress files and start from --start.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format.") from exc


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def progress_path_for(output_path: str) -> Path:
    path = Path(output_path)
    return path.with_suffix(f"{path.suffix}.progress.json")


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("EBIRD_API_KEY")
    if not api_key:
        raise RuntimeError("EBIRD_API_KEY is not set. Add it to .env or your environment.")
    return api_key


def build_query(args: argparse.Namespace) -> dict[str, Any]:
    query: dict[str, Any] = {
        "detail": args.detail,
        "rank": args.rank,
        "sppLocale": args.locale,
        "hotspot": str(args.hotspot).lower(),
        "includeProvisional": str(args.include_provisional).lower(),
    }

    if args.cat:
        query["cat"] = args.cat
    if args.max_results is not None:
        if args.max_results < 1 or args.max_results > 10000:
            raise ValueError("--max-results must be between 1 and 10000.")
        query["maxResults"] = args.max_results
    if args.locations:
        if len(args.locations) > 50:
            raise ValueError("--locations accepts at most 50 location codes.")
        query["r"] = args.locations

    return query


def fetch_observations_for_date(
    session: requests.Session,
    region: str,
    day: date,
    query: dict[str, Any],
    timeout: float,
    retries: int,
) -> list[dict[str, Any]]:
    url = EBIRD_HISTORIC_URL.format(
        region=region,
        year=day.year,
        month=day.month,
        day=day.day,
    )
    retryable_statuses = {429, 500, 502, 503, 504}
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, params=query, timeout=timeout)
            if response.status_code in retryable_statuses and attempt < attempts:
                wait_seconds = min(2 ** (attempt - 1), 60)
                print(
                    f"{day.isoformat()}: HTTP {response.status_code}; "
                    f"retrying in {wait_seconds}s ({attempt}/{retries})"
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"eBird request failed for {day.isoformat()} after "
                    f"{attempts} attempts: {exc}"
                ) from exc

            wait_seconds = min(2 ** (attempt - 1), 60)
            print(
                f"{day.isoformat()}: request timed out or connection failed; "
                f"retrying in {wait_seconds}s ({attempt}/{retries})"
            )
            time.sleep(wait_seconds)

        except requests.HTTPError as exc:
            raise RuntimeError(
                f"eBird request failed for {day.isoformat()} with HTTP "
                f"{response.status_code}: {response.url}\n{response.text}"
            ) from exc

    raise RuntimeError(f"eBird request failed for {day.isoformat()} unexpectedly.")


def matches_species(
    observation: dict[str, Any],
    species_code: str | None,
    species_name: str | None,
) -> bool:
    if species_code:
        return observation.get("speciesCode", "").lower() == species_code.lower()

    if species_name:
        needle = species_name.lower()
        common_name = str(observation.get("comName", "")).lower()
        scientific_name = str(observation.get("sciName", "")).lower()
        return needle in common_name or needle in scientific_name

    return True


def load_existing_records(output_path: str, output_format: str) -> list[dict[str, Any]]:
    path = Path(output_path)
    if not path.exists():
        return []

    if output_format == "json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Existing JSON output is not a list: {output_path}")
        return records

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_last_processed_date(output_path: str, records: list[dict[str, Any]]) -> date | None:
    progress_path = progress_path_for(output_path)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        last_processed = progress.get("lastProcessedDate")
        if last_processed:
            return parse_iso_date(last_processed)

    record_dates = [
        parse_iso_date(str(record["queryDate"]))
        for record in records
        if record.get("queryDate")
    ]
    return max(record_dates) if record_dates else None


def write_progress(output_path: str, last_processed: date, record_count: int) -> None:
    path = progress_path_for(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = {
        "lastProcessedDate": last_processed.isoformat(),
        "recordCount": record_count,
        "updatedAt": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def fetch_species_observations(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.start > args.end:
        raise ValueError("--start must be on or before --end.")
    if not args.species_code and not args.species:
        raise ValueError("Provide --species-code or --species.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than 0.")
    if args.retries < 0:
        raise ValueError("--retries must be 0 or greater.")

    api_key = get_api_key()
    query = build_query(args)

    session = requests.Session()
    session.headers.update({"X-eBirdApiToken": api_key})

    matched: list[dict[str, Any]] = []
    start_date = args.start

    if not args.no_resume:
        matched = load_existing_records(args.output, args.format)
        last_processed = load_last_processed_date(args.output, matched)
        if last_processed is not None:
            start_date = max(args.start, last_processed + timedelta(days=1))
            if start_date <= args.end:
                print(
                    f"Resuming from {start_date.isoformat()} "
                    f"after {last_processed.isoformat()} "
                    f"({len(matched):,} existing records)"
                )
            else:
                print(
                    f"Output is already complete through {last_processed.isoformat()} "
                    f"({len(matched):,} records)"
                )
                return matched

    total_dates = (args.end - start_date).days + 1

    try:
        for index, current_day in enumerate(iter_dates(start_date, args.end), start=1):
            observations = fetch_observations_for_date(
                session=session,
                region=args.region,
                day=current_day,
                query=query,
                timeout=args.timeout,
                retries=args.retries,
            )
            day_matches = [
                {**obs, "queryDate": current_day.isoformat(), "queryRegion": args.region}
                for obs in observations
                if matches_species(obs, args.species_code, args.species)
            ]
            matched.extend(day_matches)
            write_output(matched, args.output, args.format)
            write_progress(args.output, current_day, len(matched))

            print(
                f"{current_day.isoformat()} ({index}/{total_dates}): "
                f"{len(day_matches)} matches from {len(observations)} observations"
            )

            if index < total_dates and args.pause > 0:
                time.sleep(args.pause)

    except Exception:
        write_output(matched, args.output, args.format)
        print(
            f"\nSaved checkpoint with {len(matched):,} records to {args.output}. "
            "Rerun the same command to resume."
        )
        raise

    return matched


def write_output(records: list[dict[str, Any]], output_path: str, output_format: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return

    fieldnames = sorted({key for record in records for key in record.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    records = fetch_species_observations(args)
    write_output(records, args.output, args.format)
    print(f"\nSaved {len(records):,} matching observations to {args.output}")


if __name__ == "__main__":
    main()
