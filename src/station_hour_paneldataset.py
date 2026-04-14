"""
MEXICO CITY: bike-share data cleaning.

Assumptions:
- Due to memory, use of duckDB is introduced for the full year parquet.
- Each yearly file is integrated into a station-hour parquet, with a lean panel:
    station_id
    datetime_hour
    departures
    arrivals
    net_flow
    abs_net_flow
    year
    month
    day
    hour
    weekday
    is_weekend
    is_morning_peak
    is_evening_peak

- The purpose of this parquet is to serve for an exploratory EDA prior to the feature engineering.

Use:
python src/station_hour_paneldataset.py

"""

from __future__ import annotations
import os
from pathlib import Path
import duckdb
import pandas as pd

# =========================================================
# 1) Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
YEARLY_DIR = BASE_DIR / "outputs" / "yearly"
OUTPUT_DIR = BASE_DIR / "outputs"

STATION_HOUR_PARQUET = OUTPUT_DIR / "ecobici_station_hour.parquet"
STATION_HOUR_CSV = OUTPUT_DIR / "ecobici_station_hour_sample.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2) Validation of parquets
# =========================================================

def validate_inputs() -> list[Path]:
    yearly_files = sorted(YEARLY_DIR.glob("ecobici_trips_*.parquet"))
    if not yearly_files:
        raise FileNotFoundError(
            f"No yearly parquet files found in: {YEARLY_DIR}"
        )

    print("Yearly parquet files found:")
    for f in yearly_files:
        print("-", f.name)

    return yearly_files


# =========================================================
# 3) Building the station-hour panel
# =========================================================

def build_station_hour_panel() -> None:
    yearly_files = validate_inputs()

    con = duckdb.connect()

    parquet_glob = str(YEARLY_DIR / "ecobici_trips_*.parquet")

    print("\n=== DEBUG: Files matched by glob ===")
    for f in YEARLY_DIR.glob("ecobici_trips_*.parquet"):
        print(f)

    print("\n=== DEBUG: Checking raw parquet content ===")

    check_df = con.execute(f"""
        SELECT
            MIN(start_dt) AS min_start,
            MAX(start_dt) AS max_start,
            COUNT(*) AS n_rows
        FROM read_parquet('{parquet_glob}')
    """).df()

    print(check_df)

    # Build departures panel
    departures_sql = f"""
        CREATE OR REPLACE TEMP TABLE departures AS
        SELECT
            CAST(start_station_id AS BIGINT) AS station_id,
            DATE_TRUNC('hour', start_dt) AS datetime_hour,
            COUNT(*) AS departures
        FROM read_parquet('{parquet_glob}')
        WHERE start_station_id IS NOT NULL
          AND start_dt IS NOT NULL
        GROUP BY 1, 2
    """

    # Build arrivals panel
    arrivals_sql = f"""
        CREATE OR REPLACE TEMP TABLE arrivals AS
        SELECT
            CAST(end_station_id AS BIGINT) AS station_id,
            DATE_TRUNC('hour', end_dt) AS datetime_hour,
            COUNT(*) AS arrivals
        FROM read_parquet('{parquet_glob}')
        WHERE end_station_id IS NOT NULL
          AND end_dt IS NOT NULL
        GROUP BY 1, 2
    """

    con.execute(departures_sql)
    con.execute(arrivals_sql)

    # Full station-hour panel
    panel_sql = f"""
        COPY (
            WITH station_hour AS (
                SELECT
                    COALESCE(d.station_id, a.station_id) AS station_id,
                    COALESCE(d.datetime_hour, a.datetime_hour) AS datetime_hour,
                    COALESCE(d.departures, 0) AS departures,
                    COALESCE(a.arrivals, 0) AS arrivals
                FROM departures d
                FULL OUTER JOIN arrivals a
                    ON d.station_id = a.station_id
                   AND d.datetime_hour = a.datetime_hour
            )
            SELECT
                station_id,
                datetime_hour,
                departures,
                arrivals,
                departures - arrivals AS net_flow,
                ABS(departures - arrivals) AS abs_net_flow,

                EXTRACT(YEAR FROM datetime_hour) AS year,
                EXTRACT(MONTH FROM datetime_hour) AS month,
                EXTRACT(DAY FROM datetime_hour) AS day,
                EXTRACT(HOUR FROM datetime_hour) AS hour,
                EXTRACT(DOW FROM datetime_hour) AS weekday,

                CASE
                    WHEN EXTRACT(DOW FROM datetime_hour) IN (0, 6) THEN 1
                    ELSE 0
                END AS is_weekend,

                CASE
                    WHEN EXTRACT(HOUR FROM datetime_hour) BETWEEN 7 AND 9 THEN 1
                    ELSE 0
                END AS is_morning_peak,

                CASE
                    WHEN EXTRACT(HOUR FROM datetime_hour) BETWEEN 17 AND 19 THEN 1
                    ELSE 0
                END AS is_evening_peak

            FROM station_hour
            ORDER BY datetime_hour, station_id
        )
        TO '{STATION_HOUR_PARQUET}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(panel_sql)

    # Summary
    summary = con.execute(f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT station_id) AS n_stations,
            MIN(datetime_hour) AS min_hour,
            MAX(datetime_hour) AS max_hour,
            SUM(departures) AS total_departures,
            SUM(arrivals) AS total_arrivals
        FROM read_parquet('{STATION_HOUR_PARQUET}')
    """).df()

    print("\n=== STATION-HOUR PANEL BUILT ===")
    print(summary.to_string(index=False))
    print(f"\nSaved parquet: {STATION_HOUR_PARQUET}")
    con.close()

# =========================================================
# 4) Output panel
# =========================================================

if __name__ == "__main__":
    build_station_hour_panel()