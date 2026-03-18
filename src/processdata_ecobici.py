"""
MEXICO CITY: bike-share data cleaning.

Assumptions:
- Files span from three different city administrations 2018, 2018-2024, 2024-25
- Input files are monthly CSVs named like: 2018-01.csv, 2018-02.csv, ..., 2025-12.csv
- Each file has 9 columns representing:
  gender, age, bike_id, start_station, start_date, start_time, end_station, end_date, end_time
- Some years have names in Spanish and others in English as well as different spellings and symbols (e.g., 'Fecha Arribo' vs 'Fecha_Arribo').

Use:
python src/processdata_ecobici.py
"""

from __future__ import annotations

import glob
import gc
import os
import re 
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd

# =========================================================
# 1) Configuration
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw") #where in directory
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs") #saved cleaned outputs
YEARLY_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "yearly")
FILE_PATTERN = "**/*.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(YEARLY_OUTPUT_FOLDER, exist_ok=True)
# -----------------------------
# 2) Column standardization
# -----------------------------

# Standardized English column naming (9 columns)
english_columns = [
    "user_gender",
    "user_age",
    "bike_id",
    "start_station_id",
    "start_date",
    "start_time",
    "end_station_id",
    "end_date",
    "end_time",
]

# Potential Spanish variants -> canonical English
# Headers are normalized first, then spaces, underscores, etc. 
column_map_normalized: Dict[str, str] = {
    "genero_usuario": "user_gender", #gender variants
    "sexo_usuario": "user_gender",
    "gender": "user_gender",
    "user_gender": "user_gender",

    "edad_usuario": "user_age", #age variants
    "age": "user_age",
    "user_age": "user_age",

    "bici": "bike_id", #bike variants
    "bicicleta": "bike_id",
    "bike": "bike_id",
    "bike_id": "bike_id",

    "ciclo_estacion_retiro": "start_station_id", #id begin
    "cicloestacion_retiro": "start_station_id",
    "estacion_retiro": "start_station_id",
    "start_station": "start_station_id",
    "start_station_id": "start_station_id",

    "fecha_retiro": "start_date", #date take
    "fecha_inicio": "start_date",
    "start_date": "start_date",

    "hora_retiro": "start_time", #time take
    "hora_inicio": "start_time",
    "start_time": "start_time",

    "ciclo_estacion_arribo": "end_station_id", #id end
    "cicloestacion_arribo": "end_station_id",
    "ciclo_estacionarribo": "end_station_id",
    "estacion_arribo": "end_station_id",
    "end_station": "end_station_id",
    "end_station_id": "end_station_id",

    "fecha_arribo": "end_date", #date return
    "fecha_fin": "end_date",
    "end_date": "end_date",

    "hora_arribo": "end_time", #time return
    "hora_fin": "end_time",
    "end_time": "end_time",
}

# -----------------------------
# 3) Helpers
# -----------------------------

def normalize_header(text: str) -> str:
    """Lowercase, remove accent, replace punctuation and spaces with underscores."""
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text

def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Rename raw columns to the desired 9 English columns.
    """
    rename_dict = {}

    for col in df.columns:
        norm_col = normalize_header(col)
        if norm_col in column_map_normalized:
            rename_dict[col] = column_map_normalized[norm_col]

    df = df.rename(columns=rename_dict)

    found = [c for c in english_columns if c in df.columns]
    missing = [c for c in english_columns if c not in df.columns]

    for col in missing:
        df[col] = pd.NA

    df = df[english_columns].copy()
    return df, found, missing

def parse_datetime(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """
    Combine date and time into a single datetime column.
    This second iteration handles:
    - day-first dates like 31/07/21
    - mixed hour formatting like 0:03:31 vs 23:57:44
    - extra spaces
    - inconsistent elements and separators
    """

    date_clean = (
        date_series.astype("string")
        .fillna("")
        .str.strip()
    )

    time_clean = (
        time_series.astype("string") #"0:03:31" -> "00:03:31"
        .fillna("")
        .str.strip()
        .replace({"": pd.NA})
    )

    time_clean = time_clean.str.replace(
        r"^(\d):(\d{2}):(\d{2})$", #H:MM:SS, pad with zero
        r"0\1:\2:\3",
        regex=True
    )

    combined = (date_clean.fillna("") + " " + time_clean.fillna("")).str.strip()

    # Format with 2-digit year
    dt = pd.to_datetime(
        combined,
        format="%d/%m/%y %H:%M:%S",
        errors="coerce"
    )

    # Fallback for rows that could fail
    mask = dt.isna() & combined.ne("")
    if mask.any():
        dt_fallback = pd.to_datetime(
            combined[mask],
            errors="coerce",
            dayfirst=True
        )
        dt.loc[mask] = dt_fallback
    return dt

def clean_gender(series: pd.Series) -> pd.Series:
        s = series.astype("string").str.strip().str.lower()

        mapping = {
            "m": "M",
            "masculino": "M",
            "male": "M",
            "hombre": "M",
            "f": "F",
            "femenino": "F",
            "female": "F",
            "mujer": "F",
        }

        s = s.replace(mapping)
        s = s.where(s.isin(["M", "F"]), pd.NA)
        return s

def clean_age(series: pd.Series, min_age: int = 10, max_age: int = 100) -> pd.Series:
    """
    Convert age to numeric and change unlikely values to NA.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.where((s >= min_age) & (s <= max_age), pd.NA)

def clean_station_id(series: pd.Series) -> pd.Series:
    """
    Convert station IDs to null integer.
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.astype("Int64")

def clean_bike_id(series: pd.Series) -> pd.Series:
    """
    Keep bike IDs as strings.
    """
    s = series.astype("string").str.strip()
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

def extract_file_period(filepath: str) -> Tuple[object, object]:
    """
    Extract year and month from filename, e.g. 2019-03.csv or 2019_03.csv
    """
    filename = os.path.basename(filepath)
    match = re.search(r"(20\d{2})[-_](0[1-9]|1[0-2])", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return pd.NA, pd.NA

def get_all_csv_paths(raw_folder: str, pattern: str = "**/*.csv") -> List[str]:
    paths = sorted(glob.glob(os.path.join(raw_folder, pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {raw_folder}")
    return paths


def get_years_from_paths(paths: List[str]) -> List[int]:
    years = []
    for path in paths:
        year, _ = extract_file_period(path)
        if pd.notna(year):
            years.append(int(year))
    return sorted(set(years))

# -----------------------------
# 4) File generation and cleaning
# -----------------------------

def clean_one_file(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Read and clean one CSV file.
    Returns:
        cleaned dataframe, metadata dictionary
    """
    meta = {
    "file": filepath,
    "rows_raw": 0,
    "rows_cleaned": 0,
    "found_columns": "",
    "missing_columns": "",
    "n_start_dt_missing": 0,
    "n_end_dt_missing": 0,
    "n_negative_duration": 0,
    "n_duration_gt_24h": 0,
    "pct_start_dt_missing": 0.0,
    "pct_end_dt_missing": 0.0,
    "all_start_station_missing": False,
    "all_end_station_missing": False,
    "status": "ok",
    "error": "",
    }   

    try:
        try:
            df = pd.read_csv(filepath, encoding="utf-8", sep=None, engine="python")
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1", sep=None, engine="python")

        meta["rows_raw"] = len(df)

        df, found, missing = standardize_columns(df)
        meta["found_columns"] = ", ".join(found)
        meta["missing_columns"] = ", ".join(missing)

        # Diagnose mapping failures in station columns
        meta["all_start_station_missing"] = bool(df["start_station_id"].isna().all())
        meta["all_end_station_missing"] = bool(df["end_station_id"].isna().all())

        if meta["all_start_station_missing"]:
            print(f"WARNING: all start_station_id values missing in {os.path.basename(filepath)}")

        if meta["all_end_station_missing"]:
            print(f"WARNING: all end_station_id values missing in {os.path.basename(filepath)}")
        
        file_year, file_month = extract_file_period(filepath)
        df["file_year"] = file_year
        df["file_month"] = file_month
        df["source_file"] = os.path.basename(filepath)

        df["user_gender"] = clean_gender(df["user_gender"])
        df["user_age"] = clean_age(df["user_age"])
        df["bike_id"] = clean_bike_id(df["bike_id"])
        df["start_station_id"] = clean_station_id(df["start_station_id"])
        df["end_station_id"] = clean_station_id(df["end_station_id"])

        df["start_dt"] = parse_datetime(df["start_date"], df["start_time"])
        df["end_dt"] = parse_datetime(df["end_date"], df["end_time"])

        df["trip_duration_min"] = (df["end_dt"] - df["start_dt"]).dt.total_seconds() / 60

        df["trip_year"] = df["start_dt"].dt.year.astype("Int64")
        df["trip_month"] = df["start_dt"].dt.month.astype("Int64")
        df["trip_day"] = df["start_dt"].dt.day.astype("Int64")
        df["trip_hour"] = df["start_dt"].dt.hour.astype("Int64")
        df["trip_weekday"] = df["start_dt"].dt.dayofweek.astype("Int64")

        df["flag_missing_start_dt"] = df["start_dt"].isna()
        df["flag_missing_end_dt"] = df["end_dt"].isna()
        df["flag_negative_duration"] = df["trip_duration_min"] < 0
        df["flag_duration_gt_24h"] = df["trip_duration_min"] > 24 * 60
        df["flag_same_station"] = (
            df["start_station_id"].notna() &
            df["end_station_id"].notna() &
            (df["start_station_id"] == df["end_station_id"])
        )

        meta["n_start_dt_missing"] = int(df["flag_missing_start_dt"].sum())
        meta["n_end_dt_missing"] = int(df["flag_missing_end_dt"].sum())
        meta["n_negative_duration"] = int(df["flag_negative_duration"].sum())
        meta["n_duration_gt_24h"] = int(df["flag_duration_gt_24h"].sum())

        if len(df) > 0:
            meta["pct_start_dt_missing"] = round(meta["n_start_dt_missing"] / len(df), 4)
            meta["pct_end_dt_missing"] = round(meta["n_end_dt_missing"] / len(df), 4)

        if meta["pct_start_dt_missing"] > 0.05:
            print(
                f"WARNING: {os.path.basename(filepath)} has high missing start_dt rate "
                f"({meta['pct_start_dt_missing']:.1%})"
            )

        if meta["pct_end_dt_missing"] > 0.05:
            print(
                f"WARNING: {os.path.basename(filepath)} has high missing end_dt rate "
                f"({meta['pct_end_dt_missing']:.1%})"
            )

        df = df.loc[~df["start_dt"].isna()].copy()

        meta["rows_cleaned"] = len(df)
        return df, meta

    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        return pd.DataFrame(), meta

# -----------------------------
# 5) Batch cleaning
# -----------------------------

def clean_files_by_year(raw_folder: str, pattern: str = "**/*.csv") -> pd.DataFrame:
    paths = get_all_csv_paths(raw_folder, pattern)
    years = get_years_from_paths(paths)

    logs = []

    print("\n=== YEARS DETECTED ===")
    print(years)

    for year in years:
        print(f"\n=== PROCESSING YEAR {year} ===")

        year_paths = []
        for p in paths:
            file_year, _ = extract_file_period(p)
            if pd.notna(file_year) and int(file_year) == year:
                year_paths.append(p)

        year_frames = []

        for path in year_paths:
            print(f"Processing: {os.path.basename(path)}")
            cleaned_df, meta = clean_one_file(path)
            logs.append(meta)

            if not cleaned_df.empty:
                year_frames.append(cleaned_df)

        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)

            year_output = os.path.join(
                YEARLY_OUTPUT_FOLDER,
                f"ecobici_trips_{year}.parquet"
            )
            year_df.to_parquet(year_output, index=False)

            print(f"Saved: {year_output}")
            print(f"Rows: {len(year_df):,}")

            del year_df
            del year_frames
            gc.collect()
        else:
            print(f"No cleaned data produced for year {year}")

    log_df = pd.DataFrame(logs)
    return log_df

# -----------------------------
# 6) Parquet full saving outputs 
# -----------------------------

def build_full_parquet_from_yearly() -> None:
    yearly_paths = sorted(glob.glob(os.path.join(YEARLY_OUTPUT_FOLDER, "*.parquet")))

    if not yearly_paths:
        print("No yearly parquet files found. Skipping full parquet build.")
        return

    print("\n=== BUILDING FULL PARQUET FROM YEARLY FILES ===")
    frames = []

    for path in yearly_paths:
        print(f"Loading: {os.path.basename(path)}")
        frames.append(pd.read_parquet(path))

    full_df = pd.concat(frames, ignore_index=True)
    full_output = os.path.join(OUTPUT_FOLDER, "ecobici_trips_clean_full.parquet")
    full_df.to_parquet(full_output, index=False)

    print(f"Saved full parquet: {full_output}")
    print(f"Rows: {len(full_df):,}")

def save_log(log_df: pd.DataFrame) -> None:
    log_csv = os.path.join(OUTPUT_FOLDER, "ecobici_cleaning_log.csv")
    log_df.to_csv(log_csv, index=False)
    print(f"\nSaved cleaning log: {log_csv}")

# -----------------------------
# 7) Saving Outputs
# -----------------------------

if __name__ == "__main__":
    cleaning_log = clean_files_by_year(RAW_FOLDER, FILE_PATTERN)

    print("\n=== CLEANING LOG SUMMARY ===")
    print(f"Files processed: {cleaning_log['file'].nunique():,}")
    print(f"Files failed: {(cleaning_log['status'] == 'failed').sum():,}")

    print("\n=== SAMPLE LOG ===")
    print(cleaning_log.head())

    save_log(cleaning_log)