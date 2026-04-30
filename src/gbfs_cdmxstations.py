import requests
import pandas as pd
import glob
from pathlib import Path

# Paths based on structure
PROJECT_DIR = Path(".")
REF_DIR = PROJECT_DIR / "data" / "reference"
YEARLY_DIR = PROJECT_DIR / "outputs" / "yearly"

REF_DIR.mkdir(parents=True, exist_ok=True)

print("Reference directory:", REF_DIR.resolve())
print("Yearly parquet directory:", YEARLY_DIR.resolve())

# GBFS API pull, since there is no historical data
GBFS_ROOT_URL = "https://gbfs.mex.lyftbikes.com/gbfs/gbfs.json"

response = requests.get(GBFS_ROOT_URL, timeout=30)
response.raise_for_status()

gbfs_root = response.json()

# Get API info
station_response = requests.get(station_info_url, timeout=30)
station_response.raise_for_status()

station_json = station_response.json()

print("Station info top-level keys:", station_json.keys())
print("Station data keys:", station_json["data"].keys())

# Data frame and inspect
stations_raw = pd.DataFrame(station_json["data"]["stations"])

print("Shape:", stations_raw.shape)
stations_raw.head()
print(stations_raw.columns.tolist())

# Define station characteristics 
station_characteristics = [
    c for c in [
        "station_id",
        "name",
        "short_name",
        "lat",
        "lon",
        "capacity",
        "region_id",
        "address"
    ]
    if c in stations_raw.columns
]

stations_ref = stations_raw[keep_cols].copy()

# Standardize station_id for potential merges
stations_ref["station_id"] = stations_ref["station_id"].astype(str).str.strip()

# Remove duplicate station_ids if any
stations_ref = stations_ref.drop_duplicates(subset=["station_id"])

print("Clean station reference shape:", stations_ref.shape)
stations_ref.head()

# Basic quality check
print("Number of stations:", len(stations_ref))
print("Missing station_id:", stations_ref["station_id"].isna().sum())

if "lat" in stations_ref.columns:
    print("Missing lat:", stations_ref["lat"].isna().sum())

if "lon" in stations_ref.columns:
    print("Missing lon:", stations_ref["lon"].isna().sum())

stations_ref.sample(min(5, len(stations_ref)))

# Save GBFS station reference
csv_path = REF_DIR / "stations_gbfs_current.csv"
parquet_path = REF_DIR / "stations_gbfs_current.parquet"

stations_ref.to_csv(csv_path, index=False)
stations_ref.to_parquet(parquet_path, index=False)

print("Saved:")
print(csv_path.resolve())
print(parquet_path.resolve())