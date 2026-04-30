from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# pip install holidays is an optional dependency 

# =========================================================
# Configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[1] #makes src the BASE DIR
INPUT_FILE = BASE_DIR / "outputs" / "yearly" / "ecobici_trips_2018.parquet"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trip hour and duration filters for panel

time_grain = "h"  # use "30min" if you want to compare later
year = 2018

min_trip_duration = 1
max_trip_duration = 180

# =========================================================
# Helpers
# =========================================================

# What holidays do we have that can be modeled?
def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an official Mexico holiday flag using the `holidays` package.
    Install first with: pip install holidays
    """
    try:
        import holidays
    except ImportError as exc:
        raise ImportError(
            "The `holidays` package is required for add_holiday_flag(). "
            "Install it with `pip install holidays`."
        ) from exc

    mx_holidays = holidays.country_holidays("MX", years=[year])

    df = df.copy()
    dt = pd.to_datetime(df["datetime"])
    df["is_holiday"] = dt.dt.date.astype("datetime64[ns]").isin(pd.to_datetime(list(mx_holidays.keys()))) #checks if there's a holiday
    return df #true or false if the station-time row took place on a holiday

# How concentrated is demand (e.g. top X stations concentrate this percantage of demand)?
def station_activity_rank(series: pd.Series) -> pd.DataFrame:
    totals = series.sort_values(ascending=False).reset_index(drop=True) #list of stations with most to next most trips (transactions)
    shares = totals.cumsum() / totals.sum()
    out = pd.DataFrame({
        "top_n_stations": np.arange(1, len(totals) + 1),
        "share of total": shares.values,
    })
    return out


# =========================================================
# Load and filter trip master table
# =========================================================
print(f"Loading {INPUT_FILE} ...")
df = pd.read_parquet(INPUT_FILE)
print(f"Raw rows: {len(df):,}")

# Remove NAs and keep only rows valid enough for EDA of flows
mask = (
    df["start_station_id"].notna() & #origin
    df["end_station_id"].notna() & #destination
    df["start_dt"].notna() & #validstart
    df["end_dt"].notna() & #validend
    ~df["flag_negative_duration"].fillna(False) & #no negative duration
    ~df["flag_duration_gt_24h"].fillna(False) & #no longer than 24
    df["trip_duration_min"].between(min_trip_duration, max_trip_duration, inclusive="both") #between 1-180 mins
)

trips = df.loc[mask].copy() #clean subset for trips

# =========================================================
# Basic trip-level organization
# =========================================================
trips["start_interval"] = trips["start_dt"].dt.floor(time_grain) #round trip timestamps
trips["end_interval"] = trips["end_dt"].dt.floor(time_grain) #round trip timestamps

trips["start_weekday"] = trips["start_dt"].dt.dayofweek
trips["start_is_weekend"] = trips["start_weekday"].isin([5, 6])
trips["start_hour"] = trips["start_dt"].dt.hour
trips["start_date_only"] = trips["start_dt"].dt.date

# =========================================================
# Average trip duration and peak-period
# =========================================================

# What is the trip duration by day and hour?
avg_trip_duration = (
    trips.groupby(["start_date_only", "start_hour"], as_index=False)["trip_duration_min"]
    .mean()
    .rename(columns={"trip_duration_min": "avg_trip_duration_min"})
)

# What is the number of total trips in the hourly interval?
hourly_system_demand = (
    trips.groupby("start_interval", as_index=False)
    .size()
    .rename(columns={"size": "system_departures", "start_interval": "datetime"})
)

# Average demand profile by weekday and hour
hourly_profile = (
    hourly_system_demand.assign(
        hour=lambda x: pd.to_datetime(x["datetime"]).dt.hour,
        weekday=lambda x: pd.to_datetime(x["datetime"]).dt.dayofweek,
    )
    .groupby(["weekday", "hour"], as_index=False)["system_departures"]
    .mean()
    .rename(columns={"system_departures": "avg_departures"})
)

peak_threshold = hourly_profile["avg_departures"].quantile(0.90) #highest demand periods are those in the top 10%
peak_periods = hourly_profile.loc[hourly_profile["avg_departures"] >= peak_threshold].copy()
peak_periods["is_peak_period"] = True

# =========================================================
# Departures / arrivals station-time panel
# =========================================================

# How many trips start at each station in each hour?
departures = (
    trips.groupby(["start_station_id", "start_interval"], as_index=False)
    .size()
    .rename(columns={
        "start_station_id": "station_id",
        "start_interval": "datetime",
        "size": "departures",
    })
)

#How many trips end at each station in each hour?
arrivals = (
    trips.groupby(["end_station_id", "end_interval"], as_index=False)
    .size()
    .rename(columns={
        "end_station_id": "station_id",
        "end_interval": "datetime",
        "size": "arrivals",
    })
)

# =========================================================
# Merging into one panel
# =========================================================

panel = departures.merge(arrivals, on=["station_id", "datetime"], how="outer") # some stations may have departures but no arrivals and vice versa
panel[["departures", "arrivals"]] = panel[["departures", "arrivals"]].fillna(0) # missings are filled with 0
panel["station_id"] = panel["station_id"].astype("Int64")
panel = panel.sort_values(["station_id", "datetime"]).reset_index(drop=True)

# What is the station imbalance? 
# A positive flow would be more departures than arrivals (deficiency)
# A negative flow would mean more arrivals than departures (surplus)

panel["net_flow"] = panel["departures"] - panel["arrivals"]
panel["abs_imbalance"] = panel["net_flow"].abs() # to identify the most unstable stations
panel["transactions"] = panel["departures"] + panel["arrivals"] # total activity

# What is the priority score of station_i? Based on Maya-Duque et al. 
system_transactions = panel.groupby("datetime")["transactions"].transform("sum")
panel["priority_score"] = np.where(
    system_transactions.gt(0),
    (panel["transactions"] * panel["net_flow"]) / system_transactions,
    0,
)

# Rank and simple class labels based on imbalance (balanced, deficiency, surplus)
panel["imbalance_rank_desc"] = panel.groupby("datetime")["abs_imbalance"].rank(method="dense", ascending=False)

# A simple shortage/surplus class based on net flow sign and interval-specific magnitude
imb_q75 = panel.groupby("datetime")["abs_imbalance"].transform(lambda s: s.quantile(0.75) if len(s) > 0 else 0)
panel["imbalance_class"] = "balanced"
panel.loc[(panel["net_flow"] > 0) & (panel["abs_imbalance"] >= imb_q75), "imbalance_class"] = "deficiency"
panel.loc[(panel["net_flow"] < 0) & (panel["abs_imbalance"] >= imb_q75), "imbalance_class"] = "surplus"

# =========================================================
# Adding time/calendar features for EDA
# =========================================================
panel["year"] = pd.to_datetime(panel["datetime"]).dt.year
panel["month"] = pd.to_datetime(panel["datetime"]).dt.month
panel["day"] = pd.to_datetime(panel["datetime"]).dt.day
panel["hour"] = pd.to_datetime(panel["datetime"]).dt.hour
panel["weekday"] = pd.to_datetime(panel["datetime"]).dt.dayofweek
panel["is_weekend"] = panel["weekday"].isin([5, 6])
panel = add_holiday_flag(panel)

# =========================================================
# Lag / moving-average features for later modeling
# =========================================================
panel = panel.sort_values(["station_id", "datetime"]).copy()

for col in ["departures", "arrivals", "net_flow", "abs_imbalance"]:
    panel[f"{col}_lag_1"] = panel.groupby("station_id")[col].shift(1)
    panel[f"{col}_lag_24"] = panel.groupby("station_id")[col].shift(24)
    panel[f"{col}_rolling_mean_3"] = (
        panel.groupby("station_id")[col]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

# =========================================================
# Main summary outputs for EDA
# =========================================================
station_demand = panel.groupby("station_id")["transactions"].sum().sort_values(ascending=False) # total activity per station
station_imbalance = panel.groupby("station_id")["abs_imbalance"].sum().sort_values(ascending=False) # total imbalance per station

# Is demand/imbalance concentrated in a few stations? Imporant to subsample later.
station_demand_concentration = station_concentration_summary(station_demand)
station_imbalance_concentration = station_concentration_summary(station_imbalance)

# Station to station matrix in long format to analyze flows
od_station = (
    trips.groupby(["start_station_id", "end_station_id"], as_index=False)
    .size()
    .rename(columns={
        "start_station_id": "origin_station_id",
        "end_station_id": "destination_station_id",
        "size": "trip_count",
    })
    .sort_values("trip_count", ascending=False)
)

# What are the flows that start and end at the same station?
same_station_share = float(trips["flag_same_station"].fillna(False).mean())

eda_summary = {
    "raw_rows": int(len(df)),
    "rows_after_eda_filters": int(len(trips)),
    "n_unique_start_stations": int(trips["start_station_id"].nunique()),
    "n_unique_end_stations": int(trips["end_station_id"].nunique()),
    "n_unique_panel_stations": int(panel["station_id"].nunique()),
    "mean_trip_duration_min": float(trips["trip_duration_min"].mean()),
    "median_trip_duration_min": float(trips["trip_duration_min"].median()),
    "p90_trip_duration_min": float(trips["trip_duration_min"].quantile(0.90)),
    "same_station_trip_share": same_station_share,
    "time_grain": time_grain,
}
summary_df = pd.DataFrame([eda_summary])

# =========================================================
# Save outputs
# =========================================================
panel_out = OUTPUT_DIR / f"ecobici_station_panel_{YEAR}_{TIME_GRAIN.replace('min', 'm')}.parquet"
peak_out = OUTPUT_DIR / f"ecobici_peak_periods_{YEAR}_{TIME_GRAIN.replace('min', 'm')}.csv"
hourly_profile_out = OUTPUT_DIR / f"ecobici_hourly_profile_{YEAR}_{TIME_GRAIN.replace('min', 'm')}.csv"
avg_duration_out = OUTPUT_DIR / f"ecobici_avg_trip_duration_{YEAR}_{TIME_GRAIN.replace('min', 'm')}.csv"
od_out = OUTPUT_DIR / f"ecobici_od_station_{YEAR}.parquet"
demand_conc_out = OUTPUT_DIR / f"ecobici_station_demand_concentration_{YEAR}.csv"
imbalance_conc_out = OUTPUT_DIR / f"ecobici_station_imbalance_concentration_{YEAR}.csv"
summary_out = OUTPUT_DIR / f"ecobici_summary_outputs_{YEAR}_{TIME_GRAIN.replace('min', 'm')}.csv"

panel.to_parquet(panel_out, index=False)
peak_periods.to_csv(peak_out, index=False)
hourly_profile.to_csv(hourly_profile_out, index=False)
avg_trip_duration.to_csv(avg_duration_out, index=False)
od_station.to_parquet(od_out, index=False)
station_demand_concentration.to_csv(demand_conc_out, index=False)
station_imbalance_concentration.to_csv(imbalance_conc_out, index=False)
summary_df.to_csv(summary_out, index=False)

print("Saved outputs:")
print(f" - {panel_out}")
print(f" - {peak_out}")
print(f" - {hourly_profile_out}")
print(f" - {avg_duration_out}")
print(f" - {od_out}")
print(f" - {demand_conc_out}")
print(f" - {imbalance_conc_out}")
print(f" - {summary_out}")
