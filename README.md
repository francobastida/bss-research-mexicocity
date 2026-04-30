# 🚲 Bicycle-Sharing System Research Repository

---

## Overview

This repository contains all code, processed data, and outputs associated with the thesis:

> **Where Bike Thou? A Machine Learning Approach to Demand Prediction and Bike-sharing Station Prioritization in Mexico City**

The project has been submitted as part of the requirements for the Master of Data Science for Public Policy degree at the Hertie School. The thesis asks whether historical bike-sharing flows can be used to **predict station-level demand** and **inform prioritization strategies for bike rebalancing** in a large-scale urban system.

The modeling framework is structured around two main components:

1. **Demand prediction models**  
> Machine learning models (Linear, Poisson, Decision Tree, Random Forest, XGBoost) are used to forecast hourly station-level departures.

2. **Station prioritization framework**  
> A prediction-informed prioritization algorithm ranks stations based on **expected demand pressure and imbalance**, identifying high-impact targets for rebalancing.

The approach is evaluated using time-based splits and **out-of-sample data (2025)**, comparing prediction-informed prioritization against observed imbalance patterns.

---

## Data Source

Mexico City’s public bike-sharing system is **ECOBICI**  
(*Sistema de Transporte Individual en Bicicleta Pública ECOBICI*).

Historical data is publicly available through:

> ECOBICI Open Data Portal  
> https://ecobici.cdmx.gob.mx/en/open-data/

---

### Data Notes

- Raw CSV files are **not included** due to:
  - formatting inconsistencies across years and city administrations  
  - encoding differences and schema variation  
  - file size constraints  

- Instead, this repository provides:
  - cleaned and processed datasets (`.parquet`)
  - feature-engineered inputs and modeling datasets

---

## Repository Structure

```text
ECOBICI-2026/
│
├── data/
│ ├── raw/ (ignored)
│ └── reference/ (processed)
│ ├── stations_gbfs_current.parquet
│ ├── stations_gbfs_with_metro_distance.parquet
│ ├── weather_hourly_cdmx_2018_2025.parquet
│ └── mx_holidays.parquet
│
├── outputs/
│ ├── figures/ 
│ ├── ecobici_station_hour.parquet
│ ├── ecobici_station_hour_features.parquet
│ └── opt_df.parquet (for prioritization)

├── src/ (archive)
│ ├── cleandata_ecobici.py
│ ├── station_hour_paneldataset.py
│ ├── preprocessing_eda.py
│ └── gbfs_cdmxstations.py
│
├── notebooks/
│ ├── 01_data_reference.ipynb
│ ├── 02_station_hour_panel_eda.ipynb
│ ├── 03_cdmx_multimodality_feature.ipynb
│ ├── 04_cdmx_weather_feature.ipynb
│ ├── 05_cdmx_holiday_feature.ipynb
│ ├── 06_station_hour_ALLfeatures.ipynb
│ ├── 07_prediction_model.ipynb
│ └── 08_optimization_model.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Workflow Summary

The analysis follows a structured pipeline:

1. **Data Cleaning (`src/`)**
   - Harmonizes inconsistencies in language and format
   - Parses timestamps and validates trip durations

2. **Station-Hour Panel Construction**
   - Aggregates trips into a station-hour panel
   - Computes departures, arrivals, net flow, and imbalance metrics

3. **Feature Engineering (`notebooks/03–06`)**
   - Temporal features (hour, weekday, peak periods)
   - Weather, holiday and multimodality proxies (distance to nearest metro)

4. **Demand Prediction (`notebooks/07`)**
   - Models evaluated: Linear, Poisson, Decision Tree, Random Forest, XGBoost
   - Error metrics: MAE and RMSE
   - Validation using 2025 data

5. **Station Prioritization (`notebooks/08`)**
   - Priority score algorithm based on predicted demand and imbalance
   - Identification of top 5% high-priority stations
   - Evaluation against observed imbalance in 2025

---

## Disclaimer

The preprocessing scripts are provided for **transparency**, but may not fully reproduce identical outputs without access to the original raw CSV files due to formatting inconsistencies across data sources.
Large datasets (e.g., yearly parquet files) are excluded due to GitHub size constraints and for efficiency purposes.

---

## AI Usage

This project used ChatGPT and Claude for code assistance, debugging, and formatting. DeepL was used for language translation. All outputs were reviewed, adapted, and validated by the author.

---

## Contact

For questions, feedback, or collaboration inquiries, please contact:  
**Franco Bastida**  
F.Bastida@students.hertie-school.org
