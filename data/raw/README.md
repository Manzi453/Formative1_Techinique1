# Raw Data

This folder holds the raw, untouched mobile network traffic dataset. Files
placed here are **excluded from version control** (see root `.gitignore`) to
keep the repository small and to respect the original dataset's license.

## Expected file

- `mobile_traffic.csv` (path configurable in [`config.yaml`](../../config.yaml)
  under `data.raw_path`)

## Expected schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Timestamp of the observation, regularly spaced (e.g. hourly). |
| `traffic_volume` | float | Target variable — aggregate network traffic (e.g. total data volume in GB, or call+SMS+internet activity for a cell). |
| *(optional)* additional columns | float/int | Any auxiliary signals available (e.g. per-service breakdown, weather, events) can be used as extra regressors for SARIMAX/XGBoost. |

## Data source

> **TODO (author):** Replace this section with the exact source, access date,
> and license of the dataset actually used, for example:
> - Telecom Italia "Big Data Challenge" — Milano mobile traffic dataset
>   (call/SMS/internet activity per grid cell, publicly released for research).
> - A Kaggle/UCI mobile or cellular network traffic dataset.
> - Proprietary/anonymized operator data (state any usage restrictions here).

Include:
1. Direct link or citation for the dataset.
2. Date the data was retrieved.
3. License / usage terms.
4. Any known caveats (missing periods, sensor changes, anonymization).

## How to obtain it

1. Download the dataset from the source above.
2. Place the file(s) in this directory (`data/raw/`).
3. Update `data.raw_path` in [`config.yaml`](../../config.yaml) if the
   filename differs.
