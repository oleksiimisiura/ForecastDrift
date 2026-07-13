"""
Daily accumulator for the 10-day lead time, which Open-Meteo does not archive
historically (Previous Runs API caps at 7 days; Single Runs API only keeps a
rolling ~90-day window of past model runs). There is no way to backfill this
lead time for past years - the only option is to record it going forward,
one day at a time.

Intended to run once per day (e.g. via a scheduled task). Each run:
  1. Takes the ECMWF IFS run from exactly 10 days before the target date
     (default: today) and reads what it forecast for the target date.
  2. Fetches the actual tmax/tmin for the target date.
  3. Appends one row to the running archive CSV, skipping dates already
     present so the script is safe to re-run.

Usage:
    python fetch_single_run_daily.py --lat 48.8566 --lon 2.3522 \
        --out data/paris/10day_archive.csv
"""

import argparse
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests

SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LEAD_DAYS = 10
DEFAULT_MODEL = "ecmwf_ifs025"


def fetch_forecast(lat: float, lon: float, target: date, model: str, timezone: str):
    run_date = target - timedelta(days=LEAD_DAYS)
    run = f"{run_date.isoformat()}T00:00"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "models": model,
        "run": run,
        "forecast_days": LEAD_DAYS + 2,
        "timezone": timezone,
    }
    r = requests.get(SINGLE_RUN_URL, params=params, timeout=60)
    if r.status_code != 200:
        return None, r.text

    data = r.json()
    if "hourly" not in data:
        return None, data

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    day_rows = df[df["time"].dt.date == target]
    if day_rows.empty:
        return None, f"run did not cover target date {target}"

    return {
        "forecast_tmax": day_rows["temperature_2m"].max(),
        "forecast_tmin": day_rows["temperature_2m"].min(),
    }, None


def fetch_actual(lat: float, lon: float, target: date, timezone: str):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min",
        "start_date": target.isoformat(),
        "end_date": target.isoformat(),
        "timezone": timezone,
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=60)
    if r.status_code != 200:
        return None, r.text

    data = r.json()
    if "daily" not in data or not data["daily"]["time"]:
        return None, data

    return {
        "actual_tmax": data["daily"]["temperature_2m_max"][0],
        "actual_tmin": data["daily"]["temperature_2m_min"][0],
    }, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--date", help="Target date YYYY-MM-DD (default: yesterday, "
                                   "so the actual tmax/tmin reflects a full day)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timezone", default="Europe/Paris")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    target = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
              else date.today() - timedelta(days=1))

    try:
        existing = pd.read_csv(args.out, parse_dates=["date"])
        existing_dates = set(existing["date"].dt.date)
    except FileNotFoundError:
        existing = None
        existing_dates = set()

    if target in existing_dates:
        print(f"{target} already in archive, skipping.")
        return

    forecast, ferr = fetch_forecast(args.lat, args.lon, target, args.model, args.timezone)
    if forecast is None:
        print(f"No forecast available for {target} "
              f"(run {target - timedelta(days=LEAD_DAYS)} may be outside the "
              f"rolling archive window): {ferr}", file=sys.stderr)
        raise SystemExit(1)

    actual, aerr = fetch_actual(args.lat, args.lon, target, args.timezone)
    if actual is None:
        print(f"No actual data yet for {target}: {aerr}", file=sys.stderr)
        raise SystemExit(1)

    row = {
        "date": target.isoformat(),
        "lead_days": LEAD_DAYS,
        "model": args.model,
        "latitude": args.lat,
        "longitude": args.lon,
        **actual,
        **forecast,
        "error_tmax": actual["actual_tmax"] - forecast["forecast_tmax"],
        "error_tmin": actual["actual_tmin"] - forecast["forecast_tmin"],
    }
    row_df = pd.DataFrame([row])
    row_df["date"] = pd.to_datetime(row_df["date"])

    combined = pd.concat([existing, row_df], ignore_index=True) if existing is not None else row_df
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(args.out, index=False)
    print(f"Added {target}: forecast_tmax={forecast['forecast_tmax']}, "
          f"actual_tmax={actual['actual_tmax']}, error={row['error_tmax']:.1f}. "
          f"Archive now has {len(combined)} rows.")


if __name__ == "__main__":
    main()
