"""
Collect forecast-vs-actual temperature data for a location using Open-Meteo's
Previous Runs API, at fixed lead times of 1/3/5/7 days.

Output: one row per (date, lead_days) with the actual daily tmax/tmin, the
value that had been forecast `lead_days` days earlier, and the resulting
error (actual - forecast).

Usage:
    python fetch_previous_runs.py --lat 48.8566 --lon 2.3522 \
        --start 2022-01-01 --end 2026-07-01 --out data/paris.parquet
"""

import argparse
import sys

import pandas as pd
import requests

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# GFS has the longest continuous archive (back to March 2021) and stays the
# same model across the whole period, which matters for a bias study: mixing
# models mid-series would confound "model bias" with "model switch".
DEFAULT_MODEL = "gfs_seamless"
DEFAULT_LEAD_DAYS = [1, 3, 5, 7]


def fetch_hourly(lat: float, lon: float, start: str, end: str, model: str,
                  lead_days: list[int], timezone: str) -> pd.DataFrame:
    variables = ["temperature_2m"] + [f"temperature_2m_previous_day{d}" for d in lead_days]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "models": model,
        "start_date": start,
        "end_date": end,
        "timezone": timezone,
    }
    try:
        r = requests.get(API_URL, params=params, timeout=120)
    except requests.exceptions.SSLError as e:
        print(
            "SSL certificate verification failed. This is common behind corporate "
            "proxies/firewalls that intercept TLS. Fix options:\n"
            "  1. Set REQUESTS_CA_BUNDLE to a CA bundle that includes your proxy's "
            "root certificate.\n"
            "  2. Re-run with --insecure to skip verification (not recommended, "
            "only for trusted networks).",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if r.status_code != 200:
        print(f"API error {r.status_code}: {r.text}", file=sys.stderr)
        raise SystemExit(1)

    data = r.json()
    if "hourly" not in data:
        print(f"Unexpected response: {data}", file=sys.stderr)
        raise SystemExit(1)

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def aggregate_daily(hourly: pd.DataFrame, lead_days: list[int]) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["date"] = hourly["time"].dt.date

    agg = {"temperature_2m": ["max", "min"]}
    for d in lead_days:
        agg[f"temperature_2m_previous_day{d}"] = ["max", "min"]

    daily = hourly.groupby("date").agg(agg)
    daily.columns = ["_".join(c) for c in daily.columns]
    daily = daily.rename(columns={
        "temperature_2m_max": "actual_tmax",
        "temperature_2m_min": "actual_tmin",
    })

    rows = []
    for d in lead_days:
        sub = daily[[f"temperature_2m_previous_day{d}_max",
                      f"temperature_2m_previous_day{d}_min",
                      "actual_tmax", "actual_tmin"]].copy()
        sub.columns = ["forecast_tmax", "forecast_tmin", "actual_tmax", "actual_tmin"]
        sub["lead_days"] = d
        sub = sub.reset_index()
        rows.append(sub)

    out = pd.concat(rows, ignore_index=True)
    out["error_tmax"] = out["actual_tmax"] - out["forecast_tmax"]
    out["error_tmin"] = out["actual_tmin"] - out["forecast_tmin"]
    out = out.sort_values(["date", "lead_days"]).reset_index(drop=True)
    return out[["date", "lead_days", "actual_tmax", "actual_tmin",
                "forecast_tmax", "forecast_tmin", "error_tmax", "error_tmin"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--lead-days", type=int, nargs="+", default=DEFAULT_LEAD_DAYS)
    p.add_argument("--timezone", default="Europe/Paris")
    p.add_argument("--out", required=True, help="Output path (.parquet or .csv)")
    p.add_argument("--insecure", action="store_true",
                    help="Skip TLS certificate verification (corporate proxy workaround)")
    args = p.parse_args()

    if args.insecure:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        global API_URL
        session_get = requests.get

        def patched_get(*a, **kw):
            kw["verify"] = False
            return session_get(*a, **kw)

        requests.get = patched_get

    hourly = fetch_hourly(args.lat, args.lon, args.start, args.end,
                           args.model, args.lead_days, args.timezone)
    daily = aggregate_daily(hourly, args.lead_days)
    daily.insert(0, "longitude", args.lon)
    daily.insert(0, "latitude", args.lat)
    daily.insert(0, "model", args.model)

    n_missing = daily["forecast_tmax"].isna().sum()
    print(f"{len(daily)} rows ({daily['date'].min()} .. {daily['date'].max()}), "
          f"{n_missing} rows with missing forecast (edge of archive)")

    if args.out.endswith(".csv"):
        daily.to_csv(args.out, index=False)
    else:
        daily.to_parquet(args.out, index=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
