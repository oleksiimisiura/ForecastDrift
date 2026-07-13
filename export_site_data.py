"""
Export collected data to compact JSON for the static GitHub Pages dashboard
in docs/. Client-side JS filters by date range and recomputes MAE/RMSE/bias -
the anomaly flag (is_hot_anomaly) is precomputed here since it depends on a
climatology built from the full series.

Usage:
    python export_site_data.py --locations-file locations.csv --out-dir docs/data
"""

import argparse
import json

import pandas as pd

from analyze import add_climatology

COLUMNS = ["date", "lead_days", "actual_tmax", "actual_tmin", "forecast_tmax",
           "forecast_tmin", "error_tmax", "error_tmin", "is_hot_anomaly"]


def round_or_none(x):
    return None if pd.isna(x) else round(float(x), 2)


def export_location(name: str, climatology_window: int, anomaly_threshold: float) -> dict:
    df = pd.read_parquet(f"data/{name}/leadtime.parquet")
    df["date"] = pd.to_datetime(df["date"])

    daily_actual = df[["date", "actual_tmax"]].drop_duplicates("date")
    daily_actual = add_climatology(daily_actual, climatology_window, anomaly_threshold)
    df = df.merge(daily_actual[["date", "is_hot_anomaly"]], on="date")
    df = df.sort_values(["date", "lead_days"])

    rows = []
    for r in df.itertuples(index=False):
        rows.append([
            r.date.strftime("%Y-%m-%d"),
            int(r.lead_days),
            round_or_none(r.actual_tmax), round_or_none(r.actual_tmin),
            round_or_none(r.forecast_tmax), round_or_none(r.forecast_tmin),
            round_or_none(r.error_tmax), round_or_none(r.error_tmin),
            bool(r.is_hot_anomaly),
        ])
    return {"columns": COLUMNS, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locations-file", default="locations.csv")
    p.add_argument("--out-dir", default="docs/data")
    p.add_argument("--climatology-window", type=int, default=7)
    p.add_argument("--anomaly-threshold", type=float, default=5.0)
    args = p.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    locations = pd.read_csv(args.locations_file)
    meta = []
    for _, row in locations.iterrows():
        name = row["name"]
        try:
            payload = export_location(name, args.climatology_window, args.anomaly_threshold)
        except FileNotFoundError:
            print(f"Skipping {name}: no data/{name}/leadtime.parquet")
            continue
        out_path = f"{args.out_dir}/{name}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        print(f"Wrote {out_path} ({len(payload['rows'])} rows)")
        meta.append({"name": name, "lat": row["lat"], "lon": row["lon"], "region": row["region"]})

    with open(f"{args.out_dir}/locations.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.out_dir}/locations.json ({len(meta)} locations)")


if __name__ == "__main__":
    main()
