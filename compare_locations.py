"""
Compare the hot-vs-normal tmax bias across all locations in locations.csv, to
check whether the heatwave underforecast effect is a general model property
or specific to one city (e.g. Paris's urban heat island).

Requires each location to already have data/{name}/leadtime.parquet (from
fetch_previous_runs.py) - this script runs analyze.py's logic on each and
combines the results.

Usage:
    python compare_locations.py --out-dir data
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analyze import add_climatology, summarize, style_axes, COLOR_NORMAL, COLOR_HOT, COLOR_TEXT

BAR_WIDTH = 0.35


def load_location(name: str, climatology_window: int, anomaly_threshold: float) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{name}/leadtime.parquet")
    df["date"] = pd.to_datetime(df["date"])

    daily_actual = df[["date", "actual_tmax"]].drop_duplicates("date")
    daily_actual = add_climatology(daily_actual, climatology_window, anomaly_threshold)
    df = df.merge(daily_actual[["date", "is_hot_anomaly"]], on="date")

    summary = summarize(df, "tmax")
    summary.insert(0, "location", name)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--locations-file", default="locations.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--lead-days", type=int, default=7,
                    help="which lead time to compare across locations")
    p.add_argument("--climatology-window", type=int, default=7)
    p.add_argument("--anomaly-threshold", type=float, default=5.0)
    args = p.parse_args()

    locations = pd.read_csv(args.locations_file)

    all_summaries = []
    for _, row in locations.iterrows():
        name = row["name"]
        try:
            summary = load_location(name, args.climatology_window, args.anomaly_threshold)
        except FileNotFoundError:
            print(f"Skipping {name}: no data/{name}/leadtime.parquet")
            continue
        summary["region"] = row["region"]
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    out_csv = f"{args.out_dir}/locations_comparison.csv"
    combined.to_csv(out_csv, index=False)
    print(combined.to_string(index=False))
    print(f"\nSaved to {out_csv}")

    at_lead = combined[combined["lead_days"] == args.lead_days].sort_values("bias_hot", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    style_axes(ax)
    x = range(len(at_lead))
    ax.bar([i - BAR_WIDTH / 2 for i in x], at_lead["bias_normal"], BAR_WIDTH,
           color=COLOR_NORMAL, label="Normal days")
    ax.bar([i + BAR_WIDTH / 2 for i in x], at_lead["bias_hot"], BAR_WIDTH,
           color=COLOR_HOT, label="Anomalous heat days")
    ax.set_xticks(list(x))
    ax.set_xticklabels(at_lead["location"], rotation=30, ha="right")
    ax.set_ylabel("Mean error, actual − forecast (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title(f"Tmax bias by location, {args.lead_days}-day lead", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_png = f"{args.out_dir}/locations_comparison_lead{args.lead_days}.png"
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"Saved chart to {out_png}")


if __name__ == "__main__":
    main()
