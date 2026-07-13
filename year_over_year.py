"""
Group forecast error by calendar week + lead time and compare across years -
"was the same week in August 2022 forecast worse than the same week in 2024?"

Two outputs:
  1. A year x ISO-week heatmap of mean tmax bias at a fixed lead time, to
     spot which weeks/years had the worst forecast bias at a glance.
  2. A year-over-year comparison for the single climatologically hottest week
     of the year, showing bias by lead time with one line per year - the
     direct "same week, different years" comparison.

Usage:
    python year_over_year.py --in data/paris/leadtime.parquet --out-dir data/paris
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyze import style_axes, COLOR_TEXT, COLOR_MUTED

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"]
)

YEAR_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#eb6834"]


def build_week_year_table(df: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    g = df[df["lead_days"] == lead_days].copy()
    iso = pd.to_datetime(g["date"]).dt.isocalendar()
    g["iso_year"] = iso["year"]
    g["iso_week"] = iso["week"]
    pivot = g.pivot_table(index="iso_year", columns="iso_week", values="error_tmax", aggfunc="mean")
    return pivot


def plot_heatmap(pivot: pd.DataFrame, lead_days: int, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=150)
    vmax = np.nanmax(np.abs(pivot.values))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                        cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, shading="nearest")
    ax.set_yticks(pivot.index)
    ax.set_xlabel("ISO week of year", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Year", color=COLOR_TEXT, fontsize=10)
    ax.set_title(f"Tmax bias by week and year, {lead_days}-day lead "
                 f"(red = actual hotter than forecast)", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Mean error, actual − forecast (°C)", color=COLOR_TEXT, fontsize=9)
    cbar.ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def hottest_week(df: pd.DataFrame) -> int:
    actual = df[["date", "actual_tmax"]].drop_duplicates("date").copy()
    actual["iso_week"] = pd.to_datetime(actual["date"]).dt.isocalendar().week
    by_week = actual.groupby("iso_week")["actual_tmax"].mean()
    return int(by_week.idxmax())


def plot_hottest_week_by_year(df: pd.DataFrame, week: int, out_path: str):
    g = df.copy()
    iso = pd.to_datetime(g["date"]).dt.isocalendar()
    g["iso_year"] = iso["year"]
    g["iso_week"] = iso["week"]
    g = g[g["iso_week"] == week]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)
    years = sorted(g["iso_year"].unique())
    lead_days = sorted(g["lead_days"].unique())
    for i, year in enumerate(years):
        yg = g[g["iso_year"] == year].groupby("lead_days")["error_tmax"].mean().reindex(lead_days)
        color = YEAR_COLORS[i % len(YEAR_COLORS)]
        ax.plot(lead_days, yg.values, color=color, linewidth=2, marker="o", markersize=5, label=str(year))

    ax.set_xlabel("Lead time (days)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Mean error, actual − forecast (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title(f"Tmax bias by lead time, ISO week {week}, by year", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(lead_days)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/paris/leadtime.parquet")
    p.add_argument("--out-dir", default="data/paris")
    p.add_argument("--lead-days", type=int, default=7, help="lead time shown in the heatmap")
    args = p.parse_args()

    df = pd.read_parquet(args.inp) if args.inp.endswith(".parquet") else pd.read_csv(args.inp, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])

    pivot = build_week_year_table(df, args.lead_days)
    heatmap_csv = f"{args.out_dir}/year_over_year_week_heatmap.csv"
    pivot.to_csv(heatmap_csv)
    heatmap_png = f"{args.out_dir}/year_over_year_week_heatmap.png"
    plot_heatmap(pivot, args.lead_days, heatmap_png)
    print(f"Saved heatmap to {heatmap_png} and {heatmap_csv}")

    week = hottest_week(df)
    print(f"Climatologically hottest ISO week across the dataset: week {week}")
    hottest_png = f"{args.out_dir}/year_over_year_hottest_week.png"
    plot_hottest_week_by_year(df, week, hottest_png)
    print(f"Saved hottest-week year comparison to {hottest_png}")


if __name__ == "__main__":
    main()
