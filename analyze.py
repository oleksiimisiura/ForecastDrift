"""
Compute forecast error metrics (MAE, RMSE, systematic bias) by lead time, and
test whether bias differs between anomalously hot periods and normal periods.

"Anomalously hot" = actual tmax exceeds a smoothed climatological normal for
that calendar day by at least --anomaly-threshold degrees. The climatology is
built from this dataset itself (there's no independent long-term normal here),
so with ~4-5 years of data it is noisy - treat it as indicative, not authoritative.

Usage:
    python analyze.py --in data/paris_leadtime.parquet --out-dir data
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLOR_NORMAL = "#2a78d6"
COLOR_HOT = "#e34948"
COLOR_OVERALL = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT = "#0b0b0b"
COLOR_MUTED = "#898781"


def add_climatology(daily_actual: pd.DataFrame, window_days: int, threshold: float) -> pd.DataFrame:
    df = daily_actual.sort_values("date").reset_index(drop=True)
    doy = pd.to_datetime(df["date"]).dt.dayofyear.to_numpy()
    tmax = df["actual_tmax"].to_numpy()
    n = len(df)

    clim = np.empty(n)
    for i in range(n):
        circ_dist = np.abs(((doy - doy[i] + 183) % 366) - 183)
        clim[i] = tmax[circ_dist <= window_days].mean()

    df["climatology_tmax"] = clim
    df["anomaly_tmax"] = df["actual_tmax"] - df["climatology_tmax"]
    df["is_hot_anomaly"] = df["anomaly_tmax"] >= threshold
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lead, g in df.groupby("lead_days"):
        g = g.dropna(subset=["error_tmax"])
        hot = g[g["is_hot_anomaly"]]
        normal = g[~g["is_hot_anomaly"]]

        row = {
            "lead_days": lead,
            "n": len(g),
            "mae_tmax": g["error_tmax"].abs().mean(),
            "rmse_tmax": (g["error_tmax"] ** 2).mean() ** 0.5,
            "bias_tmax": g["error_tmax"].mean(),
            "n_hot": len(hot),
            "bias_tmax_hot": hot["error_tmax"].mean() if len(hot) else float("nan"),
            "n_normal": len(normal),
            "bias_tmax_normal": normal["error_tmax"].mean() if len(normal) else float("nan"),
        }
        if len(hot) >= 2 and len(normal) >= 2:
            t, p = stats.ttest_ind(hot["error_tmax"], normal["error_tmax"], equal_var=False)
            row["welch_t"] = t
            row["p_value"] = p
        else:
            row["welch_t"] = float("nan")
            row["p_value"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("lead_days")


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    ax.axhline(0, color=COLOR_AXIS, linewidth=1)


def plot_bias_by_leadtime(summary: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)

    ax.plot(summary["lead_days"], summary["bias_tmax"], color=COLOR_OVERALL,
            linewidth=1.5, linestyle="--", marker="o", markersize=4, label="Overall")
    ax.plot(summary["lead_days"], summary["bias_tmax_normal"], color=COLOR_NORMAL,
            linewidth=2, marker="o", markersize=5, label="Normal days")
    ax.plot(summary["lead_days"], summary["bias_tmax_hot"], color=COLOR_HOT,
            linewidth=2, marker="o", markersize=5, label="Anomalous heat days")

    ax.set_xlabel("Lead time (days)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Mean error, actual − forecast (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Tmax forecast bias by lead time", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(summary["lead_days"])
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def plot_mae_by_leadtime(summary: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)

    ax.plot(summary["lead_days"], summary["mae_tmax"], color=COLOR_NORMAL,
            linewidth=2, marker="o", markersize=5, label="MAE")
    ax.plot(summary["lead_days"], summary["rmse_tmax"], color=COLOR_HOT,
            linewidth=2, marker="o", markersize=5, label="RMSE")

    ax.set_xlabel("Lead time (days)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Error (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Tmax forecast error magnitude by lead time", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(summary["lead_days"])
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/paris_leadtime.parquet")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--climatology-window", type=int, default=7,
                    help="±days around each calendar day used to build the climatological normal")
    p.add_argument("--anomaly-threshold", type=float, default=5.0,
                    help="°C above climatology to classify a day as anomalous heat")
    args = p.parse_args()

    df = pd.read_parquet(args.inp) if args.inp.endswith(".parquet") else pd.read_csv(args.inp, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])

    daily_actual = df[["date", "actual_tmax"]].drop_duplicates("date")
    n_years = (daily_actual["date"].max() - daily_actual["date"].min()).days / 365.25
    if n_years < 6:
        print(f"Note: climatology is built from only ~{n_years:.1f} years of data in this "
              f"dataset (no independent long-term normal available) - treat anomaly "
              f"classification as indicative, not authoritative.")

    daily_actual = add_climatology(daily_actual, args.climatology_window, args.anomaly_threshold)
    df = df.merge(daily_actual[["date", "climatology_tmax", "anomaly_tmax", "is_hot_anomaly"]], on="date")

    summary = summarize(df)
    print(summary.to_string(index=False))

    summary_path = f"{args.out_dir}/metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved metrics to {summary_path}")

    plot_bias_by_leadtime(summary, f"{args.out_dir}/bias_by_leadtime.png")
    plot_mae_by_leadtime(summary, f"{args.out_dir}/mae_rmse_by_leadtime.png")
    print(f"Saved charts to {args.out_dir}/bias_by_leadtime.png and {args.out_dir}/mae_rmse_by_leadtime.png")

    n_hot_total = int(df.drop_duplicates("date")["is_hot_anomaly"].sum())
    n_days_total = df["date"].nunique()
    print(f"\n{n_hot_total} / {n_days_total} days classified as anomalous heat "
          f"(>= {args.anomaly_threshold}°C above climatology, ±{args.climatology_window}-day window)")


if __name__ == "__main__":
    main()
