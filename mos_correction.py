"""
Simulate standard MOS-style bias correction and test whether the hot-day
underforecast bias survives it, or whether it's just the kind of bias
ordinary post-processing would already remove.

Two correction methods, both fit on a chronological training window and
evaluated out-of-sample on a held-out test window (no leakage):
  - constant: corrected = forecast + mean_bias_train(lead_days)
    (the simplest possible bias correction - subtract the average error)
  - linear: corrected = a + b * forecast, fit by regressing actual on
    forecast per lead_days (closer to how real MOS works - it can absorb
    bias that varies with the forecast value itself, e.g. if hot days
    already show up as elevated forecasts)

If the hot-vs-normal bias gap survives both corrections, that's evidence
the effect isn't just an easily-removable static bias.

Usage:
    python mos_correction.py --in data/paris_leadtime.parquet --out-dir data
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze import add_climatology, style_axes, COLOR_NORMAL, COLOR_HOT, COLOR_MUTED, COLOR_TEXT


def chronological_split(df: pd.DataFrame, train_frac: float):
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]
    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()


def fit_and_apply(train: pd.DataFrame, test: pd.DataFrame, lead_days: list[int]) -> pd.DataFrame:
    test = test.copy()
    test["forecast_const"] = np.nan
    test["forecast_linear"] = np.nan

    for lead in lead_days:
        tr = train[train["lead_days"] == lead].dropna(subset=["forecast_tmax", "actual_tmax"])
        te_mask = test["lead_days"] == lead

        mean_bias = (tr["actual_tmax"] - tr["forecast_tmax"]).mean()
        test.loc[te_mask, "forecast_const"] = test.loc[te_mask, "forecast_tmax"] + mean_bias

        b, a = np.polyfit(tr["forecast_tmax"], tr["actual_tmax"], 1)
        test.loc[te_mask, "forecast_linear"] = a + b * test.loc[te_mask, "forecast_tmax"]

    test["error_raw"] = test["actual_tmax"] - test["forecast_tmax"]
    test["error_const"] = test["actual_tmax"] - test["forecast_const"]
    test["error_linear"] = test["actual_tmax"] - test["forecast_linear"]
    return test


def compare_hot_vs_normal(test: pd.DataFrame, error_col: str) -> pd.DataFrame:
    rows = []
    for lead, g in test.groupby("lead_days"):
        g = g.dropna(subset=[error_col])
        hot = g[g["is_hot_anomaly"]][error_col]
        normal = g[~g["is_hot_anomaly"]][error_col]
        if len(hot) >= 2 and len(normal) >= 2:
            t, p = stats.ttest_ind(hot, normal, equal_var=False)
        else:
            t, p = float("nan"), float("nan")
        rows.append({
            "lead_days": lead, "n_hot": len(hot), "n_normal": len(normal),
            "bias_hot": hot.mean(), "bias_normal": normal.mean(),
            "gap": hot.mean() - normal.mean(), "p_value": p,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/paris_leadtime.parquet")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--climatology-window", type=int, default=7)
    p.add_argument("--anomaly-threshold", type=float, default=5.0)
    args = p.parse_args()

    df = pd.read_parquet(args.inp) if args.inp.endswith(".parquet") else pd.read_csv(args.inp, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    lead_days = sorted(df["lead_days"].unique())

    daily_actual = df[["date", "actual_tmax"]].drop_duplicates("date")
    daily_actual = add_climatology(daily_actual, args.climatology_window, args.anomaly_threshold)
    df = df.merge(daily_actual[["date", "is_hot_anomaly"]], on="date")

    train, test = chronological_split(df, args.train_frac)
    print(f"Train: {train['date'].min().date()} .. {train['date'].max().date()} "
          f"({train['date'].nunique()} days)")
    print(f"Test:  {test['date'].min().date()} .. {test['date'].max().date()} "
          f"({test['date'].nunique()} days, "
          f"{int(test.drop_duplicates('date')['is_hot_anomaly'].sum())} hot-anomaly days)")

    test = fit_and_apply(train, test, lead_days)

    results = {}
    for label, col in [("raw", "error_raw"), ("const-corrected", "error_const"),
                        ("linear-corrected", "error_linear")]:
        summary = compare_hot_vs_normal(test, col)
        results[label] = summary
        print(f"\n=== {label} (test set, out-of-sample) ===")
        print(summary.to_string(index=False))

    all_rows = []
    for label, summary in results.items():
        s = summary.copy()
        s.insert(0, "method", label)
        all_rows.append(s)
    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = f"{args.out_dir}/mos_correction_summary.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    style_axes(ax)
    styles = {"raw": ("--", COLOR_MUTED), "const-corrected": ("-", COLOR_NORMAL),
              "linear-corrected": ("-", COLOR_HOT)}
    for label, summary in results.items():
        ls, color = styles[label]
        ax.plot(summary["lead_days"], summary["bias_hot"], linestyle=ls, color=color,
                linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xlabel("Lead time (days)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Hot-anomaly-day bias, actual − forecast (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title("Does MOS-style correction remove the heat-day bias?", color=COLOR_TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(lead_days)
    fig.tight_layout()
    out_png = f"{args.out_dir}/mos_correction_hotbias.png"
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"Saved chart to {out_png}")


if __name__ == "__main__":
    main()
