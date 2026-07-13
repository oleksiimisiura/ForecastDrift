"""
Approximate what the forecast error looks like at 10-day lead, since Open-Meteo
has no historical archive at that horizon (see fetch_single_run_daily.py's
docstring). Two independent estimates, shown side by side:

1. Physically-motivated extrapolation from the real 1/3/5/7-day data. As lead
   time grows, a forecast's skill decays and it reverts toward the
   climatological normal for that calendar day - that's the well-known
   "regression to climatology" behavior of NWP models at long lead times. So
   bias on hot-anomaly days shouldn't grow forever: it should saturate at the
   mean anomaly of those hot days themselves (the bias you'd get if the
   forecast were simply "today will be like a normal day" - i.e. zero skill).
   That ceiling is computable directly from the dataset, no extrapolation
   needed. We fit a saturating-exponential curve through it, anchored exactly
   at the observed 1-day bias, with only the decay rate fit from the 3/5/7-day
   points - then read off the value at lead=10.

2. The real (if still small) empirical 10-day sample being accumulated daily
   via fetch_single_run_daily.py / GitHub Actions - a direct measurement,
   not a model, but based on far fewer days.

If the two roughly agree, that's decent evidence the extrapolation is sane.
If they diverge a lot, trust the small real sample less than either - it's
too small to be definitive, and say so.

Usage:
    python extrapolate_10day.py --leadtime data/paris/leadtime.parquet --archive10 data/paris/10day_archive.csv --out-dir data/paris
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import add_climatology, style_axes, COLOR_NORMAL, COLOR_HOT, COLOR_TEXT, COLOR_MUTED

TARGET_LEAD = 10


def fit_saturating(t_obs, y_obs, ceiling):
    """bias(t) = ceiling - (ceiling - y_obs[0]) * exp(-k*(t - t_obs[0])), fit k
    by least squares against the remaining points. Anchored exactly at the
    first (most reliable, highest-n) observed point."""
    t0, y0 = t_obs[0], y_obs[0]
    amp = ceiling - y0

    def predict(k):
        return ceiling - amp * np.exp(-k * (np.array(t_obs) - t0))

    ks = np.linspace(0.001, 3, 3000)
    sse = [np.sum((predict(k) - np.array(y_obs)) ** 2) for k in ks]
    best_k = ks[int(np.argmin(sse))]

    def f(t):
        return ceiling - amp * np.exp(-best_k * (t - t0))

    return f, best_k


def build_leadtime_summary(leadtime_path: str, metric: str, climatology_window: int, anomaly_threshold: float):
    df = pd.read_parquet(leadtime_path)
    df["date"] = pd.to_datetime(df["date"])
    daily_actual = df[["date", "actual_tmax"]].drop_duplicates("date")
    daily_actual = add_climatology(daily_actual, climatology_window, anomaly_threshold)
    df = df.merge(daily_actual[["date", "anomaly_tmax", "is_hot_anomaly"]], on="date")

    err_col = f"error_{metric}"
    rows = []
    for lead, g in df.groupby("lead_days"):
        hot = g[g["is_hot_anomaly"]][err_col].dropna()
        normal = g[~g["is_hot_anomaly"]][err_col].dropna()
        rows.append({"lead_days": lead, "bias_hot": hot.mean(), "bias_normal": normal.mean(),
                     "n_hot": len(hot), "n_normal": len(normal)})
    summary = pd.DataFrame(rows).sort_values("lead_days")

    hot_anomaly_values = daily_actual.loc[daily_actual["is_hot_anomaly"], "anomaly_tmax"]
    normal_anomaly_values = daily_actual.loc[~daily_actual["is_hot_anomaly"], "anomaly_tmax"]
    ceilings = {"hot": hot_anomaly_values.mean(), "normal": normal_anomaly_values.mean()}
    return summary, ceilings


def empirical_10day(archive_path: str, metric: str, climatology_window: int, anomaly_threshold: float):
    arc = pd.read_csv(archive_path, parse_dates=["date"])
    daily_actual = arc[["date", "actual_tmax"]].drop_duplicates("date")
    daily_actual = add_climatology(daily_actual, climatology_window, anomaly_threshold)
    arc = arc.merge(daily_actual[["date", "is_hot_anomaly"]], on="date")

    err_col = f"error_{metric}"
    hot = arc[arc["is_hot_anomaly"]][err_col].dropna()
    normal = arc[~arc["is_hot_anomaly"]][err_col].dropna()
    return {"bias_hot": hot.mean(), "n_hot": len(hot),
            "bias_normal": normal.mean(), "n_normal": len(normal)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leadtime", default="data/paris/leadtime.parquet")
    p.add_argument("--archive10", default="data/paris/10day_archive.csv")
    p.add_argument("--out-dir", default="data/paris")
    p.add_argument("--metric", choices=["tmax", "tmin"], default="tmax")
    p.add_argument("--climatology-window", type=int, default=7)
    p.add_argument("--anomaly-threshold", type=float, default=5.0)
    args = p.parse_args()

    summary, ceilings = build_leadtime_summary(args.leadtime, args.metric, args.climatology_window, args.anomaly_threshold)
    print("Observed (1/3/5/7 days):")
    print(summary.to_string(index=False))
    print(f"\nSaturation ceilings (mean anomaly of each group - the bias you'd get "
          f"at zero forecast skill): hot={ceilings['hot']:.2f}°C, normal={ceilings['normal']:.2f}°C")

    t_obs = summary["lead_days"].tolist()
    f_hot, k_hot = fit_saturating(t_obs, summary["bias_hot"].tolist(), ceilings["hot"])
    f_normal, k_normal = fit_saturating(t_obs, summary["bias_normal"].tolist(), ceilings["normal"])

    extrap_hot = f_hot(TARGET_LEAD)
    extrap_normal = f_normal(TARGET_LEAD)
    print(f"\nExtrapolated to {TARGET_LEAD} days (saturating-exponential fit, "
          f"k_hot={k_hot:.3f}, k_normal={k_normal:.3f}):")
    print(f"  bias_hot  ~ {extrap_hot:+.2f}C")
    print(f"  bias_normal ~ {extrap_normal:+.2f}C")

    try:
        emp = empirical_10day(args.archive10, args.metric, args.climatology_window, args.anomaly_threshold)
        print(f"\nActual empirical 10-day sample (still accumulating daily):")
        print(f"  bias_hot  = {emp['bias_hot']:+.2f}°C (n={emp['n_hot']})")
        print(f"  bias_normal = {emp['bias_normal']:+.2f}°C (n={emp['n_normal']})")
        if emp['n_hot'] < 20:
            print(f"  Note: n_hot={emp['n_hot']} is small - treat this number as very noisy, "
                  f"not a reliable measurement on its own.")
    except FileNotFoundError:
        emp = None
        print(f"\nNo empirical 10-day archive found at {args.archive10}")

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    style_axes(ax)
    t_smooth = np.linspace(1, TARGET_LEAD, 100)
    ax.plot(t_smooth, f_normal(t_smooth), color=COLOR_NORMAL, linewidth=1.5, linestyle="--")
    ax.plot(t_smooth, f_hot(t_smooth), color=COLOR_HOT, linewidth=1.5, linestyle="--")
    ax.scatter(summary["lead_days"], summary["bias_normal"], color=COLOR_NORMAL, s=40, zorder=5, label="Normal days (observed)")
    ax.scatter(summary["lead_days"], summary["bias_hot"], color=COLOR_HOT, s=40, zorder=5, label="Anomalous heat days (observed)")
    ax.scatter([TARGET_LEAD], [extrap_normal], color=COLOR_NORMAL, s=70, marker="D", zorder=5, label="Normal days (extrapolated)")
    ax.scatter([TARGET_LEAD], [extrap_hot], color=COLOR_HOT, s=70, marker="D", zorder=5, label="Anomalous heat days (extrapolated)")
    if emp is not None:
        ax.scatter([TARGET_LEAD], [emp["bias_normal"]], color=COLOR_NORMAL, s=70, marker="x", zorder=6, label=f"Normal days (real sample, n={emp['n_normal']})")
        ax.scatter([TARGET_LEAD], [emp["bias_hot"]], color=COLOR_HOT, s=70, marker="x", zorder=6, label=f"Heat days (real sample, n={emp['n_hot']})")
    ax.axhline(ceilings["hot"], color=COLOR_HOT, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(ceilings["normal"], color=COLOR_NORMAL, linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Lead time (days)", color=COLOR_TEXT, fontsize=10)
    ax.set_ylabel("Mean error, actual − forecast (°C)", color=COLOR_TEXT, fontsize=10)
    ax.set_title(f"{args.metric}: observed vs extrapolated bias, dotted lines = saturation ceiling",
                 color=COLOR_TEXT, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out_png = f"{args.out_dir}/extrapolation_{args.metric}.png"
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    print(f"\nSaved chart to {out_png}")


if __name__ == "__main__":
    main()
