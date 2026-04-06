from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


@dataclass
class BucketEdaConfig:
    data_dir: Path
    output_dir: Path
    start_date: str = "2009-01-01"
    end_date: str = "2017-12-31"
    dpi: int = 300


def prepare_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    stock_df = df.copy()

    stock_df["Date"] = pd.to_datetime(stock_df["Date"], errors="coerce")
    stock_df["Close"] = pd.to_numeric(stock_df["Close"], errors="coerce")
    stock_df = stock_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    if stock_df.empty:
        return pd.DataFrame(columns=["Date", "Close", "normalized_price", "daily_return"])

    first_close = stock_df["Close"].iloc[0]
    stock_df["normalized_price"] = stock_df["Close"] / first_close
    stock_df["daily_return"] = stock_df["Close"].pct_change()
    return stock_df[["Date", "normalized_price", "daily_return"]]


def load_sector_panel(cfg: BucketEdaConfig) -> pd.DataFrame:
    # Expected structure: data_dir/<sector_name>/*.csv
    sector_dirs = sorted([p for p in cfg.data_dir.iterdir() if p.is_dir()])
    if not sector_dirs:
        raise FileNotFoundError(f"No sector folders found in {cfg.data_dir}")

    frames: List[pd.DataFrame] = []
    for sector_dir in sector_dirs:
        sector_name = sector_dir.name
        csv_files = list(sector_dir.glob("*.csv"))
        print(f"Folder {sector_dir.name} has {len(csv_files)} file CSV.")
        for csv_path in sorted(sector_dir.glob("*.csv")):
            stock_df = prepare_stock_data(pd.read_csv(csv_path))
            if stock_df.empty:
                continue

            stock_df["sector"] = sector_name
            stock_df["ticker"] = csv_path.stem.upper()
            frames.append(stock_df)

    if not frames:
        raise ValueError(
            "No usable CSV data found. Check folder structure and required columns Date/Close."
        )

    panel_df = pd.concat(frames, ignore_index=True)
    return panel_df.sort_values(["Date", "sector", "ticker"]).reset_index(drop=True)


def build_sector_daily(
    panel_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    panel_df = panel_df[(panel_df["Date"] >= start_ts) & (panel_df["Date"] <= end_ts)].copy()
    if panel_df.empty:
        raise ValueError("No rows left after date filtering. Check start_date/end_date.")

    sector_daily_all = (
        panel_df.groupby(["Date", "sector"], as_index=False)
        .agg(
            avg_normalized_price=("normalized_price", "mean"),
            sector_return=("daily_return", "mean"),
        )
        .sort_values(["Date", "sector"])
    )

    # Keep intersection dates where all sectors are present to make trends comparable.
    counts = sector_daily_all.groupby("Date")["sector"].nunique()
    all_sector_count = sector_daily_all["sector"].nunique()
    intersection_dates = counts[counts == all_sector_count].index
    sector_daily_intersection = sector_daily_all[
        sector_daily_all["Date"].isin(intersection_dates)
    ].copy()

    if sector_daily_intersection.empty:
        raise ValueError("No common dates across sectors after filtering.")
    return sector_daily_all, sector_daily_intersection


def plot_avg_normalized_price_by_sector(
    sector_daily: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    out_path = output_dir / "sector_avg_normalized_price_intersection_trend.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=sector_daily,
        x="Date",
        y="avg_normalized_price",
        hue="sector",
        linewidth=2,
        ax=ax,
    )
    ax.set(
        title="Average Normalized Price by Sector",
        xlabel="Date",
        ylabel="Average Normalized Price",
    )
    ax.grid(alpha=0.25)
    ax.legend(title="Sector", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def plot_stock_volatility_by_sector(
    panel_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    out_path = output_dir / "stock_volatility_by_sector_combined.png"

    # Compute per-stock volatility and keep one bar per stock.
    stock_vol_df = (
        panel_df.groupby(["sector", "ticker"], as_index=False)["daily_return"]
        .std()
        .rename(columns={"daily_return": "volatility"})
        .dropna(subset=["volatility"])
        .sort_values(["sector", "volatility", "ticker"], ascending=[True, False, True])
    )

    if stock_vol_df.empty:
        raise ValueError("No valid volatility values to plot.")

    sectors = stock_vol_df["sector"].unique().tolist()
    palette_colors = sns.color_palette("Set2", n_colors=len(sectors))
    sector_palette = dict(zip(sectors, palette_colors))

    x_positions = []
    vol_values = []
    bar_colors = []
    sector_centers = {}
    sector_boundaries = []
    sector_ranges = {}

    x_cursor = 0
    sector_gap = 2
    for sector, group in stock_vol_df.groupby("sector", sort=False):
        n_bars = len(group)
        xs = list(range(x_cursor, x_cursor + n_bars))

        x_positions.extend(xs)
        vol_values.extend(group["volatility"].tolist())
        bar_colors.extend([sector_palette[sector]] * n_bars)
        sector_centers[sector] = (xs[0] + xs[-1]) / 2
        sector_ranges[sector] = (xs[0] - 0.4, xs[-1] + 0.4)

        sector_boundaries.append(xs[-1] + 0.5)
        x_cursor = xs[-1] + 1 + sector_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_positions, vol_values, color=bar_colors, width=0.85, alpha=0.9)

    sector_avg = stock_vol_df.groupby("sector")["volatility"].mean()
    for sector, avg_vol in sector_avg.items():
        x_start, x_end = sector_ranges[sector]
        ax.hlines(
            y=avg_vol,
            xmin=x_start,
            xmax=x_end,
            color="black",
            linestyle="--",
            linewidth=1.4,
            alpha=0.75,
        )

    ax.set_xticks(list(sector_centers.values()))
    ax.set_xticklabels(list(sector_centers.keys()), rotation=20)

    legend_handles = [Patch(facecolor=sector_palette[s], label=s) for s in sectors]
    legend_handles.append(
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.6, label="Sector Avg")
    )
    ax.legend(handles=legend_handles, title="Sector", loc="upper right")

    ax.set(
        title="Stock Volatility by Sector (Each Bar = One Stock)",
        xlabel="Sector",
        ylabel="Volatility",
    )
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def run_pipeline(cfg: BucketEdaConfig) -> Tuple[Path, Path]:
    sns.set_style("whitegrid")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    panel_df = load_sector_panel(cfg)
    sector_daily_all, sector_daily_intersection = build_sector_daily(
        panel_df,
        cfg.start_date,
        cfg.end_date,
    )
    chart_1 = plot_avg_normalized_price_by_sector(
        sector_daily_intersection,
        cfg.output_dir,
        cfg.dpi,
    )
    chart_2 = plot_stock_volatility_by_sector(panel_df, cfg.output_dir, cfg.dpi)
    return chart_1, chart_2


def parse_args() -> BucketEdaConfig:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Simple reproducible sector-folder EDA pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" ,
        help="Root folder containing one folder per sector, each with CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "EDA" / "sector_visualisations",
    )
    parser.add_argument("--start-date", type=str, default="2009-01-01")
    parser.add_argument("--end-date", type=str, default="2017-12-31")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    return BucketEdaConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        dpi=args.dpi,
    )


def main() -> None:
    chart_1, chart_2 = run_pipeline(parse_args())
    print("Saved chart 1:", chart_1)
    print("Saved chart 2:", chart_2)


if __name__ == "__main__":
    main()



