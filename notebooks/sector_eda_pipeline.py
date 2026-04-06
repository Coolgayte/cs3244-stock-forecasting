from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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

    # Single groupby computes both chart metrics efficiently.
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

    # Compute per-stock volatility, keep sector label
    vol_df = (
        panel_df.groupby(["sector", "ticker"], as_index=False)["daily_return"]
        .std()
        .rename(columns={"daily_return": "volatility"})
        .sort_values(["sector", "ticker"])   # sorted by sector first -> same-sector bars adjacent
    )
    # Ordered ticker list (sector-grouped) for x-axis
    ticker_order = vol_df["ticker"].tolist()

    # Sector average volatility (for horizontal reference lines)
    sector_avg = vol_df.groupby("sector")["volatility"].mean()

    # X-index boundaries per sector (to span the dashed line across only that sector's bars)
    sector_x_ranges = {}
    for sector, group in vol_df.groupby("sector", sort=False):
        idxs = [ticker_order.index(t) for t in group["ticker"]]
        sector_x_ranges[sector] = (min(idxs) - 0.4, max(idxs) + 0.4)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=vol_df,
        x="ticker",
        y="volatility",
        hue="sector",
        order=ticker_order,
        dodge=False,
        ax=ax,
    )

    # Draw a dashed horizontal line at the sector-average volatility.
    # Darken the bar color so the line is clearly visible against the bars.
    palette = {
        handle.get_label(): handle.get_facecolor()
        for handle in ax.legend_.legend_handles
    }
    for sector, avg_val in sector_avg.items():
        bar_color = palette.get(sector, "gray")
        # Darken by scaling RGB channels to 50% of original
        r, g, b, *_ = mcolors.to_rgba(bar_color)
        line_color = (r * 0.5, g * 0.5, b * 0.5)
        x_start, x_end = sector_x_ranges[sector]
        ax.hlines(
            y=avg_val,
            xmin=x_start,
            xmax=x_end,
            colors=line_color,
            linestyle="--",
            lw=2.5,
            alpha=0.9,
        )
        ax.text(
            x=(x_start + x_end) / 2,
            y=avg_val*1.02,
            s=f"{sector} Avg",
            color=line_color,
            va="bottom",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set(
        title="Volatility by Stock (Std Dev of Daily Returns)",
        xlabel="Ticker",
        ylabel="Volatility",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Sector", loc="upper right")
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
        default=project_root / "data" / "raw" / "sector_stocks",
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



