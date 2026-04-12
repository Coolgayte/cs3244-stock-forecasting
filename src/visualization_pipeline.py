from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    data_dir: Path = Path("data/top_volume_stocks")
    output_dir: Path = Path("results/plots")
    processed_dir: Path = Path("data/processed")
    date_col: str = "Date"
    close_col: str = "Close"
    rolling_window: int = 21
    figure_dpi: int = 140
    hist_bins: int = 60
    
def load_stock(path: Path, cfg: PlotConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    if cfg.date_col not in df.columns:
        raise ValueError(f"{path.name}: missing '{cfg.date_col}' column")
    if cfg.close_col not in df.columns:
        raise ValueError(f"{path.name}: missing '{cfg.close_col}' column")
    
    df = df[[cfg.date_col, cfg.close_col]].copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df[cfg.close_col] = pd.to_numeric(df[cfg.close_col], errors="coerce")
    
    df = df.dropna(subset=[cfg.date_col, cfg.close_col])
    df = df.drop_duplicates(subset=[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    
    if df.empty: 
        raise ValueError(f"{path.name}: no valid rows after cleaning")
    
    first_close = df[cfg.close_col].iloc[0]
    df["normalized_price"] = df[cfg.close_col] / first_close
    df["daily_return"] = df[cfg.close_col].pct_change()
    df["rolling_volatility"] = (
        df["daily_return"]
        .rolling(window=cfg.rolling_window)
        .std()
    )
    
    return df

def load_all_stocks(cfg: PlotConfig) -> Dict[str, pd.DataFrame]:
    files = sorted(cfg.data_dir.glob("*.csv"))
    
    if not files:
        raise FileNotFoundError(f"No CSV files in {cfg.data_dir}")
    
    stocks: Dict[str, pd.DataFrame] = {}
    for f in files:
        ticker = f.stem.upper()
        try:
            stocks[ticker] = load_stock(f, cfg)
            log.info("Loaded %-6s rows=%d", ticker, len(stocks[ticker]))
        except Exception:
            log.exception("SKIP %s - failed to load", ticker)
            
    return stocks
    
def build_bucket_frames(
    stocks: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for ticker, df in stocks.items():
        tmp = df[["Date", "normalized_price", "daily_return", "rolling_volatility"]].copy()
        tmp["ticker"] = ticker
        frames.append(tmp)
    
    long_df = pd.concat(frames, ignore_index=True).sort_values(["Date", "ticker"])
    returns_wide = long_df.pivot(index="Date", columns="ticker", values="daily_return").sort_index()
    
    return long_df, returns_wide
    
def _save(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_bucket(long_df: pd.DataFrame, returns_wide: pd.DataFrame, cfg: PlotConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Define storage directory
    bdir = cfg.output_dir / "bucket"
    
    # Common dates where all tickers exist
    num_tickers = long_df["ticker"].nunique()
    date_counts = long_df.groupby("Date")["ticker"].count()
    common_return_dates = returns_wide.dropna(how="any").index
    # common_dates = date_counts[date_counts == num_tickers].index
    
    
    # Apply common-date filter ONCE and reuse for all 5 plots
    intersect_df = long_df[long_df["Date"].isin(common_return_dates)].copy()
    returns_common = returns_wide.loc[returns_wide.index.intersection(common_return_dates)].copy()

    
    # --- 1. Average Normalized Price (Intersection Trend) ---
    avg_norm = intersect_df.groupby("Date")["normalized_price"].mean().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(avg_norm.index, avg_norm.values, linewidth=2)
    ax.set(
        title="Bucket Trend = Average Normalized Price",
        xlabel="Date",
        ylabel="Avg Normalized Price",
    )
    ax.grid(alpha=0.25)
    _save(bdir / "avg_normalized_price_bucket_trend.png", cfg.figure_dpi)
    
    # --- 2. Equal-Weighted Cumulative Return ---
    ew_daily = returns_common.mean(axis=1, skipna=True).fillna(0.0)
    ew_cum = (1.0 + ew_daily).cumprod()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ew_cum.index, ew_cum.values, linewidth=2)
    ax.set(
        title="Equal-Weighted Cumulative Return (Bucket)",
        xlabel="Date",
        ylabel="Portfolio Growth (start = 1)",
    )
    ax.grid(alpha=0.5)
    _save(bdir / "equal_weighted_cumulative_return.png", cfg.figure_dpi)
    
    # --- 3. Boxplot of Returns (Distribution & Outliers) ---
    returns_long = (
        returns_common
        .stack(future_stack=True)
        .rename("daily_return")
        .reset_index()
        .dropna(subset=["daily_return"])
    )
    
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=returns_long, x="ticker", y="daily_return", showfliers=True, ax=ax)
    ax.set(
        title="Boxplot of Daily Returns by Stock",
        xlabel="Ticker",
        ylabel="Daily Return",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    _save(bdir / "boxplot_returns_all_stocks.png", cfg.figure_dpi)
    
    # --- 4. Bar Chart of Volatility (Risk Ranking) ---
    vol = returns_common.std(skipna=True).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(vol.index, vol.values, color="steelblue")
    ax.set(
        title="Volatility by Stock (Std Dev of Daily Returns)",
        xlabel="Ticker",
        ylabel="Volatility",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.2)
    _save(bdir / "bar_volatility_per_stock.png", cfg.figure_dpi)
    
    # --- 5. Correlation Heatmap (Interdependency) ---
    corr = returns_common.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.78},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Daily Returns)")
    _save(bdir / "correlation_heatmap_returns.png", cfg.figure_dpi)
    
    return intersect_df, returns_common
    
def save_processed(
    long_df: pd.DataFrame,
    returns_wide: pd.DataFrame,
    long_df_common: pd.DataFrame,
    returns_common: pd.DataFrame,
    cfg: PlotConfig,
) -> None:
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    
    long_df.to_csv(cfg.processed_dir / "bucket_long_panel_full.csv", index=False)
    returns_wide.to_csv(cfg.processed_dir / "bucket_returns_wide_full.csv")
    long_df_common.to_csv(cfg.processed_dir / "bucket_long_panel_common.csv", index=False)
    returns_common.to_csv(cfg.processed_dir / "bucket_returns_wide_common.csv")
    log.info("Processed artifacts -> %s", cfg.processed_dir)
    
def run_pipeline(cfg: PlotConfig | None = None) -> None:
    cfg = cfg or PlotConfig()
    sns.set_style("whitegrid")
    
    log.info("=== LOAD === data_dir=%s", cfg.data_dir)
    stocks = load_all_stocks(cfg)
    if not stocks:
        log.error("No stocks loaded - aborting.")
        return
    
    log.info("=== BUILD BUCKET FRAMES ===")
    long_df, returns_wide = build_bucket_frames(stocks)
    
    log.info("=== BUCKET PLOTS ===")
    long_df_common, returns_common = plot_bucket(long_df, returns_wide, cfg)
    
    log.info("=== SAVE PROCESSED ===")
    save_processed(long_df, returns_wide, long_df_common, returns_common, cfg)
    
    log.info(
        "Pipeline complete: 5 bucket = %d plots",
        5,
    )
    
def _parse_args() -> PlotConfig:
    p = argparse.ArgumentParser(description="Stock visualization pipeline")
    p.add_argument("--data-dir", type=Path, default=PlotConfig.data_dir)
    p.add_argument("--output-dir", type=Path, default=PlotConfig.output_dir)
    p.add_argument("--processed-dir", type=Path, default=PlotConfig.processed_dir)
    p.add_argument("--rolling-window", type=int, default=PlotConfig.rolling_window)
    p.add_argument("--dpi", type=int, default=PlotConfig.figure_dpi)
    p.add_argument("--hist-bins", type=int, default=PlotConfig.hist_bins)
    args = p.parse_args()
    
    return PlotConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        rolling_window=args.rolling_window,
        figure_dpi=args.dpi,
        hist_bins=args.hist_bins
    )
    
if __name__ == "__main__":
    run_pipeline(_parse_args())
    
    