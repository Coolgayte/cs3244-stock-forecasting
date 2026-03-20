
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Define project paths relative to this script so teammates can run it from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "top_volume_stocks"
OUT_DIR = PROJECT_ROOT / "data"
ENG_DIR = OUT_DIR / "top_volume_stocks_eng_v2"
BUCKET_RAW_PATH = OUT_DIR / "bucket_top_vol_v2_2009_2017.csv"
BUCKET_ENG_PATH = OUT_DIR / "bucket_top_vol_eng_v2_2009_2017.csv"
RUN_SUMMARY_PATH = OUT_DIR / "run_summary_v2.csv"

# Warm-up periods used by indicators
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
EMA_SHORT = 50
EMA_LONG = 200
REQUIRED_COLS = {"Date", "Close", "Volume"}
START_DATE = pd.Timestamp("2009-01-01")
END_DATE = pd.Timestamp("2017-12-31")
MIN_RETAINED_FRACTION = 0.05


def normalize_ticker(file_path: Path) -> str:
    """Strip exchange suffixes and uppercase the ticker symbol."""
    # Extract leading alphanumeric ticker portion before any non-alphanumeric characters or suffixes.
    match = re.match(r"([A-Za-z0-9]+)", file_path.name)
    ticker = match.group(1) if match else file_path.stem
    return ticker.upper()


def read_stock_file(file_path: Path) -> pd.DataFrame:
    """Read a single stock file (.csv or .txt) and return a DataFrame."""
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        # Parse dates so time-based calculations work as expected.
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def missing_required_columns(df: pd.DataFrame) -> List[str]:
    # Check for core fields needed for indicators and validation.
    return [col for col in REQUIRED_COLS if col not in df.columns]


def drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    # Stable sort ensures deterministic ordering before deduplication.
    ordered = df.sort_values("Date", kind="mergesort")
    return ordered.drop_duplicates(subset=["Date"], keep="last")


def compute_log_return(df: pd.DataFrame) -> pd.Series:
    # Use natural log to stabilize variance and align with many ML models.
    return np.log(df["Close"] / df["Close"].shift(1))


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    # Calculate price changes between consecutive days.
    delta = close.diff()

    # Separate gains and losses; losses are stored as positive numbers for smoothing.
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Exponential smoothing with alpha=1/period (not SMA-seeded Wilder initialization).
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series) -> pd.DataFrame:
    # Exponential moving averages capture momentum with different speeds (12 fast vs 26 slow).
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    })


def compute_bollinger(close: pd.Series) -> pd.DataFrame:
    # 20-day simple moving average represents the mid-band; std measures typical deviation.
    mid = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std()

    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower) / mid

    return pd.DataFrame({
        "bb_mid": mid,
        "bb_std": std,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
    })


def compute_ema_context(close: pd.Series) -> pd.DataFrame:
    # Long and short EMAs capture trend context; ratio summarizes their relative position.
    ema_50 = close.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_200 = close.ewm(span=EMA_LONG, adjust=False).mean()
    ratio = ema_50 / ema_200

    return pd.DataFrame({
        "ema_50": ema_50,
        "ema_200": ema_200,
        "ema_50_200_ratio": ratio,
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy to avoid mutating the caller's DataFrame.
    out = df.copy()

    # Indicators expect chronological order with one row per date.
    out = drop_duplicate_dates(out)

    # Core log return (first row becomes NaN by definition due to shift).
    out["log_return"] = compute_log_return(out)

    # RSI block with exponential smoothing.
    out["rsi_14"] = compute_rsi(out["Close"], RSI_PERIOD)

    # MACD block.
    macd_df = compute_macd(out["Close"])
    out = pd.concat([out, macd_df], axis=1)

    # Bollinger Bands block (rolling window introduces warm-up NaNs).
    bb_df = compute_bollinger(out["Close"])
    out = pd.concat([out, bb_df], axis=1)

    # EMA context block (ewm starts immediately, so no warm-up NaNs beyond initial data).
    ema_df = compute_ema_context(out["Close"])
    out = pd.concat([out, ema_df], axis=1)

    return out


def process_bucket(raw_dir: Path = RAW_DIR) -> None:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {raw_dir}")

    # Collect all stock files (csv or txt) in the target bucket directory.
    stock_files: List[Path] = [p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}]
    if len(stock_files) != 20:
        print(f"Warning: expected 20 stocks in bucket, found {len(stock_files)}")

    ENG_DIR.mkdir(parents=True, exist_ok=True)

    raw_frames: List[pd.DataFrame] = []
    eng_frames: List[pd.DataFrame] = []
    summary_rows: List[dict] = []

    for file_path in sorted(stock_files):
        ticker = normalize_ticker(file_path)
        print(f"Processing {file_path.name} -> ticker {ticker} ...")

        stock_df = read_stock_file(file_path)
        missing_cols = missing_required_columns(stock_df)
        raw_rows = len(stock_df)

        if missing_cols:
            print(f"Skipping {file_path.name}: missing required columns {missing_cols}")
            summary_rows.append({
                "ticker": ticker,
                "raw_rows": raw_rows,
                "rows_after_dedup": 0,
                "duplicate_count": 0,
                "output_file": "",
                "min_date": "",
                "max_date": "",
                "missing_required_cols": ",".join(missing_cols),
            })
            continue

        # Use cleaned ticker consistently.
        stock_df["ticker"] = ticker

        # Enforce stable sorting and deduplicate by date.
        stock_df = drop_duplicate_dates(stock_df)
        rows_after = len(stock_df)
        duplicate_count = raw_rows - rows_after

        # Filter to the approved research window before any feature calculations.
        in_window = stock_df[
            (stock_df["Date"] >= START_DATE) & (stock_df["Date"] <= END_DATE)
        ]
        retained_fraction = (len(in_window) / raw_rows) if raw_rows else 0.0
        if len(in_window) == 0 or retained_fraction < MIN_RETAINED_FRACTION:
            warning_msg = (
                f"{ticker}: only {len(in_window)} rows remain in 2009-2017 window "
                f"({retained_fraction:.2%} of {raw_rows}); aborting for data quality."
            )
            print(f"ERROR: {warning_msg}")
            raise SystemExit(warning_msg)
        stock_df = in_window.reset_index(drop=True)

        # Save the cleaned raw data with ticker for bucket-level aggregation later.
        raw_frames.append(stock_df.copy())

        # Engineer Version 2 indicators.
        eng_df = engineer_features(stock_df)
        eng_frames.append(eng_df.copy())

        # Save per-stock engineered output.
        out_path = ENG_DIR / f"{ticker}_eng_v2.csv"
        eng_df.to_csv(out_path, index=False)

        # Collect run summary metadata for this ticker.
        date_series = stock_df["Date"].dropna()
        min_date = date_series.min().date().isoformat() if not date_series.empty else ""
        max_date = date_series.max().date().isoformat() if not date_series.empty else ""

        summary_rows.append({
            "ticker": ticker,
            "raw_rows": raw_rows,
            "rows_after_dedup": rows_after,
            "duplicate_count": duplicate_count,
            "output_file": out_path.name,
            "min_date": min_date,
            "max_date": max_date,
            "missing_required_cols": "",
        })

    if not eng_frames or not raw_frames:
        print("No valid stock files processed; bucket outputs were not created.")
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(RUN_SUMMARY_PATH, index=False)
        return

    # Combine raw bucket (vertically) and save.
    bucket_raw = pd.concat(raw_frames, axis=0, ignore_index=True)
    bucket_raw = bucket_raw.sort_values(["ticker", "Date"], kind="mergesort")
    bucket_raw.to_csv(BUCKET_RAW_PATH, index=False)

    # Combine engineered bucket and save.
    bucket_eng = pd.concat(eng_frames, axis=0, ignore_index=True)
    bucket_eng = bucket_eng.sort_values(["ticker", "Date"], kind="mergesort")
    bucket_eng.to_csv(BUCKET_ENG_PATH, index=False)

    # Write run summary for quick inspection.
    pd.DataFrame(summary_rows).to_csv(RUN_SUMMARY_PATH, index=False)

    # Print quick diagnostics for teammates.
    print("\n=== bucket_top_vol_eng_v2 HEAD ===")
    print(bucket_eng.head())
    print("\n=== bucket_top_vol_eng_v2 INFO ===")
    bucket_eng.info()
    print("\n=== bucket_top_vol_eng_v2 NaN counts ===")
    print(bucket_eng.isna().sum())


if __name__ == "__main__":
    process_bucket()
