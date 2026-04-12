from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Keep the pipeline anchored to the agreed project folders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "top_volume_stocks"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
ENG_DIR = OUT_DIR / "top_volume_stocks_eng"
BUCKET_RAW_PATH = OUT_DIR / "bucket_top_vol_2009_2017.csv"
BUCKET_ENG_PATH = OUT_DIR / "bucket_top_vol_eng_2009_2017.csv"
RUN_SUMMARY_PATH = OUT_DIR / "run_summary.csv"

# Window settings for the approved research period.
START_DATE = pd.Timestamp("2009-01-01")
END_DATE = pd.Timestamp("2017-12-31")
MIN_RETAINED_FRACTION = 0.05

# Indicator parameters.
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
ATR_PERIOD = 14
EMA_SHORT = 50
EMA_LONG = 200
REQUIRED_COLS = {"Date", "Close", "Volume"}

# Keep only representative features from each indicator family.
ENGINEERED_COLS = [
    "log_return",
    "rsi_14",
    "macd_line",
    "macd_hist",
    "ema_50_200_ratio",
    "obv",
    "vwap",
    "bb_width",
    "atr_14",
]


def normalize_ticker(file_path: Path) -> str:
    """Strip exchange suffixes and uppercase the ticker symbol."""
    match = re.match(r"([A-Za-z0-9]+)", file_path.name)
    ticker = match.group(1) if match else file_path.stem
    return ticker.upper()


def read_stock_file(file_path: Path) -> pd.DataFrame:
    """Read a single stock file (.csv or .txt) and parse the date column."""
    df = pd.read_csv(file_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def missing_required_columns(df: pd.DataFrame) -> List[str]:
    """Return any required columns that are missing from the input file."""
    return [col for col in REQUIRED_COLS if col not in df.columns]


def drop_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Use a stable sort so duplicate-date handling stays deterministic."""
    ordered = df.sort_values("Date", kind="mergesort")
    return ordered.drop_duplicates(subset=["Date"], keep="last")


def compute_log_return(close: pd.Series) -> pd.Series:
    """Log return is a compact price-change feature for downstream models."""
    return np.log(close / close.shift(1))


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI captures short-horizon momentum using smoothed gains and losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> pd.DataFrame:
    """Keep only representative MACD outputs to limit redundancy."""
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_hist": macd_hist,
        }
    )


def compute_ema_context(close: pd.Series) -> pd.Series:
    """Use the short/long EMA ratio as the single trend-context feature."""
    ema_50 = close.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_200 = close.ewm(span=EMA_LONG, adjust=False).mean()
    return ema_50 / ema_200


def compute_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add OBV and VWAP to capture participation and price-volume interaction."""
    close_delta = df["Close"].diff()
    direction = np.sign(close_delta).fillna(0)
    obv = (direction * df["Volume"]).cumsum()

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_price_volume = (typical_price * df["Volume"]).cumsum()
    cumulative_volume = df["Volume"].cumsum().replace(0, np.nan)
    vwap = cumulative_price_volume / cumulative_volume

    return pd.DataFrame(
        {
            "obv": obv,
            "vwap": vwap,
        }
    )


def compute_bollinger(close: pd.Series) -> pd.DataFrame:
    """Compress Bollinger bands into a single width feature to avoid redundancy."""
    mid = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std()
    upper = mid + (2 * std)
    lower = mid - (2 * std)
    width = (upper - lower) / mid.replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_width": width,
        }
    )


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """ATR uses true range decomposition to measure recent volatility."""
    prev_close = df["Close"].shift(1)
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - prev_close).abs()
    low_prev_close = (df["Low"] - prev_close).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the pruned feature set, then drop rolling-window NaN rows."""
    out = drop_duplicate_dates(df.copy())

    # Base return and momentum features.
    out["log_return"] = compute_log_return(out["Close"])
    out["rsi_14"] = compute_rsi(out["Close"], RSI_PERIOD)
    out["ema_50_200_ratio"] = compute_ema_context(out["Close"])

    # MACD family: keep the line and histogram, drop the redundant signal line.
    out = pd.concat([out, compute_macd(out["Close"])], axis=1)

    # Volume features use both price and volume columns, so compute them together.
    out = pd.concat([out, compute_volume_indicators(out)], axis=1)

    # Volatility block keeps two representatives: Bollinger width and ATR.
    out = pd.concat([out, compute_bollinger(out["Close"])], axis=1)
    out["atr_14"] = compute_atr(out, ATR_PERIOD)

    # Remove only the leading rows made invalid by indicator windows.
    required_for_model = list(REQUIRED_COLS) + ENGINEERED_COLS
    out = out.dropna(subset=required_for_model).reset_index(drop=True)
    return out


def process_bucket(raw_dir: Path = RAW_DIR) -> None:
    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {raw_dir}")

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
            summary_rows.append(
                {
                    "ticker": ticker,
                    "raw_rows": raw_rows,
                    "rows_after_dedup": 0,
                    "rows_after_cleaning": 0,
                    "duplicate_count": 0,
                    "output_file": "",
                    "min_date": "",
                    "max_date": "",
                    "missing_required_cols": ",".join(missing_cols),
                }
            )
            continue

        # Preserve the source columns and attach the normalized ticker before FE.
        stock_df["ticker"] = ticker
        stock_df = drop_duplicate_dates(stock_df)
        rows_after_dedup = len(stock_df)
        duplicate_count = raw_rows - rows_after_dedup

        # Apply the strict research window before any feature calculations.
        stock_df = stock_df[
            (stock_df["Date"] >= START_DATE) & (stock_df["Date"] <= END_DATE)
        ].reset_index(drop=True)
        retained_fraction = (len(stock_df) / raw_rows) if raw_rows else 0.0
        if len(stock_df) == 0 or retained_fraction < MIN_RETAINED_FRACTION:
            warning_msg = (
                f"{ticker}: only {len(stock_df)} rows remain in 2009-2017 window "
                f"({retained_fraction:.2%} of {raw_rows}); aborting for data quality."
            )
            print(f"ERROR: {warning_msg}")
            raise SystemExit(warning_msg)

        raw_frames.append(stock_df.copy())

        eng_df = engineer_features(stock_df)
        eng_frames.append(eng_df.copy())

        out_path = ENG_DIR / f"{ticker}_eng.csv"
        eng_df.to_csv(out_path, index=False)

        date_series = stock_df["Date"].dropna()
        min_date = date_series.min().date().isoformat() if not date_series.empty else ""
        max_date = date_series.max().date().isoformat() if not date_series.empty else ""

        summary_rows.append(
            {
                "ticker": ticker,
                "raw_rows": raw_rows,
                "rows_after_dedup": rows_after_dedup,
                "rows_after_cleaning": len(eng_df),
                "duplicate_count": duplicate_count,
                "output_file": out_path.name,
                "min_date": min_date,
                "max_date": max_date,
                "missing_required_cols": "",
            }
        )

    if not eng_frames or not raw_frames:
        print("No valid stock files processed; bucket outputs were not created.")
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(RUN_SUMMARY_PATH, index=False)
        return

    bucket_raw = pd.concat(raw_frames, axis=0, ignore_index=True)
    bucket_raw = bucket_raw.sort_values(["ticker", "Date"], kind="mergesort")
    bucket_raw.to_csv(BUCKET_RAW_PATH, index=False)

    bucket_eng = pd.concat(eng_frames, axis=0, ignore_index=True)
    bucket_eng = bucket_eng.sort_values(["ticker", "Date"], kind="mergesort")
    bucket_eng.to_csv(BUCKET_ENG_PATH, index=False)

    pd.DataFrame(summary_rows).to_csv(RUN_SUMMARY_PATH, index=False)

    print("\n=== bucket_top_vol_eng HEAD ===")
    print(bucket_eng.head())
    print("\n=== bucket_top_vol_eng INFO ===")
    bucket_eng.info()
    print("\n=== bucket_top_vol_eng NaN counts ===")
    print(bucket_eng.isna().sum())


if __name__ == "__main__":
    process_bucket()
